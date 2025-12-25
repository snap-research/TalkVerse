# Copyright 2024-2025 The Alibaba Wan Team Authors. All rights reserved.
import os
import json
import argparse
import warnings
import random
import math
import time
from tqdm import tqdm
from pathlib import Path
import re
from einops import rearrange
from safetensors.torch import save_file, load_file
import numpy as np
from datetime import datetime, timedelta
# Logging imports
try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_AVAILABLE = True
except ImportError:
    TENSORBOARD_AVAILABLE = False
    print("Warning: TensorBoard not available. Install with: pip install tensorboard")

WANDB_AVAILABLE = False
# try:
#     import wandb
#     WANDB_AVAILABLE = True
# except ImportError:
#     WANDB_AVAILABLE = False
#     print("Warning: Weights & Biases not available. Install with: pip install wandb")

# --- Audio and Dataloader ---
from transformers import Wav2Vec2FeatureExtractor
from wan.src import VideoDataset, custom_collate_fn

# --- PyTorch and Distributed Imports ---
import torch
import torch.nn as nn
import torch.distributed as dist
import torch.nn.functional as F
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp.wrap import ModuleWrapPolicy
from torch.distributed.fsdp.api import FullStateDictConfig, FullOptimStateDictConfig, StateDictType
from torch.utils.data.distributed import DistributedSampler

# --- Wan S2V 5B Imports ---
# from wan.s2v_5b import WanS2V_5B
from wan.configs.wan_s2v_5B import s2v_5B
from wan.modules.s2v.model_s2v_5b import WanModel_S2V_5B, WanS2V5BAttentionBlock
from wan.modules.vae2_2 import Wan2_2_VAE
# from wan.schedulers.flow_match import FlowMatchScheduler
from wan.utils.fm_solvers_unipc import FlowUniPCMultistepScheduler

# Suppress warnings
warnings.filterwarnings('ignore')
warnings.filterwarnings(
    "ignore",
    message="`torch.cuda.amp.autocast.*` is deprecated.*",
    category=FutureWarning
)
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ['NCCL_DEBUG'] = 'WARN'
ATTN_QK_TOKENS = ["self_attn.q.", "self_attn.k.", "cross_attn.q.", "cross_attn.k.", "cross_attn.norm_k.", "cross_attn.norm_q.", "cross_attn.o", "self_attn.norm_k.", "self_attn.norm_q.", "self_attn.o", ".norm3.", ".modulation", ".ffn."]

def setup_distributed():
    """Initializes the distributed process group and returns rank and local rank."""
    dist.init_process_group(
        backend="nccl",
        init_method="env://",
        timeout=timedelta(minutes=60)
    )
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    local_rank = int(os.environ['LOCAL_RANK'])
    torch.cuda.set_device(local_rank)
    return rank, local_rank, world_size

def main_print(msg, rank):
    """Prints a message only on the main process."""
    if rank == 0:
        print(msg)


class LoRALinear(nn.Module):
    """
    LoRA wrapper for nn.Linear.
    y = base(x) + scale * (x -> A -> B)
    A: in_features -> r, B: r -> out_features; B is zero-initialized.
    """
    def __init__(self, base_linear: nn.Linear, r: int = 128, alpha: int = 128, dropout: float = 0.0):
        super().__init__()
        assert isinstance(base_linear, nn.Linear)
        self.in_features = base_linear.in_features
        self.out_features = base_linear.out_features
        self.has_bias = base_linear.bias is not None
        self.scale = alpha / float(r) if r > 0 else 0.0

        # Frozen base
        self.base = nn.Linear(self.in_features, self.out_features, bias=self.has_bias)
        with torch.no_grad():
            self.base.weight.copy_(base_linear.weight)
            if self.has_bias:
                self.base.bias.copy_(base_linear.bias)
        for p in self.base.parameters():
            p.requires_grad = False

        # LoRA adapters
        self.lora_A = nn.Linear(self.in_features, r, bias=False)
        self.lora_B = nn.Linear(r, self.out_features, bias=False)
        # init: A with kaiming normal, B zeros => no change at start
        nn.init.kaiming_uniform_(self.lora_A.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B.weight)

        self.dropout = nn.Dropout(dropout) if dropout and dropout > 0 else nn.Identity()

    @classmethod
    def from_linear(cls, lin: nn.Linear, r=128, alpha=128, dropout=0.0):
        return cls(lin, r=r, alpha=alpha, dropout=dropout)

    def forward(self, x):
        return self.base(x) + self.scale * self.lora_B(self.lora_A(self.dropout(x)))

    @torch.no_grad()
    def merge_into_base_(self, zero_lora: bool = True):
        """
        Permanently add LoRA delta into base weight: W += scale * (B@A).
        """
        # W_delta = lora_B.weight @ lora_A.weight  => (out, in)
        W_delta = self.lora_B.weight @ self.lora_A.weight
        self.base.weight += self.scale * W_delta
        if zero_lora:
            self.lora_A.weight.zero_()
            self.lora_B.weight.zero_()

def _get_parent_and_child_by_name(root: nn.Module, dotted_name: str):
    """
    Given 'blocks.25.self_attn.q', return (parent_module, 'q'), handling numeric
    indices for ModuleList/Sequential along the way.
    """
    parts = dotted_name.split(".")
    parent = root
    for p in parts[:-1]:
        if p.isdigit():
            parent = parent[int(p)]
        else:
            parent = getattr(parent, p)
    return parent, parts[-1]

def inject_lora_qk(model: nn.Module, r=128, alpha=128, dropout=0.0,
                   attn_qk_tokens=ATTN_QK_TOKENS, verbose=True):
    """
    Replace all q/k Linear modules in attention with LoRALinear.
    Returns list of replaced module names.
    """
    replaced = []
    # Snapshot before mutation
    named_modules = list(model.named_modules())
    for name, mod in named_modules:
        if isinstance(mod, nn.Linear) and any(tok.rstrip(".") in name for tok in attn_qk_tokens):
            parent, child = _get_parent_and_child_by_name(model, name)
            lora = LoRALinear.from_linear(mod, r=r, alpha=alpha, dropout=dropout)
            setattr(parent, child, lora)
            replaced.append(name)
    if verbose:
        print(f"[LoRA Inject] Replaced {len(replaced)} q/k Linear layers with LoRA (r={r}, alpha={alpha}).")
        if replaced:
            print("  e.g. ", replaced[:6], "..." if len(replaced) > 6 else "")
    return replaced


class TrainingLogger:
    """Centralized logging class for training metrics and curves."""
    
    def __init__(self, args, rank=0):
        self.rank = rank
        self.args = args
        self.log_dir = Path(args.output_path) / "logs"
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize loggers
        self.tensorboard_writer = None
        self.wandb_run = None
        self.metrics_log_file = None
        
        # Metrics storage
        self.metrics_history = {
            'loss': [],
            'learning_rate': [],
            'epoch': [],
            'step': [],
            'timestamp': []
        }
        
        if rank == 0:  # Only initialize logging on main process
            self._setup_loggers()
    
    def _setup_loggers(self):
        """Setup various logging backends."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Setup TensorBoard
        if TENSORBOARD_AVAILABLE and self.args.use_tensorboard:
            tb_log_dir = self.log_dir / f"tensorboard_{timestamp}"
            self.tensorboard_writer = SummaryWriter(log_dir=str(tb_log_dir))
            main_print(f"TensorBoard logging enabled: {tb_log_dir}", self.rank)
        
        # Setup Weights & Biases
        if WANDB_AVAILABLE and self.args.use_wandb:
            wandb.init(
                project=self.args.wandb_project or "s2v-5b-training",
                name=f"s2v-5b-{timestamp}",
                config=vars(self.args),
                dir=str(self.log_dir)
            )
            self.wandb_run = wandb
            main_print(f"Weights & Biases logging enabled: {self.args.wandb_project}", self.rank)
        
        # Setup file logging
        if self.args.use_file_logging:
            self.metrics_log_file = self.log_dir / f"metrics_{timestamp}.json"
            main_print(f"File logging enabled: {self.metrics_log_file}", self.rank)
    
    def log_metrics(self, metrics_dict, step, epoch):
        """Log metrics to all enabled backends."""
        if self.rank != 0:
            return
        
        # Add to history
        for key, value in metrics_dict.items():
            if key in self.metrics_history:
                self.metrics_history[key].append(value)
        
        self.metrics_history['step'].append(step)
        self.metrics_history['epoch'].append(epoch)
        self.metrics_history['timestamp'].append(time.time())
        
        # Log to TensorBoard
        if self.tensorboard_writer:
            for key, value in metrics_dict.items():
                self.tensorboard_writer.add_scalar(key, value, step)
            self.tensorboard_writer.flush()
        
        # Log to Weights & Biases
        if self.wandb_run:
            log_dict = {f"train/{k}": v for k, v in metrics_dict.items()}
            log_dict['epoch'] = epoch
            log_dict['step'] = step
            self.wandb_run.log(log_dict, step=step)
        
        # Log to file
        if self.metrics_log_file:
            with open(self.metrics_log_file, 'a') as f:
                log_entry = {
                    'step': step,
                    'epoch': epoch,
                    'timestamp': time.time(),
                    **metrics_dict
                }
                f.write(json.dumps(log_entry) + '\n')
    
    def log_model_parameters(self, model, step):
        """Log model parameter statistics."""
        if self.rank != 0:
            return
        
        # Calculate parameter statistics
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        param_metrics = {
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'trainable_ratio': trainable_params / total_params if total_params > 0 else 0
        }
        
        self.log_metrics(param_metrics, step, 0)
    
    def log_gradients(self, model, step):
        """Log gradient statistics."""
        if self.rank != 0:
            return
        
        grad_norms = []
        for name, param in model.named_parameters():
            if param.grad is not None:
                grad_norm = param.grad.data.norm(2).item()
                grad_norms.append(grad_norm)
        
        if grad_norms:
            grad_metrics = {
                'grad_norm_mean': np.mean(grad_norms),
                'grad_norm_std': np.std(grad_norms),
                'grad_norm_max': np.max(grad_norms),
                'grad_norm_min': np.min(grad_norms)
            }
            self.log_metrics(grad_metrics, step, 0)
    
    def close(self):
        """Close all logging backends."""
        if self.rank != 0:
            return
        
        if self.tensorboard_writer:
            self.tensorboard_writer.close()
        
        if self.wandb_run:
            self.wandb_run.finish()
        
        # Save final metrics history
        if self.metrics_log_file:
            history_file = self.metrics_log_file.parent / f"metrics_history_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            with open(history_file, 'w') as f:
                json.dump(self.metrics_history, f, indent=2)
            main_print(f"Metrics history saved to: {history_file}", self.rank)



# Note: TRAINABLE_KEYWORDS removed - now using requires_grad status for checkpoint saving/loading

def init_audio_encoder(checkpoint_dir):
    """Initialize AudioEncoder using wan modules."""
    from wan.modules.s2v.audio_encoder import AudioEncoder
    audio_encoder = AudioEncoder(
        model_id=os.path.join(checkpoint_dir, "wav2vec2-large-xlsr-53-english"))
    return audio_encoder

def get_audio_embedding(audio_path=None, audio_segment=None, audio_encoder=None, infer_frames=None, device=None, dtype=None, fps=25):
    """Extract audio embedding using wan AudioEncoder, matching speech2video behavior."""
    with torch.no_grad():
        z = audio_encoder.extract_audio_feat(audio_path=audio_path, audio_segment=audio_segment, return_all_layers=True)
    audio_embed_bucket, num_repeat = audio_encoder.get_audio_embed_bucket_fps(
        z, fps=fps, batch_frames=infer_frames, m=0)
    audio_embed_bucket = audio_embed_bucket.to(device, dtype)
    audio_embed_bucket = audio_embed_bucket.unsqueeze(0)
    if len(audio_embed_bucket.shape) == 3:
        # (B, feat, T)
        audio_embed_bucket = audio_embed_bucket.permute(0, 2, 1)
    elif len(audio_embed_bucket.shape) == 4:
        # (B, layers, feat, T)
        audio_embed_bucket = audio_embed_bucket.permute(0, 2, 3, 1)
    return audio_embed_bucket, num_repeat


def _gaussian_blur_3d_depthwise(x, sigma_t=0.0, sigma_h=1.0, sigma_w=1.0, k_mult=3):
    """
    Depthwise separable Gaussian blur over (T,H,W) for a 5D tensor x: (B, C, T, H, W).
    Each sigma<=0 skips that axis. Uses replicate padding and normalized kernels.
    """
    assert x.dim() == 5, "expected (B, C, T, H, W)"
    B, C, T, H, W = x.shape
    dtype = x.dtype
    device = x.device
    x = x.to(dtype=torch.float32)  # do conv in float32

    def _kernel1d(sigma):
        if sigma is None or sigma <= 0:
            return None, 0
        radius = max(1, int(math.ceil(k_mult * float(sigma))))
        ksize = 2 * radius + 1
        coords = torch.arange(-radius, radius + 1, device=device, dtype=torch.float32)
        kernel = torch.exp(-0.5 * (coords / float(sigma)) ** 2)
        kernel = kernel / kernel.sum()
        return kernel, radius

    # T axis
    kt, rt = _kernel1d(sigma_t)
    if kt is not None:
        w = kt.view(1, 1, -1, 1, 1).repeat(C, 1, 1, 1, 1)  # (C,1,kT,1,1)
        x = F.pad(x, (0, 0, 0, 0, rt, rt), mode='replicate')
        x = F.conv3d(x, w, groups=C)

    # H axis
    kh, rh = _kernel1d(sigma_h)
    if kh is not None:
        w = kh.view(1, 1, 1, -1, 1).repeat(C, 1, 1, 1, 1)  # (C,1,1,kH,1)
        x = F.pad(x, (0, 0, rh, rh, 0, 0), mode='replicate')
        x = F.conv3d(x, w, groups=C)

    # W axis
    kw, rw = _kernel1d(sigma_w)
    if kw is not None:
        w = kw.view(1, 1, 1, 1, -1).repeat(C, 1, 1, 1, 1)  # (C,1,1,1,kW)
        x = F.pad(x, (rw, rw, 0, 0, 0, 0), mode='replicate')
        x = F.conv3d(x, w, groups=C)

    return x.to(dtype=dtype)

    
def _make_latent_weights(
    face_masks,
    body_masks,
    target_t,
    target_h,
    target_w,
    device,
    dtype,
    *,
    smooth=True,
    sigma_t=0.0,      # set >0 for temporal smoothing (e.g., 0.5–1.0)
    sigma_h=1.0,      # spatial smoothing in pixels (latent space)
    sigma_w=1.0,
    k_mult=3,         # kernel radius ~= k_mult * sigma
    min_face_fraction=0.1,  # if (area / H*W) < this, boost by min_face_fraction / (area/H*W)
    boost_cap=5.0          # optional max multiplier (e.g., 5.0); None = no cap
):
    """
    face_masks, body_masks: (B, T_main, H, W) in {0,1}
    Returns W: (B, 1, target_t, target_h, target_w)
      where W ~ 1 + boosted_smooth(face) + smooth(body).

    Boost rule (per-frame):
      If binary face area fraction < min_face_fraction,
      multiply the *smoothed* face map by (min_face_fraction / area_fraction).
      This preserves smoothness while increasing emphasis on tiny faces.
    """
    B = face_masks.shape[0]
    if face_masks.numel() == 0 or face_masks.shape[1] == 0:
        return torch.ones(B, 1, target_t, target_h, target_w, device=device, dtype=dtype)

    # Work in float32 for resampling/convolution, then cast back
    f = face_masks.to(device=device, dtype=torch.float32).unsqueeze(1)  # (B,1,T,H,W)
    b = body_masks.to(device=device, dtype=torch.float32).unsqueeze(1)

    # Clean any accidental NaNs/Infs from dataset pipeline
    f = torch.nan_to_num(f, nan=0.0, posinf=1.0, neginf=0.0)
    b = torch.nan_to_num(b, nan=0.0, posinf=1.0, neginf=0.0)

    # Resize to latent grid (nearest keeps mask alignment)
    f_near = F.interpolate(f, size=(target_t, target_h, target_w), mode='nearest')
    b_near = F.interpolate(b, size=(target_t, target_h, target_w), mode='nearest')

    # Binary copy for area computation (do this BEFORE smoothing)
    f_bin = (f_near > 0.5).float()  # (B,1,T,H,W)

    if smooth:
        # Smooth boundaries with separable 3D Gaussian (depthwise)
        f_smooth = _gaussian_blur_3d_depthwise(f_near, sigma_t, sigma_h, sigma_w, k_mult=k_mult).clamp_(0.0, 1.0)
        b_smooth = _gaussian_blur_3d_depthwise(b_near, sigma_t, sigma_h, sigma_w, k_mult=k_mult).clamp_(0.0, 1.0)
    else:
        # Keep binary if desired
        f_smooth = f_bin
        b_smooth = (b_near > 0.5).float()

    # ---- Adaptive boost for small faces (per frame) ----
    # area_frac shape: (B,1,T,1,1) for easy broadcasting
    eps = 1e-6
    area = f_bin.sum(dim=(-2, -1), keepdim=True)                               # (B,1,T,1,1)
    hw = float(target_h * target_w)
    area_frac = area / (hw + eps)                                              # (B,1,T,1,1)

    # Compute boost only where area>0 (skip empty frames) and area_frac < min_face_fraction
    boost = torch.ones_like(area_frac)
    needs_boost = (area > 0) & (area_frac < float(min_face_fraction))
    desired = float(min_face_fraction) / (area_frac + eps)                      # (B,1,T,1,1)
    if boost_cap is not None:
        desired = torch.clamp(desired, max=float(boost_cap))
    boost = torch.where(needs_boost, desired, boost)                            # (B,1,T,1,1)

    # Apply boost AFTER smoothing to preserve soft edges
    f_weighted = f_smooth * boost

    # Final weight map
    W = 1.0 + f_weighted + b_smooth
    W = torch.nan_to_num(W, nan=1.0)  # guard against any stray NaNs

    return W.to(dtype=dtype).detach()


def roi_train_loss(face_masks_main, body_masks_main, train_target, model_pred, device, dtype):
    t_lat = int(train_target.shape[2])
    h_lat = int(train_target.shape[3])
    w_lat = int(train_target.shape[4])

    W = _make_latent_weights(
        face_masks=face_masks_main,
        body_masks=body_masks_main,
        target_t=t_lat, target_h=h_lat, target_w=w_lat,
        device=device, dtype=dtype
    )  # (B,1,Tlat,Hlat,Wlat)
    loss = F.mse_loss(model_pred.float(), train_target.float(), reduction="none")
    loss = (loss * W).mean() / (W.mean() + 1e-6)
    return loss

class S2V5BTrainingModule(torch.nn.Module):
    def __init__(self,
        args,
        config,
        checkpoint_dir,
        rank=0,
        load_text_encoder=True):
        super().__init__()
        self.args = args
        self.audio_file_key = "audio"
        self.config = config
        self.load_text_encoder = load_text_encoder
        self.param_dtype = config.param_dtype
        self.task = "s2v-5B"

        # Models that are not sharded are initialized on the current rank's device
        device = torch.device(f"cuda:{rank}")
        self.scheduler = FlowUniPCMultistepScheduler(
                num_train_timesteps=config.num_train_timesteps,
                shift=1.0,
                use_dynamic_shifting=False)
        self.scheduler.set_timesteps(1000, device=device, shift=config.sample_shift)
        main_print(f"Setting scheduler timesteps to {self.scheduler.timesteps}, shift to {config.sample_shift}", rank)
        # Components like T5, VAE are used for conditioning and not trained.
        # They are loaded on each rank.
        if load_text_encoder:
            main_print(f"Loading T5EncoderModel from {checkpoint_dir}", rank)
            from wan.modules.t5 import T5EncoderModel
            self.text_encoder = T5EncoderModel(
                text_len=config.text_len,
                dtype=config.t5_dtype,
                device=device,
                checkpoint_path=os.path.join(checkpoint_dir, config.t5_checkpoint),
                tokenizer_path=os.path.join(checkpoint_dir, config.t5_tokenizer),
            )
        else:
            main_print(f"Skipping T5EncoderModel loading.", rank)

        main_print(f"Loading Wan2_2_VAE from {checkpoint_dir}", rank)
        self.vae_stride = config.vae_stride
        self.patch_size = config.transformer.patch_size
        self.vae = Wan2_2_VAE(
            vae_pth=os.path.join(checkpoint_dir, config.vae_checkpoint),
            device=device)

        main_print(f"Creating WanModel_S2V from {checkpoint_dir}", rank)
        main_print(f"Creating WanModel_S2V_5B with parameters: {config.transformer}", rank)
        # Initialize the S2V 5B model
        self.model = WanModel_S2V_5B(
            model_type='s2v',
            patch_size=config.transformer.patch_size,
            text_len=config.text_len,
            in_dim=48,
            dim=config.transformer.dim,
            ffn_dim=config.transformer.ffn_dim,
            freq_dim=config.transformer.freq_dim,
            text_dim=4096,
            out_dim=48,
            num_heads=config.transformer.num_heads,
            num_layers=config.transformer.num_layers,
            window_size=config.transformer.window_size,
            qk_norm=config.transformer.qk_norm,
            cross_attn_norm=config.transformer.cross_attn_norm,
            eps=config.transformer.eps,
            cond_dim=config.transformer.cond_dim,
            audio_dim=config.transformer.audio_dim,
            enable_adain=config.transformer.enable_adain,
            adain_mode=config.transformer.adain_mode,
            audio_inject_layers=config.transformer.audio_inject_layers,
            zero_init=config.transformer.zero_init,
            zero_timestep=config.transformer.zero_timestep,
            enable_motioner=config.transformer.enable_motioner,
            add_last_motion=config.transformer.add_last_motion,
            enable_framepack=config.transformer.enable_framepack,
            framepack_drop_mode=config.transformer.framepack_drop_mode
        )

        # Load pretrained weights from TI2V 5B only, audio components will be trained from scratch
        self._load_pretrained_weights(checkpoint_dir, rank)
        self.sp_size = 1

        if getattr(self.args, "unfreeze_strategy", "minimal") == "lora":
            inject_lora_qk(
                self.model,
                r=getattr(self.args, "lora_rank", 128),
                alpha=getattr(self.args, "lora_alpha", 128),
                dropout=getattr(self.args, "lora_dropout", 0.0),
                attn_qk_tokens=ATTN_QK_TOKENS,
                verbose=(rank == 0),
            )
        self.scheduler.set_timesteps(1000)

        main_print("Initializing Audio Encoder from wan modules...", rank)
        self.audio_encoder = init_audio_encoder(checkpoint_dir)
        self.audio_encoder.model.requires_grad_(False)
        self.audio_encoder.model.to(device)
        self.trainable_parameter_names = []

    def _load_pretrained_weights(self, checkpoint_dir, rank):
        """Load pretrained weights from TI2V 5B base model only, audio components will be trained from scratch."""
        main_print(f"Loading TI2V 5B weights from {checkpoint_dir}", rank)
        
        # Load TI2V 5B weights
        if checkpoint_dir.endswith('.pth'):
            ti2v_state_dict = torch.load(checkpoint_dir, map_location='cpu')
        elif checkpoint_dir.endswith('Wan2.2-TI2V-5B'):
            safetensor_paths = [f for f in os.listdir(checkpoint_dir) if f.endswith('.safetensors')]
            state_dicts = [load_file(os.path.join(checkpoint_dir, f)) for f in safetensor_paths]
            ti2v_state_dict = state_dicts[0]
            for sd in state_dicts[1:]:
                ti2v_state_dict.update(sd)
        else:
            raise ValueError(f"Unsupported TI2V checkpoint format: {checkpoint_dir}")
        main_print(f"There are {len(ti2v_state_dict)} parameters in TI2V 5B checkpoint", rank)
        
        # Save TI2V state dict for freezing logic
        self.ti2v_state_dict = ti2v_state_dict
        
        # Get current model state dict
        current_state_dict = self.model.state_dict()
        main_print(f"There are {len(current_state_dict)} parameters in model", rank)
        new_state_dict = {}

        # Load TI2V 5B base weights and initialize audio components from scratch
        ti2v_loaded = 0
        
        main_print(f"***** Loading TI2V 5B weights from {checkpoint_dir} *****", rank)
        unfreeze_components = self._get_unfreeze_components()
        for name, param in current_state_dict.items():
            if any(component in name for component in unfreeze_components):
                # Initialize with current weights for other components
                #main_print(f"Loading {name} with [current weights]", rank)
                new_state_dict[name] = param.clone()
            else:
                # Load from TI2V 5B
                #main_print(f"Loading {name} from TI2V 5B checkpoint", rank)
                new_state_dict[name] = ti2v_state_dict[name]
                ti2v_loaded += 1
        
        # Load the merged weights
        missing, unexpected = self.model.load_state_dict(new_state_dict, strict=False)
        
        main_print(f"Loaded {ti2v_loaded} parameters from TI2V 5B checkpoint", rank)
        main_print(f"Model has {len(new_state_dict)} total parameters", rank)
        
        if len(missing) > 0:
            main_print(f"Missing parameters: {len(missing)}", rank)
        if len(unexpected) > 0:
            main_print(f"Unexpected parameters: {len(unexpected)}", rank)
        
        # Freeze the base TI2V weights
        self._freeze_ti2v_weights(rank)
        
        # Log unfrozen components
        self._log_unfrozen_components(rank)
    
    def _freeze_ti2v_weights(self, rank=0):
        """Freeze TI2V base model weights with selective unfreezing options."""
        # Get TI2V parameter names from the loaded checkpoint
        ti2v_param_names = set(self.ti2v_state_dict.keys())
        
        # Define components to unfreeze based on strategy
        unfreeze_components = self._get_unfreeze_components()

        main_print(f"Debug: Unfreeze strategy: {self.args.unfreeze_strategy}", rank)
        main_print(f"Debug: Unfreeze components: {unfreeze_components}", rank)

        trainable_count = 0
        for name, param in self.model.named_parameters():
            if self.args.finetune_all:
                # If finetune_all is True, make all parameters trainable
                param.requires_grad = True
            elif name in ti2v_param_names:
                # Check if this TI2V component should be unfrozen
                should_unfreeze = any(component in name for component in unfreeze_components)
                param.requires_grad = should_unfreeze
            else:
                # Keep all other components trainable (audio, etc.)
                param.requires_grad = True
            
            if param.requires_grad:
                trainable_count += 1
        
        main_print(f"Debug: Set {trainable_count} parameters as trainable", 0)
    
    def _get_unfreeze_components(self):
        framepack_layers = ["frame_packer", "trainable_cond_mask"] + ["audio_injector", "casual_audio_encoder"]
        """Get list of components to unfreeze based on unfreeze strategy."""
        if self.args.unfreeze_strategy == "none":
            return []
        elif self.args.unfreeze_strategy == "framepack":
            return framepack_layers
        elif self.args.unfreeze_strategy == "minimal":
            # Only unfreeze critical components for lip sync
            return ["time_embedding", "time_projection", "head"] + framepack_layers
        elif self.args.unfreeze_strategy == "lora":
            #attn_qk = ["self_attn.q.", "self_attn.k.", "cross_attn.q.", "cross_attn.k."]
            return ["time_embedding", "time_projection", "head"] + framepack_layers #+ attn_qk
        elif self.args.unfreeze_strategy == "moderate":
            # Unfreeze critical + last few layers
            return ["time_embedding", "time_projection", "head", 
                   "blocks.25", "blocks.26", "blocks.27", "blocks.28", "blocks.29"] + framepack_layers
        elif self.args.unfreeze_strategy == "aggressive":
            # Unfreeze critical + last 10 layers
            return ["time_embedding", "time_projection", "head"] + \
                   [f"blocks.{i}" for i in range(20, 30)] + framepack_layers
        elif self.args.unfreeze_strategy == "full":
            # Unfreeze everything
            return ["time_embedding", "time_projection", "head"] + \
                   [f"blocks.{i}" for i in range(30)] + framepack_layers
        else:
            # Default: minimal unfreezing
            return ["time_embedding", "time_projection", "head"] + framepack_layers
    
    def _log_unfrozen_components(self, rank):
        """Log which components are unfrozen for debugging."""
        if rank != 0:
            return
            
        unfrozen_components = set()
        frozen_components = set()
        
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                # Extract component name (e.g., "blocks.25.self_attn" -> "blocks.25")
                parts = name.split('.')
                if len(parts) >= 2:
                    component = '.'.join(parts[:2])
                else:
                    component = parts[0]
                unfrozen_components.add(component)
            else:
                parts = name.split('.')
                if len(parts) >= 2:
                    component = '.'.join(parts[:2])
                else:
                    component = parts[0]
                frozen_components.add(component)
        
        main_print(f"Unfreeze strategy: {self.args.unfreeze_strategy}", rank)
        main_print(f"Unfrozen components: {sorted(unfrozen_components)}", rank)
        main_print(f"Frozen components: {sorted(frozen_components)}", rank)
        
        # Count parameters
        unfrozen_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in self.model.parameters())
        main_print(f"Unfrozen parameters: {unfrozen_params / 1e6:.2f}M / {total_params / 1e6:.2f}M ({unfrozen_params/total_params*100:.1f}%)", rank)

    def training_loss(self, **inputs):
        device = next(self.model.parameters()).device
        
        video = inputs["video"].to(device, dtype=self.param_dtype)
        cond_image = inputs["cond_image"].to(device, dtype=self.param_dtype)
        prompt = inputs["prompt"]
        audio_embedding_list = inputs["audio_embedding"]

        if self.args.enable_roi_loss:
            face_masks_main = inputs["face_masks"]  # (B, T_main, H, W)
            body_masks_main = inputs["body_masks"]  # (B, T_main, H, W)
        else:
            face_masks_main = None
            body_masks_main = None

        b, t, c, h, w = video.shape
        lat_h, lat_w = h // self.vae_stride[1], w // self.vae_stride[2]
        
        with torch.no_grad():
            video = (video - 0.5) * 2.0
            cond_image = (cond_image - 0.5) * 2.0
            video = video.permute(0, 2, 1, 3, 4) # b, c, t, h, w
            
            # For S2V 5B, we use reference image as first frame
            # Prepare per-sample single-frame tensors for reference latent
            ref_latent_list = self.vae.encode([cond_image[i].unsqueeze(1) for i in range(cond_image.shape[0])])
            ref_latent = torch.stack(ref_latent_list).to(device=device, dtype=self.param_dtype) # b, c, 1, h, w
            assert ref_latent.shape[2:] == (1, lat_h, lat_w), f"Reference latent shape mismatch: {ref_latent.shape[2:]} != (1, {lat_h}, {lat_w})"
            
            if random.random() < inputs["img_dropout_prob"]:
                ref_latent = torch.zeros_like(ref_latent, device=device, dtype=self.param_dtype)

            if self.load_text_encoder:
                context = self.text_encoder(prompt, device)
            else:
                context = [c.to(device, dtype=self.param_dtype) for c in inputs["context"]]

        frame_num = t
        motion_frames = self.config.transformer.motion_frames
        valid_frames = self.config.transformer.num_frames
        assert frame_num == valid_frames or frame_num == valid_frames + motion_frames, \
            f"Input video frame number {frame_num} does not match config {valid_frames} or {valid_frames + motion_frames}"
        lat_motion_frames = (motion_frames + 3) // 4

        
        # Handle framepack: encode motion frames and input frames separately
        if self.config.transformer.enable_framepack:
            # Encode the whole video first to preserve temporal dependencies
            whole_video_latents_list = self.vae.encode([video[i] for i in range(video.shape[0])])
            whole_video_latents = torch.stack(whole_video_latents_list).to(self.param_dtype)
            
            if frame_num == valid_frames + motion_frames:
                # Split into motion and input latents
                motion_latents = whole_video_latents[:, :, :lat_motion_frames, :, :]
                if random.random() < inputs["framepack_dropout_prob"]:
                    motion_latents = torch.zeros_like(motion_latents, device=device, dtype=self.param_dtype)
                input_latents = whole_video_latents[:, :, lat_motion_frames:, :, :]
            elif frame_num == valid_frames:
                # No motion frames provided, use zeros
                motion_latents = torch.zeros(whole_video_latents.shape[0], whole_video_latents.shape[1], lat_motion_frames, whole_video_latents.shape[3], whole_video_latents.shape[4], device=device, dtype=self.param_dtype)
                input_latents = whole_video_latents
        else:
            # Original behavior: encode all frames as input_latents
            input_latents_list = self.vae.encode([video[i] for i in range(video.shape[0])])
            input_latents = torch.stack(input_latents_list).to(self.param_dtype) # b, c, t, h, w
            
            # Motion latents are zeros for non-framepack mode
            motion_latents = torch.zeros(b, ref_latent.shape[1], lat_motion_frames, lat_h, lat_w, device=device, dtype=self.param_dtype)
        
        # Align audio to frames as in speech2video: audio_input has shape [B=1, C1, C2, T]
        # We need the slice corresponding to current clip frames
        has_motion = (frame_num == valid_frames + motion_frames)
        audio_inputs = []
        if self.config.transformer.enable_framepack and has_motion:
            left_idx = motion_frames
        else:
            left_idx = 0
        right_idx = frame_num
        
        for audio_embedding in audio_embedding_list:
            if audio_embedding.dim() == 3:
                # (B, feat, T) -> expand to (B, 1, feat, T)
                audio_embedding = audio_embedding.unsqueeze(1)
            assert audio_embedding.dim() == 4, f"Expected 4D audio embedding, got {audio_embedding.shape}"
            audio_inputs.append(audio_embedding[..., left_idx:right_idx])

        # Stack per-sample audio inputs to shape [B, C1, C2, T]
        audio_embs = torch.cat(audio_inputs, dim=0).to(self.param_dtype)

        # # Prepare condition states (empty for now)
        # cond_states = torch.zeros(b, ref_latent.shape[1], frame_num, lat_h, lat_w, device=device, dtype=self.param_dtype)
        
        timestep_size = (b,)
        timesteps = torch.randint(0, self.scheduler.num_train_timesteps, timestep_size, device=device).long()
        
        noisy_latents = []
        noises = []
        for input_latent, timestep in zip(input_latents, timesteps):
            noise = torch.randn_like(input_latent)
            noised_latent = self.scheduler.add_noise(input_latent, noise, timestep.unsqueeze(0))
            noisy_latents.append(noised_latent)
            noises.append(noise)
        noisy_latents = torch.stack(noisy_latents, dim=0).to(device, dtype=self.param_dtype) # b, c, t, h, w
        noises = torch.stack(noises, dim=0).to(device, dtype=self.param_dtype) # b, c, t, h, w

        assert noisy_latents.shape == input_latents.shape and noises.shape == input_latents.shape, \
            f"Shapes mismatch: noisy_latents {noisy_latents.shape}, input_latents {input_latents.shape}, noises {noises.shape}"
        
        max_seq_len = int(math.ceil(((frame_num - 1) // self.vae_stride[0] + 1) * lat_h * lat_w / (self.patch_size[1] * self.patch_size[2]) / self.sp_size)) * self.sp_size

        # Prepare audio input for S2V model as expected by model_s2v_5b
        # The model expects extra_kwargs['audio_input'] with shape [B, C1, C2, T]
        audio_input = audio_embs

        model_inputs = {
            'context': context,
            'seq_len': max_seq_len,
            'ref_latents': [ref_latent[i] for i in range(b)],
            'motion_latents': [motion_latents[i] for i in range(b)],
            # 'cond_states': [cond_states[i] for i in range(b)],
            'audio_input': audio_input,
            'motion_frames': [motion_frames, lat_motion_frames],
            'drop_motion_frames': False,
        }
        noise_pred = self.model(noisy_latents, t=timesteps, **model_inputs)
        training_target = self.scheduler.training_target(input_latents, noises, timesteps)

        if not self.args.enable_roi_loss:
            loss = F.mse_loss(noise_pred[0].float(), training_target.float(), reduction="mean")
        else:
            loss = roi_train_loss(face_masks_main, body_masks_main, training_target, noise_pred[0], device, self.param_dtype)

        return loss

    def forward(self, data):
        device = next(self.model.parameters()).device
        dtype = self.param_dtype
        video = data["video"]
        cond_image = data["input_image"]
        prompt = data["caption"]
        context_list = data["context"]
        context_null_list = data["context_null"]
        audio_segment_list = data[self.audio_file_key]

        if self.args.enable_roi_loss:
            face_masks_main = data["face_masks"]  # (B, T_main, H, W)
            body_masks_main = data["body_masks"]  # (B, T_main, H, W)
        else:
            face_masks_main = None
            body_masks_main = None

        # Extract audio embeddings using wan AudioEncoder (match speech2video)
        audio_emb_list = []
        frame_num = video.shape[1] if len(video.shape) > 1 else 81
        
        for audio_segment in audio_segment_list:
            try:
                audio_emb, _ = get_audio_embedding(audio_segment=audio_segment, audio_encoder=self.audio_encoder, infer_frames=frame_num, device=device, dtype=dtype, fps=25)
                # Optionally apply dropout by zeroing the embedding
                if random.random() < self.args.audio_dropout_prob:
                    audio_emb = torch.zeros_like(audio_emb, device=device, dtype=dtype)
                audio_emb_list.append(audio_emb)
            except Exception:
                # Fallback to zeros with minimal shape if extraction fails
                # Default to 4D [B=1, C1=1, C2=1, T=frame_num]
                audio_emb_list.append(torch.zeros(1, 1, 1, frame_num, device=device, dtype=dtype))
                main_print("Audio processing failed, using dummy embeddings.", rank)

        # Classifier-Free Guidance Dropout
        for i in range(len(prompt)):
            if random.random() < self.args.text_dropout_prob: 
                prompt[i] = ""
                context_list[i] = context_null_list[i]

        inputs = {
            "video": video, "cond_image": cond_image, "prompt": prompt,
            "audio_embedding": audio_emb_list, "context": context_list, "img_dropout_prob": self.args.img_dropout_prob, 'framepack_dropout_prob': self.args.framepack_dropout_prob,
            "face_masks": face_masks_main, "body_masks": body_masks_main,
        }
        
        return self.training_loss(**inputs)

def launch_training_task(rank, world_size, args, model, dataset, global_step=0):
    # Initialize logger
    logger = TrainingLogger(args, rank)
    
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=True)
    dataloader = torch.utils.data.DataLoader(
        dataset,
        sampler=sampler,
        num_workers=args.num_workers,
        pin_memory=True,
        batch_size=args.batch_size,
        collate_fn=custom_collate_fn
    )

    train_named_params = model.module.model.named_parameters()
    default_lr_params, audio_scaled_params, lora_params = [], [], []

    for name, param in list(train_named_params):
        if not param.requires_grad:
            continue

        is_lora = (".lora_A." in name) or (".lora_B." in name)
        if is_lora:
            lora_params.append(param)
            continue

        if args.finetune_all:
            if any(k in name for k in ["audio_injector", "casual_audio_encoder"]):
                audio_scaled_params.append(param)
            else:
                default_lr_params.append(param)
        else:
            if any(k in name for k in ["frame_packer"]):
                audio_scaled_params.append(param)
            else:
                default_lr_params.append(param)

    optimizer_grouped_parameters = [
        {"params": lora_params,        "lr": args.learning_rate * args.lora_lr_scale},
        {"params": audio_scaled_params,"lr": args.learning_rate * args.audio_zero_init_lr_scaling},
        {"params": default_lr_params,  "lr": args.learning_rate},
    ]
    optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=args.learning_rate)

    def lr_lambda(current_step: int):
        if current_step < args.warmup_steps:
            return float(current_step) / float(max(1, args.warmup_steps))
        return 1.0
    lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    for _ in range(global_step):
        lr_scheduler.step()

    # Log initial model parameters
    if args.log_parameters:
        logger.log_model_parameters(model, global_step)
    
    for epoch in range(args.num_epochs):
        sampler.set_epoch(global_step)
        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{args.num_epochs}", disable=(rank!=0))
        
        for i, data in enumerate(progress_bar):
            if data is None: continue
                
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                loss = model(data)
                loss = loss / args.gradient_accumulation_steps
            loss.backward()

            if (i + 1) % args.gradient_accumulation_steps == 0:
                # Log gradient statistics before clipping
                if args.log_gradients:
                    logger.log_gradients(model, global_step)
                
                torch.nn.utils.clip_grad_norm_(default_lr_params, max_norm=1.0)
                torch.nn.utils.clip_grad_norm_(audio_scaled_params, max_norm=1.0)
                torch.nn.utils.clip_grad_norm_(lora_params, max_norm=1.0)
                optimizer.step()
                optimizer.zero_grad()
                lr_scheduler.step()
                
            # Log metrics at specified intervals
            if global_step % args.log_interval == 0:
                current_lr = lr_scheduler.get_last_lr()[0]
                metrics = {
                    'loss': loss.item() * args.gradient_accumulation_steps,
                    'learning_rate': current_lr,
                    'epoch': epoch + 1
                }
                logger.log_metrics(metrics, global_step, epoch + 1)
            
            # Save checkpoint every N steps
            if global_step % args.save_steps == 0:
                dist.barrier()
                
                if args.distributed_policy == "ddp":
                    cpu_state_dict = model.module.model.state_dict()
                elif args.distributed_policy == "fsdp":
                    # --- FSDP Native Checkpointing ---
                    save_policy = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
                    with FSDP.state_dict_type(model, StateDictType.FULL_STATE_DICT, save_policy):
                        cpu_state_dict = model.state_dict()
                else:
                    raise ValueError(f"Unsupported distributed policy: {args.distributed_policy}")
                
                if rank == 0:
                    trainable_state_dict = {}
                    # Get trainable parameter names from the model
                    trainable_param_names = {name: param for name, param in model.named_parameters() if param.requires_grad}
                    
                    # main_print(f"Debug: Found {len(trainable_param_names)} trainable parameters in model", rank)
                    # main_print(f"Debug: Found {len(cpu_state_dict)} parameters in cpu_state_dict", rank)
                    
                    # Debug: Show first few parameter names
                    model_names = list(trainable_param_names.keys())[:5]
                    cpu_names = list(cpu_state_dict.keys())[:5]
                    # main_print(f"Debug: Sample model param names: {model_names}", rank)
                    # main_print(f"Debug: Sample cpu_state_dict names: {cpu_names}", rank)
                    
                    # Debug: Show corrected names for DDP
                    if args.distributed_policy == "ddp":
                        corrected_names = [f"module.model.{name}" for name in cpu_names]
                        # main_print(f"Debug: Sample corrected names for DDP: {corrected_names}", rank)
                    
                    for name, tensor in cpu_state_dict.items():
                        original_name = name.replace("_fsdp_wrapped_module.", "")
                        
                        # For DDP, we need to add the module. prefix to match model parameter names
                        if args.distributed_policy == "ddp":
                            model_name = f"module.model.{original_name}"
                        else:
                            model_name = original_name
                        
                        # Check if this parameter is trainable
                        if model_name in trainable_param_names:
                            trainable_state_dict[original_name] = tensor
                        elif args.finetune_all:
                            # If finetune_all is True, save all parameters
                            trainable_state_dict[original_name] = tensor
                    
                    if trainable_state_dict:
                        # Log which parameters are being saved
                        main_print(f"Saving {len(trainable_state_dict)} trainable parameters:", rank)
                        # for name in sorted(trainable_state_dict.keys()):
                        #     main_print(f"  - {name}", rank)
                        
                        save_path = Path(args.output_path) / f"step-{global_step}.safetensors"
                        save_path.parent.mkdir(parents=True, exist_ok=True)
                        main_print(f"\nSaved checkpoint to {save_path}", rank)
                        save_file(trainable_state_dict, save_path)
                        
                        # delete ckpt if exists too many, keep k latest
                        ckpt_files = list(Path(args.output_path).glob("step-*.safetensors"))
                        save_ckpt_count = 4
                        if len(ckpt_files) >= save_ckpt_count:
                            ckpt_files.sort(key=os.path.getctime)
                            for f in ckpt_files[:-save_ckpt_count + 1]:
                                main_print(f"Deleting old checkpoint: {f}", rank)
                                os.remove(f)
                    else:
                        main_print("\nWarning: No trainable parameters were found to save.", rank)
            
            global_step += 1
            
            if rank == 0:
                lr_str = ", ".join([f"{lr:.2e}" for lr in lr_scheduler.get_last_lr()])
                progress_bar.set_postfix(loss=f"{loss.item() * args.gradient_accumulation_steps:.4f}", lr=f"[{lr_str}]")
        
    
    # Close logger
    logger.close()
    
    dist.barrier()
    main_print("Training finished.", rank)

def s2v_5b_parser():
    parser = argparse.ArgumentParser(description="Training script for the S2V 5B model.")
    # --- Paths ---
    parser.add_argument("--dataset_base_path", type=str, required=False, default="./data")
    parser.add_argument("--dataset_metadata_path", type=str, nargs='+', required=False, default=["./data/meta.csv"])
    parser.add_argument("--ckpt_dir", type=str, required=False, default="./ckpt", help="Path to Wan S2V 5B checkpoint dir.")
    parser.add_argument("--wav2vec_dir", type=str, required=False, default="./wav2vec", help="Path to wav2vec checkpoint dir.")
    parser.add_argument("--output_path", type=str, default="./s2v_5b_output", help="Dir to save checkpoints.")
    parser.add_argument("--framepack_dropout_prob", type=float, default=0.1)
    # --- Model & Task ---
    parser.add_argument("--load_text_encoder", action="store_true", help="Load the text encoder from the checkpoint.")
    parser.add_argument("--distributed_policy", type=str, default="ddp", choices=["fsdp", "ddp"], help="Distributed training policy to use.")
    
    # --- Training Hyperparameters ---
    parser.add_argument("--learning_rate", type=float, default=1e-5)
    parser.add_argument("--warmup_steps", type=int, default=1000)
    parser.add_argument("--num_epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=1, help="Per-device batch size")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--save_steps", type=int, default=500)
    parser.add_argument("--text_dropout_prob", type=float, default=0.1)
    parser.add_argument("--img_dropout_prob", type=float, default=0.1)
    parser.add_argument("--audio_dropout_prob", type=float, default=0.1)
    parser.add_argument("--use_gradient_checkpointing", action="store_true", help="Use gradient checkpointing to save memory.")
    parser.add_argument("--audio_zero_init_lr_scaling", type=float, default=10.0, help="Scaling factor for audio zero initialization learning rate.")
    parser.add_argument("--finetune_all", action="store_true", help="Finetune all parameters.")
    parser.add_argument("--unfreeze_strategy", type=str, default="minimal", 
                       choices=["none", "minimal", "moderate", "aggressive", "full", "lora"],
                       help="Strategy for unfreezing TI2V base model layers. "
                            "none: freeze all TI2V components, "
                            "minimal: unfreeze time_embedding, time_projection, head, "
                            "moderate: + last 5 transformer layers, "
                            "aggressive: + last 10 transformer layers, "
                            "full: unfreeze all TI2V components.")
    parser.add_argument("--enable_framepack", type=str, default="true", 
                       help="Enable framepack feature (default: true)")
    parser.add_argument("--enable_roi_loss", type=str, default="false", 
                       help="Enable ROI loss feature (default: false)")
    parser.add_argument("--lora_rank", type=int, default=128)
    parser.add_argument("--lora_alpha", type=int, default=128)
    parser.add_argument("--lora_dropout", type=float, default=0.0)
    parser.add_argument("--lora_lr_scale", type=float, default=10.0)
    
    # --- Dataloader & Video Config ---
    parser.add_argument("--size_bucket", type=str, default="fasttalk-480")
    parser.add_argument("--num_frames", type=int, default=120)
    parser.add_argument("--frame_interval", type=int, default=1)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--dataset_repeat", type=int, default=1)
    parser.add_argument("--always_use_motion_frames", type=str, default="false", help="Always use motion frames even if framepack is disabled.")
    
    # --- Logging Configuration ---
    parser.add_argument("--use_tensorboard", action="store_true", help="Enable TensorBoard logging.")
    parser.add_argument("--use_wandb", action="store_true", help="Enable Weights & Biases logging.")
    parser.add_argument("--use_file_logging", action="store_true", default=True, help="Enable JSON file logging.")
    parser.add_argument("--wandb_project", type=str, default="s2v-5b-training", help="Weights & Biases project name.")
    parser.add_argument("--log_interval", type=int, default=10, help="Log metrics every N steps.")
    parser.add_argument("--log_gradients", action="store_true", help="Log gradient statistics.")
    parser.add_argument("--log_parameters", action="store_true", help="Log model parameter statistics.")
    
    return parser

if __name__ == "__main__":
    parser = s2v_5b_parser()
    args = parser.parse_args()

    for p in Path('.').glob('core.*'):
        if p.is_file() or p.is_symlink():
            p.unlink(missing_ok=True)

    # --- Distributed Setup ---
    rank, local_rank, world_size = setup_distributed()
    main_print(f"Initialized distributed training on {world_size} GPUs.", local_rank)
    
    current_device = torch.device(f"cuda:{local_rank}")
    config = s2v_5B
    param_dtype = config.param_dtype
    
    # Override config.transformer.enable_framepack based on command line argument
    enable_framepack = args.enable_framepack.lower() in ['true', '1', 'yes', 'on']
    config.transformer.enable_framepack = enable_framepack
    main_print(f"Setting config.transformer.enable_framepack to {enable_framepack}", local_rank)
    
    # Parse enable_roi_loss argument
    enable_roi_loss = args.enable_roi_loss.lower() in ['true', '1', 'yes', 'on']
    main_print(f"Setting enable_roi_loss to {enable_roi_loss}", local_rank)
    config.transformer.num_frames = args.num_frames

    always_use_motion_frames = args.always_use_motion_frames.lower() in ['true', '1', 'yes', 'on']
    main_print(f"Setting always_use_motion_frames to {always_use_motion_frames}", local_rank)
    # --- Dataset ---
    if enable_framepack:
        if enable_roi_loss:
            main_print("Initializing VideoDatasetWithContextDynamicReso with ROI loss (framepack enabled).", local_rank)
            from wan.src.dataset_resolution_bins_roi import VideoDatasetWithContextDynamicReso
        else:
            main_print("Initializing VideoDatasetWithContextDynamicReso (framepack enabled).", local_rank)
            from wan.src.dataset_resolution_bins import VideoDatasetWithContextDynamicReso
        dataset = VideoDatasetWithContextDynamicReso(
            size_bucket=args.size_bucket,
            base_path=args.dataset_base_path,
            metadata_path=args.dataset_metadata_path,
            frame_interval=args.frame_interval,
            num_frames=args.num_frames,
            num_motion_frames=config.transformer.motion_frames,
            always_use_motion_frames=always_use_motion_frames,
            audio_file_key="audio",
            repeat=args.dataset_repeat,
            args=args,
            debug=False,
            text_only=False,
            rank=rank,
            text_embedding_path="./openhumanvid_features/text_embeddings",
            panda_text_embedding_path="./panda70m_features/text_embeddings"
        )
    else:
        main_print("Initializing VideoDataset (framepack disabled).", local_rank)
        dataset = VideoDataset(args=args, rank=rank)
    # --- Model Initialization ---
    main_print("Initializing S2V 5B Training Module.", local_rank)
    model = S2V5BTrainingModule(
        args=args,
        config=config,
        checkpoint_dir=args.ckpt_dir,
        rank=local_rank,
        load_text_encoder=args.load_text_encoder
    )

    # resume from checkpoint if available
    checkpoint_files = list(Path(args.output_path).glob("step-*.safetensors"))
    if checkpoint_files:
        def get_step_from_filename(path):
            match = re.search(r"step-(\d+)\.safetensors", path.name)
            if match:
                return int(match.group(1))
            return -1
        latest_checkpoint = max(checkpoint_files, key=get_step_from_filename)
        
        # Try to load checkpoint - if it fails, start from scratch
        try:
            audio_state_dict = load_file(latest_checkpoint)
            current_state_dict = model.state_dict()

            filtered_audio_state_dict = {}
            for name, tensor in audio_state_dict.items():
                # saved keys are for the inner Wan model; add prefix
                model_name = name if name.startswith("model.") else f"model.{name}"

                if model_name in current_state_dict:
                    if tensor.shape == current_state_dict[model_name].shape:
                        # ✅ load even if the param is currently frozen
                        filtered_audio_state_dict[model_name] = tensor
                    else:
                        # special-case: expand trainable_cond_mask [2, dim] -> [3, dim]
                        if (name == "trainable_cond_mask.weight"
                            and current_state_dict.get(model_name, None) is not None
                            and tensor.ndim == 2
                            and current_state_dict[model_name].ndim == 2
                            and tensor.shape[0] == 2
                            and current_state_dict[model_name].shape[0] == 3
                            and tensor.shape[1] == current_state_dict[model_name].shape[1]):
                            last_row = tensor[1:2]
                            filtered_audio_state_dict[model_name] = torch.cat([tensor, last_row], dim=0)
                        else:
                            main_print(f"Shape mismatch (skip): {name} "
                                    f"{tuple(tensor.shape)} -> {tuple(current_state_dict[model_name].shape)}", local_rank)
                else:
                    main_print(f"Missing in current model (skip): {name}", local_rank)

            m, u = model.load_state_dict(filtered_audio_state_dict, strict=False)
            main_print(f"Resumed {len(filtered_audio_state_dict)} params from {latest_checkpoint}", local_rank)

            global_step = int(latest_checkpoint.stem.split('-')[1])
            
        except Exception as e:
            main_print(f"Error loading checkpoint {latest_checkpoint}: {e}", local_rank)
            main_print("Starting from scratch.", local_rank)
            global_step = 0
    else:
        main_print("No checkpoint found, starting from scratch.", local_rank)
        global_step = 0
    
    if args.distributed_policy == "ddp":
        find_unused_parameters = False
        # Parameter freezing is now handled in _freeze_ti2v_weights() method
        # Count trainable parameters for logging
        count = sum(1 for p in model.parameters() if p.requires_grad)
        frozen_count = sum(1 for p in model.parameters() if not p.requires_grad)
        main_print(f"Total trainable tensors: {count}", local_rank)
        main_print(f"Total frozen tensors: {frozen_count}", local_rank)

        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in model.parameters())
        print(f"Total Parameters: {total_params / 1e6:.2f}M | Trainable: {trainable_params / 1e6:.2f}M", rank)
        # --- Model Wrapping ---
        model = model.to(param_dtype).to(current_device)
        
        # Enable gradient checkpointing if requested
        if args.use_gradient_checkpointing:
            try:
                # Try the built-in method first
                model.gradient_checkpointing_enable()
                main_print("Gradient checkpointing enabled (built-in) - memory usage reduced", local_rank)
            except AttributeError:
                # Fall back to custom implementation
                main_print("Using custom gradient checkpointing implementation", local_rank)
                model.model.use_gradient_checkpointing = True
                main_print(f"Set use_gradient_checkpointing = {model.model.use_gradient_checkpointing}", local_rank)
        else:
            main_print("Gradient checkpointing disabled", local_rank)
            model.model.use_gradient_checkpointing = False
            
        main_print("Wrapping model with DDP...", local_rank)

        model = torch.nn.parallel.DistributedDataParallel(
            model,
            device_ids=[local_rank],
            output_device=local_rank,
            find_unused_parameters=find_unused_parameters
        )
        main_print("Model wrapped successfully with DDP.", local_rank)
    elif args.distributed_policy == "fsdp":
        # --- FSDP Policies ---
        mp_policy = torch.distributed.fsdp.MixedPrecision(
            param_dtype=param_dtype,
            reduce_dtype=param_dtype,
            buffer_dtype=param_dtype,
        )
        
        auto_wrap_policy = ModuleWrapPolicy({ WanS2V5BAttentionBlock })

        # Enable gradient checkpointing if requested (before FSDP wrapping)
        if args.use_gradient_checkpointing:
            try:
                # Try the built-in method first
                model.gradient_checkpointing_enable()
                main_print("Gradient checkpointing enabled (built-in) - memory usage reduced", local_rank)
            except AttributeError:
                # Fall back to custom implementation
                main_print("Using custom gradient checkpointing implementation", local_rank)
                model.model.use_gradient_checkpointing = True
                main_print(f"Set use_gradient_checkpointing = {model.model.use_gradient_checkpointing}", local_rank)
        else:
            main_print("Gradient checkpointing disabled", local_rank)
            model.model.use_gradient_checkpointing = False
            
        # --- FSDP Model Wrapping ---
        main_print("Wrapping model with FSDP...", local_rank)
        model = FSDP(
            model,
            auto_wrap_policy=auto_wrap_policy,
            mixed_precision=mp_policy,
            device_id=torch.cuda.current_device(),
            use_orig_params=True
        )
        main_print("Model FSDP wrapped successfully.", local_rank)
        # Parameter freezing is now handled in _freeze_ti2v_weights() method
        # Count trainable parameters for logging
        count = sum(1 for p in model.parameters() if p.requires_grad)
        main_print(f"Total trainable tensors: {count}", local_rank)
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in model.parameters())
        print(f"Total Parameters: {total_params / 1e6:.2f}M | Trainable: {trainable_params / 1e6:.2f}M", rank)
    else:
        raise ValueError(f"Unsupported distributed policy: {args.distributed_policy}")

    # --- Launch Training ---
    if dataset is not None:
        main_print("Launching S2V 5B Training Task...", local_rank)
        launch_training_task(rank, world_size, args, model, dataset, global_step)
    else:
        main_print("Dataset is None. Please implement VideoDataset or provide dataset path.", local_rank)
        main_print("Training script is ready but requires dataset implementation.", local_rank)
