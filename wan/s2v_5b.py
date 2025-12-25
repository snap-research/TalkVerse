# Copyright 2024-2025 The Alibaba Wan Team Authors. All rights reserved.
import gc
import logging
import math
import os
import re
import random
import sys
import types
from contextlib import contextmanager
from copy import deepcopy
from functools import partial
from collections import OrderedDict
from typing import Dict

import numpy as np
import torch
import torch.nn as nn
import torch.cuda.amp as amp
import torch.distributed as dist
import torchvision.transforms.functional as TF
from decord import VideoReader
from PIL import Image
from safetensors import safe_open
from safetensors.torch import load_file
from torchvision import transforms
from tqdm import tqdm

from .distributed.fsdp import shard_model
from .distributed.sequence_parallel import sp_attn_forward, sp_dit_forward
from .distributed.util import get_world_size
from .modules.s2v.audio_encoder import AudioEncoder
from .modules.s2v.model_s2v_5b import WanModel_S2V_5B, sp_attn_forward_s2v_5b
from .modules.t5 import T5EncoderModel
from .modules.vae2_2 import Wan2_2_VAE  # Use VAE2.2 from TI2V 5B
from .utils.fm_solvers import (
    FlowDPMSolverMultistepScheduler,
    get_sampling_sigmas,
    retrieve_timesteps,
)
from .utils.fm_solvers_unipc import FlowUniPCMultistepScheduler


# ---- Two-bin inference buckets (match training) ----
ASPECT_RATIO_627_FALLBACK = {
     '0.26': ([320, 1216], 1), '0.38': ([384, 1024], 1), '0.50': ([448, 896], 1), '0.67': ([512, 768], 1),
     '0.82': ([576, 704], 1),  '1.00': ([640, 640], 1),  '1.22': ([704, 576], 1), '1.50': ([768, 512], 1),
     '1.86': ([832, 448], 1),  '2.00': ([896, 448], 1),  '2.50': ([960, 384], 1), '2.83': ([1088, 384], 1),
     '3.60': ([1152, 320], 1), '3.80': ([1216, 320], 1), '4.00': ([1280, 320], 1)
}

ASPECT_RATIO_960_FALLBACK = {
     '0.22': ([448, 2048], 1), '0.29': ([512, 1792], 1), '0.36': ([576, 1600], 1), '0.45': ([640, 1408], 1),
     '0.55': ([704, 1280], 1), '0.63': ([768, 1216], 1), '0.76': ([832, 1088], 1), '0.88': ([896, 1024], 1),
     '1.00': ([960, 960], 1),  '1.14': ([1024, 896], 1), '1.31': ([1088, 832], 1), '1.50': ([1152, 768], 1),
     '1.58': ([1216, 768], 1), '1.82': ([1280, 704], 1), '1.91': ([1344, 704], 1), '2.20': ([1408, 640], 1),
     '2.30': ([1472, 640], 1), '2.67': ([1536, 576], 1), '2.89': ([1664, 576], 1), '3.62': ([1856, 512], 1),
     '3.75': ([1920, 512], 1)
}

def _load_infer_buckets():
    """
    Try to import bucket dicts from wan.utils.multitalk_utils, fallback to included tables.
    """
    try:
        import importlib
        mod = importlib.import_module("wan.utils.multitalk_utils")
        ar627 = getattr(mod, "ASPECT_RATIO_627", ASPECT_RATIO_627_FALLBACK)
        ar960 = getattr(mod, "ASPECT_RATIO_960", ASPECT_RATIO_960_FALLBACK)
    except Exception:
        ar627 = ASPECT_RATIO_627_FALLBACK
        ar960 = ASPECT_RATIO_960_FALLBACK
    return {
        "fasttalk-480": ar627,  # shorter-side ~480 logic as in training bucket
        "fasttalk-720": ar960,  # shorter-side ~720 logic as in training bucket
    }

def _closest_bucket_hw_for_image(pil_image, bucket_dict):
    """
    Choose (H,W) from the selected bucket that best matches the ref aspect ratio.
    """
    ratio = pil_image.height / float(pil_image.width + 1e-8)
    best_key = min(bucket_dict.keys(), key=lambda k: abs(float(k) - ratio))
    target_h, target_w = bucket_dict[best_key][0]
    return int(target_h), int(target_w)




def load_safetensors(path):
    tensors = {}
    with safe_open(path, framework="pt", device="cpu") as f:
        for key in f.keys():
            tensors[key] = f.get_tensor(key)
    return tensors

ATTN_QK_TOKENS = ["self_attn.q.", "self_attn.k.", "cross_attn.q.", "cross_attn.k.", "cross_attn.norm_k.", "cross_attn.norm_q.", "cross_attn.o", "self_attn.norm_k.", "self_attn.norm_q.", "self_attn.o", ".norm3.", ".modulation", ".ffn."]


def _get_parent_and_child_by_name(root: nn.Module, dotted_name: str):
    parts = dotted_name.split(".")
    parent = root
    for p in parts[:-1]:
        parent = parent[int(p)] if p.isdigit() else getattr(parent, p)
    return parent, parts[-1]

class LoRALinear(nn.Module):
    def __init__(self, base_linear: nn.Linear, r: int = 128, alpha: int = 128):
        super().__init__()
        assert isinstance(base_linear, nn.Linear)
        self.in_features = base_linear.in_features
        self.out_features = base_linear.out_features
        self.has_bias = base_linear.bias is not None
        self.scale = alpha / float(r) if r > 0 else 0.0

        self.base = nn.Linear(self.in_features, self.out_features, bias=self.has_bias)
        with torch.no_grad():
            self.base.weight.copy_(base_linear.weight)
            if self.has_bias:
                self.base.bias.copy_(base_linear.bias)
        for p in self.base.parameters():
            p.requires_grad = False

        self.lora_A = nn.Linear(self.in_features, r, bias=False)
        self.lora_B = nn.Linear(r, self.out_features, bias=False)
        nn.init.zeros_(self.lora_B.weight)

    def forward(self, x):
        return self.base(x) + self.scale * self.lora_B(self.lora_A(x))

@torch.no_grad()
def merge_lora_into_base(model: nn.Module, lora_sd: dict, r=128, alpha=128):
    """
    Merge LoRA tensors from a checkpoint directly into model.state_dict weights.
    Works regardless of wrapper prefixes or module types.
    """
    from collections import defaultdict

    scale = alpha / float(r) if r > 0 else 0.0

    # 1) Group A/B by their base name
    groups = defaultdict(dict)
    for k, v in lora_sd.items():
        if not (k.endswith("lora_A.weight") or k.endswith("lora_B.weight")):
            continue

        # Strip optional leading "model." (or any single top-level prefix) to match inference model keys
        k_ = k
        if k_.startswith("model."):
            k_ = k_[len("model."):]
        # You can add more prefixes here if needed:
        # if k_.startswith("noise_model."): k_ = k_[len("noise_model."):]

        if k_.endswith("lora_A.weight"):
            base = k_[: -len("lora_A.weight")].rstrip(".")
            groups[base]["A"] = v
        elif k_.endswith("lora_B.weight"):
            base = k_[: -len("lora_B.weight")].rstrip(".")
            groups[base]["B"] = v

    # 2) Pull current state_dict (CPU for safety), mutate in place
    sd = model.state_dict()
    merged_names = []
    missed = []

    for base_name, ab in groups.items():
        if "A" not in ab or "B" not in ab:
            continue
        A, B = ab["A"], ab["B"]  # A: [r, in], B: [out, r]
        delta = (B @ A) * scale   # [out, in]

        # Candidate destinations in state_dict
        candidates = [
            f"{base_name}.weight",        # plain Linear
            f"{base_name}.base.weight",   # LoRALinear(base=Linear)
        ]

        placed = False
        for key in candidates:
            if key in sd:
                if sd[key].shape != delta.shape:
                    # shape mismatch: try next candidate
                    continue
                # add delta on CPU to avoid dtype/cuda mismatches
                w = sd[key]
                # ensure types match
                delta_cast = delta.to(dtype=w.dtype, device=w.device)
                w.add_(delta_cast)
                sd[key] = w
                merged_names.append(key)
                placed = True
                break

        if not placed:
            missed.append(base_name)

    # 3) Load the updated weights back
    missing, unexpected = model.load_state_dict(sd, strict=False)

    print(
        f"Merged {len(merged_names)} LoRA linear weights into base model.\n"
        f"Examples: {merged_names[:6]}\n"
        f"Missed (no matching base weight key): {missed[:6]}\n"
        f"load_state_dict: missing={len(missing)}, unexpected={len(unexpected)}"
    )


"""@torch.no_grad()
def merge_lora_into_base(model: nn.Module, lora_sd: dict, r=128, alpha=128):
    merged_status=[]
    scale = alpha / float(r) if r > 0 else 0.0
    # loop over saved LoRA tensors: "...q.lora_A.weight" / "...q.lora_B.weight"
    # group by base name "...q"
    from collections import defaultdict
    groups = defaultdict(dict)
    for k, v in lora_sd.items():
        if k.endswith("lora_A.weight"):
            groups[k[:-len("lora_A.weight")] + "" ]["A"] = v
        elif k.endswith("lora_B.weight"):
            groups[k[:-len("lora_B.weight")] + "" ]["B"] = v

    for prefix, ab in groups.items():
        # strip trailing dot that split logic introduced
        base_name = prefix[:-1] if prefix.endswith(".") else prefix
        if "A" not in ab or "B" not in ab:
            continue
        A, B = ab["A"], ab["B"]  # A: [r, in], B: [out, r]
        # find the base linear
        parent, child = _get_parent_and_child_by_name(model, base_name)
        lin = getattr(parent, child)
        if not isinstance(lin, nn.Linear):
            # if inference already has LoRA wrapper, merge into its base instead
            if isinstance(lin, LoRALinear):
                lin.base.weight += scale * (B @ A)
                merged_status.append(base_name)
                #print(f"Merged LoRA into existing LoRALinear base: {base_name}")
            continue
        lin.weight += scale * (B @ A)

    print(f"Merged {len(merged_status)} LoRA layers into base model. {merged_status}")"""



class WanS2V_5B:

    def __init__(
        self,
        config,
        checkpoint_dir,
        device_id=0,
        rank=0,
        t5_fsdp=False,
        dit_fsdp=False,
        use_sp=False,
        t5_cpu=False,
        init_on_cpu=True,
        convert_model_dtype=False,
        lora_ckpt=None, lora_rank=128, lora_alpha=128, lora_merge=True,
        size_bucket: str = "fasttalk-720",
    ):
        r"""
        Initializes the 5B speech-to-video generation model components.
        Combines the 5B WanTI2V base model with audio components from the 14B S2V model.

        Args:
            config (EasyDict):
                Object containing model parameters initialized from config.py
            checkpoint_dir (`str`):
                Path to directory containing model checkpoints
            device_id (`int`,  *optional*, defaults to 0):
                Id of target GPU device
            rank (`int`,  *optional*, defaults to 0):
                Process rank for distributed training
            t5_fsdp (`bool`, *optional*, defaults to False):
                Enable FSDP sharding for T5 model
            dit_fsdp (`bool`, *optional*, defaults to False):
                Enable FSDP sharding for DiT model
            use_sp (`bool`, *optional*, defaults to False):
                Enable distribution strategy of sequence parallel.
            t5_cpu (`bool`, *optional*, defaults to False):
                Whether to place T5 model on CPU. Only works without t5_fsdp.
            init_on_cpu (`bool`, *optional*, defaults to True):
                Enable initializing Transformer Model on CPU. Only works without FSDP or USP.
            convert_model_dtype (`bool`, *optional*, defaults to False):
                Convert DiT model parameters dtype to 'config.param_dtype'.
                Only works without FSDP.
        """
        self.device = torch.device(f"cuda:{device_id}")
        self.config = config
        self.rank = rank
        self.t5_cpu = t5_cpu
        self.init_on_cpu = init_on_cpu

        self.num_train_timesteps = config.num_train_timesteps
        self.param_dtype = config.param_dtype

        self.lora_ckpt = lora_ckpt
        self.lora_rank = lora_rank
        self.lora_alpha = lora_alpha
        self.lora_merge = lora_merge
        if t5_fsdp or dit_fsdp or use_sp:
            self.init_on_cpu = False

        shard_fn = partial(shard_model, device_id=device_id)
        self.text_encoder = T5EncoderModel(
            text_len=config.text_len,
            dtype=config.t5_dtype,
            device=torch.device('cpu'),
            checkpoint_path=os.path.join(checkpoint_dir, config.t5_checkpoint),
            tokenizer_path=os.path.join(checkpoint_dir, config.t5_tokenizer),
            shard_fn=shard_fn if t5_fsdp else None,
        )

        # Use VAE2.2 from TI2V 5B (larger compression)
        self.vae = Wan2_2_VAE(
            vae_pth=os.path.join(checkpoint_dir, config.vae_checkpoint),
            device=self.device)

        logging.info(f"Creating WanModel_S2V_5B from {checkpoint_dir}")
        # Initialize the S2V 5B model with same parameters as training script
        logging.info(f"Creating WanModel_S2V_5B with parameters: {config.transformer}")
        self.noise_model = WanModel_S2V_5B(
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
            framepack_drop_mode=config.transformer.framepack_drop_mode,
            v_scale=config.transformer.v_scale
        )

        # Load pretrained weights from TI2V 5B checkpoint
        self._load_pretrained_weights(checkpoint_dir)
        
        # # Zero initialize audio weights for debugging (audio will have no effect on video generation)
        # logging.info("Zero initializing audio weights for debugging...")
        # self.noise_model.zero_init_weights()

        self.noise_model = self._configure_model(
            model=self.noise_model,
            use_sp=use_sp,
            dit_fsdp=dit_fsdp,
            shard_fn=shard_fn,
            convert_model_dtype=convert_model_dtype)

        # Audio encoder from S2V 14B - match training script pattern
        self.audio_encoder = AudioEncoder(
            model_id=os.path.join(checkpoint_dir,
                                  "wav2vec2-large-xlsr-53-english"))
        self.audio_encoder.model.requires_grad_(False)
        self.audio_encoder.model.to(self.device)

        if use_sp:
            self.sp_size = get_world_size()
        else:
            self.sp_size = 1

        self.sample_neg_prompt = config.sample_neg_prompt
        self.num_motion_frames = config.transformer.motion_frames
        self.drop_first_motion = config.drop_first_motion
        self.fps = config.sample_fps
        self.audio_sample_m = 0
        # ---- NEW: resolution bin for inference ----
        self.size_bucket = size_bucket
        self.infer_buckets = _load_infer_buckets()
        if self.size_bucket not in self.infer_buckets:
            raise ValueError(f"Invalid size_bucket '{self.size_bucket}'. Use 'fasttalk-480' or 'fasttalk-720'.")


    def _load_pretrained_weights(self, checkpoint_dir):
        """Load complete S2V 5B model weights: TI2V base + trained audio components.
        """
        import os, re
        from collections import OrderedDict
        from safetensors.torch import load_file

        logging.info(f"Loading complete S2V 5B model weights from {checkpoint_dir}")

        # --- helpers (scoped to this method) ---
        def _natural_shard_key(name: str):
            m = re.search(r"-(\d+)-of-\d+\.safetensors$", name)
            return int(m.group(1)) if m else name

        def _list_safetensors_sorted(dirpath: str):
            names = [f for f in os.listdir(dirpath) if f.endswith(".safetensors")]
            return sorted(names, key=_natural_shard_key)

        # --------------------------
        # Step 1: Load TI2V 5B base weights (original logic, but shard order is stable)
        logging.info("Loading TI2V 5B base weights...")
        if checkpoint_dir.endswith('.pth'):
            ti2v_state_dict = torch.load(checkpoint_dir, map_location='cpu')
        elif checkpoint_dir.endswith('Wan2.2-TI2V-5B') or checkpoint_dir.endswith('Wan2.2-TI2V-5B/'):
            names = _list_safetensors_sorted(checkpoint_dir)
            state_dicts = [load_file(os.path.join(checkpoint_dir, f)) for f in names]
            ti2v_state_dict = state_dicts[0]
            for sd in state_dicts[1:]:
                ti2v_state_dict.update(sd)
        else:
            raise ValueError(f"Unsupported TI2V checkpoint format: {checkpoint_dir}")

        logging.info(f"Loaded {len(ti2v_state_dict)} parameters from TI2V 5B base checkpoint")

        # --------------------------
        # Step 3: Try to load trained audio components
        audio_state_dict = {}
        if self.lora_ckpt is not None:
            logging.info(f"Loading S2V 5B training checkpoint from {self.lora_ckpt}")
            try:
                audio_state_dict = load_file(self.lora_ckpt)
                logging.info(f"Loaded {len(audio_state_dict)} audio component parameters from training checkpoint")
            except Exception as e:
                logging.warning(f"Failed to load audio components: {e}")
                logging.info("Audio components will be initialized from scratch")
        else:
            raise NotImplementedError
        
        # Step 4: Merge TI2V base weights and audio components
        current_state_dict = self.noise_model.state_dict()
        merged_state_dict = {}
        ti2v_loaded = 0
        audio_loaded = 0
        
        for name, param in current_state_dict.items():
            if name in audio_state_dict: # and name not in ti2v_state_dict:
                #logging.info(f"Loading audio component parameters from training checkpoint: {name}")
                # Load from trained audio components
                merged_state_dict[name] = audio_state_dict[name]
                audio_loaded += 1
            elif name in ti2v_state_dict:
                # Load from TI2V 5B base
                #logging.info(f"Loading {name} from TI2V 5B base")
                merged_state_dict[name] = ti2v_state_dict[name]
                ti2v_loaded += 1
            else:
                # Initialize from scratch (shouldn't happen for complete model)
                logging.info(f"WARNING: Initialize {name} from scratch (shouldn't happen for complete model)")
                merged_state_dict[name] = param.clone()
        
        # Load the merged weights
        missing, unexpected = self.noise_model.load_state_dict(merged_state_dict, strict=False)
        # ---- Optional: merge LoRA from the same finetune ckpt (if present) ----
        def _has_lora_keys(sd: dict):
            for k in sd.keys():
                if k.endswith("lora_A.weight") or k.endswith("lora_B.weight"):
                    return True
            return False

        if _has_lora_keys(audio_state_dict):
            logging.info(
                f"Found LoRA tensors in training checkpoint "
                f"(A={sum(k.endswith('lora_A.weight') for k in audio_state_dict)}, "
                f"B={sum(k.endswith('lora_B.weight') for k in audio_state_dict)}). "
                f"Merging into base (r={self.lora_rank}, alpha={self.lora_alpha})."
            )
            merge_lora_into_base(self.noise_model, audio_state_dict, r=self.lora_rank, alpha=self.lora_alpha)
        else:
            logging.info("No LoRA tensors found in training checkpoint; skipping LoRA merge.")


        logging.info(f"Successfully loaded complete S2V 5B model:")
        logging.info(f"  - TI2V base parameters: {ti2v_loaded}")
        logging.info(f"  - Audio component parameters: {audio_loaded}")
        logging.info(f"  - Total model parameters: {len(current_state_dict)}")
        
        if len(missing) > 0:
            logging.info(f"Missing parameters: {len(missing)}")
        if len(unexpected) > 0:
            logging.info(f"Unexpected parameters: {len(unexpected)}")
        
        # Freeze all weights for inference
        self._freeze_all_weights()
    
    def _freeze_all_weights(self):
        """Freeze all model weights for inference."""
        # For inference, freeze all weights (no training needed)
        for name, param in self.noise_model.named_parameters():
            param.requires_grad = False

    def _configure_model(self, model, use_sp, dit_fsdp, shard_fn,
                         convert_model_dtype):
        """
        Configures a model object. This includes setting evaluation modes,
        applying distributed parallel strategy, and handling device placement.

        Args:
            model (torch.nn.Module):
                The model instance to configure.
            use_sp (`bool`):
                Enable distribution strategy of sequence parallel.
            dit_fsdp (`bool`):
                Enable FSDP sharding for DiT model.
            shard_fn (callable):
                The function to apply FSDP sharding.
            convert_model_dtype (`bool`):
                Convert DiT model parameters dtype to 'config.param_dtype'.
                Only works without FSDP.

        Returns:
            torch.nn.Module:
                The configured model.
        """
        model.eval().requires_grad_(False)
        if use_sp:
            for block in model.blocks:
                block.self_attn.forward = types.MethodType(
                    sp_attn_forward_s2v_5b, block.self_attn)
            model.use_context_parallel = True

        if dist.is_initialized():
            dist.barrier()

        if dit_fsdp:
            model = shard_fn(model)
        else:
            if convert_model_dtype:
                model.to(self.param_dtype)
            if not self.init_on_cpu:
                model.to(self.device)

        return model

    def get_size_less_than_area(self,
                                height,
                                width,
                                target_area=1024 * 704,
                                divisor=64):
        if height * width <= target_area:
            # If the original image area is already less than or equal to the target,
            # no resizing is needed—just padding. Still need to ensure that the padded area doesn't exceed the target.
            max_upper_area = target_area
            min_scale = 0.1
            max_scale = 1.0
        else:
            # Resize to fit within the target area and then pad to multiples of `divisor`
            max_upper_area = target_area  # Maximum allowed total pixel count after padding
            d = divisor - 1
            b = d * (height + width)
            a = height * width
            c = d**2 - max_upper_area

            # Calculate scale boundaries using quadratic equation
            min_scale = (-b + math.sqrt(b**2 - 2 * a * c)) / (
                2 * a)  # Scale when maximum padding is applied
            max_scale = math.sqrt(max_upper_area /
                                  (height * width))  # Scale without any padding

        # We want to choose the largest possible scale such that the final padded area does not exceed max_upper_area
        # Use binary search-like iteration to find this scale
        find_it = False
        for i in range(100):
            scale = max_scale - (max_scale - min_scale) * i / 100
            new_height, new_width = int(height * scale), int(width * scale)

            # Pad to make dimensions divisible by 64
            pad_height = (64 - new_height % 64) % 64
            pad_width = (64 - new_width % 64) % 64
            pad_top = pad_height // 2
            pad_bottom = pad_height - pad_top
            pad_left = pad_width // 2
            pad_right = pad_width - pad_left

            padded_height, padded_width = new_height + pad_height, new_width + pad_width

            if padded_height * padded_width <= max_upper_area:
                find_it = True
                break

        if find_it:
            return padded_height, padded_width
        else:
            # Fallback: calculate target dimensions based on aspect ratio and divisor alignment
            aspect_ratio = width / height
            target_width = int(
                (target_area * aspect_ratio)**0.5 // divisor * divisor)
            target_height = int(
                (target_area / aspect_ratio)**0.5 // divisor * divisor)

            # Ensure the result is not larger than the original resolution
            if target_width >= width or target_height >= height:
                target_width = int(width // divisor * divisor)
                target_height = int(height // divisor * divisor)

            return target_height, target_width

    def prepare_default_cond_input(self,
                                   map_shape=[3, 12, 64, 64],
                                   motion_frames=5,
                                   lat_motion_frames=2,
                                   enable_mano=False,
                                   enable_kp=False,
                                   enable_pose=False):
        default_value = [1.0, -1.0, -1.0]
        cond_enable = [enable_mano, enable_kp, enable_pose]
        cond = []
        for d, c in zip(default_value, cond_enable):
            if c:
                map_value = torch.ones(
                    map_shape, dtype=self.param_dtype, device=self.device) * d
                cond_lat = torch.cat([
                    map_value[:, :, 0:1].repeat(1, 1, motion_frames, 1, 1),
                    map_value
                ],
                                     dim=2)
                cond_lat = torch.stack(
                    self.vae.encode([cond_lat[i].to(self.param_dtype) for i in range(cond_lat.shape[0])]))[:, :, lat_motion_frames:].to(
                            self.param_dtype)

                cond.append(cond_lat)
        if len(cond) >= 1:
            cond = torch.cat(cond, dim=1)
        else:
            cond = None
        return cond

    def encode_audio(self, audio_path, infer_frames):
        z = self.audio_encoder.extract_audio_feat(
            audio_path, return_all_layers=True)
        audio_embed_bucket, num_repeat = self.audio_encoder.get_audio_embed_bucket_fps(
            z, fps=self.fps, batch_frames=infer_frames, m=self.audio_sample_m)
        audio_embed_bucket = audio_embed_bucket.to(self.device,
                                                   self.param_dtype)
        audio_embed_bucket = audio_embed_bucket.unsqueeze(0)
        if len(audio_embed_bucket.shape) == 3:
            audio_embed_bucket = audio_embed_bucket.permute(0, 2, 1)
        elif len(audio_embed_bucket.shape) == 4:
            audio_embed_bucket = audio_embed_bucket.permute(0, 2, 3, 1)
        return audio_embed_bucket, num_repeat

    def read_last_n_frames(self,
                           video_path,
                           n_frames,
                           target_fps=16,
                           reverse=False):
        """
        Read the last `n_frames` from a video at the specified frame rate.

        Parameters:
            video_path (str): Path to the video file.
            n_frames (int): Number of frames to read.
            target_fps (int, optional): Target sampling frame rate. Defaults to 16.
            reverse (bool, optional): Whether to read frames in reverse order. 
                                    If True, reads the first `n_frames` instead of the last ones.

        Returns:
            np.ndarray: A NumPy array of shape [n_frames, H, W, 3], representing the sampled video frames.
        """
        vr = VideoReader(video_path)
        original_fps = vr.get_avg_fps()
        total_frames = len(vr)

        interval = max(1, round(original_fps / target_fps))

        required_span = (n_frames - 1) * interval

        start_frame = max(0, total_frames - required_span -
                          1) if not reverse else 0

        sampled_indices = []
        for i in range(n_frames):
            indice = start_frame + i * interval
            if indice >= total_frames:
                break
            else:
                sampled_indices.append(indice)

        return vr.get_batch(sampled_indices).asnumpy()

    def load_pose_cond(self, num_repeat, infer_frames, size):
        """Load pose conditioning - for S2V 5B model, pose is not supported, return zeros"""
        HEIGHT, WIDTH = size
        # S2V 5B model does not support pose_video - return zero conditioning
        cond_tensors = [-torch.ones([1, 3, infer_frames, HEIGHT, WIDTH])]

        COND = []
        for r in range(len(cond_tensors)):
            cond = cond_tensors[r]
            cond = torch.cat([cond[:, :, 0:1].repeat(1, 1, 1, 1, 1), cond],
                             dim=2)
            cond_lat = torch.stack(
                self.vae.encode([cond[i].to(dtype=self.param_dtype, device=self.device) for i in range(cond.shape[0])]))[:, :,
                                                  1:].cpu()  # for mem save
            COND.append(cond_lat)
        return COND

    def get_gen_size(self, size, max_area, ref_image_path, pre_video_path):
        """
        If self.size_bucket is set, pick (H,W) from that bucket using the reference frame's aspect ratio.
        Otherwise, use the legacy area-based scaling.
        """
        if size is not None:
            HEIGHT, WIDTH = size
            return (HEIGHT, WIDTH)

        # get a reference image (from video or still)
        if pre_video_path:
            ref_frame = self.read_last_n_frames(pre_video_path, n_frames=1)[0]
            ref_pil = Image.fromarray(ref_frame)
        else:
            # Check if ref_image_path is a video file
            video_extensions = ['.mp4', '.avi', '.mov', '.mkv', '.flv', '.wmv', '.webm', '.m4v']
            is_video = any(ref_image_path.lower().endswith(ext) for ext in video_extensions)
            
            if is_video:
                # Extract first frame from video and save as image for debugging
                print(f"Detected video input: {ref_image_path}, extracting and saving first frame...")
                ref_frame = self.read_last_n_frames(ref_image_path, n_frames=1, target_fps=25, reverse=True)[0]
                
                # Save the first frame as an image
                first_frame_path = ref_image_path.rsplit('.', 1)[0] + '_first_frame.png'
                Image.fromarray(ref_frame).save(first_frame_path)
                print(f"Saved first frame to: {first_frame_path}")
                
                # Load the saved image as usual
                ref_pil = Image.open(first_frame_path).convert('RGB')
            else:
                ref_pil = Image.open(ref_image_path).convert('RGB')

        # --- Two-bin policy (match training) ---
        if hasattr(self, "infer_buckets") and self.size_bucket in self.infer_buckets:
            bucket = self.infer_buckets[self.size_bucket]
            HEIGHT, WIDTH = _closest_bucket_hw_for_image(ref_pil, bucket)
            return (int(HEIGHT), int(WIDTH))

        # --- Fallback: legacy area-based method ---
        HEIGHT, WIDTH = ref_pil.size[1], ref_pil.size[0]
        HEIGHT, WIDTH = self.get_size_less_than_area(HEIGHT, WIDTH, target_area=max_area)
        return (HEIGHT, WIDTH)


    def generate(
        self,
        input_prompt,
        ref_image_path,
        audio_path,
        enable_tts,
        tts_prompt_audio,
        tts_prompt_text,
        tts_text,
        num_repeat=1,
        max_area=720 * 1280,
        infer_frames=120,
        shift=5.0,
        sample_solver='unipc',
        sampling_steps=40,
        guide_scale=5.0,
        n_prompt="",
        seed=-1,
        offload_model=True,
        init_first_frame=False,
        dubbing_noise_strength=0.95,
    ):
        r"""
        Generates video frames from input image and text prompt using diffusion process.
        Combines the 5B TI2V base model with audio components from the 14B S2V model.

        Args:
            input_prompt (`str`):
                Text prompt for content generation.
            ref_image_path ('str'):
                Input image path
            audio_path ('str'):
                Audio for video driven
            num_repeat ('int'):
                Number of clips to generate; will be automatically adjusted based on the audio length
            max_area (`int`, *optional*, defaults to 720*1280):
                Maximum pixel area for latent space calculation. Controls video resolution scaling
            infer_frames (`int`, *optional*, defaults to 80):
                How many frames to generate per clips. The number should be 4n
            shift (`float`, *optional*, defaults to 5.0):
                Noise schedule shift parameter. Affects temporal dynamics
                [NOTE]: If you want to generate a 480p video, it is recommended to set the shift value to 3.0.
            sample_solver (`str`, *optional*, defaults to 'unipc'):
                Solver used to sample the video.
            sampling_steps (`int`, *optional*, defaults to 40):
                Number of diffusion sampling steps. Higher values improve quality but slow generation
            guide_scale (`float` or tuple[`float`], *optional*, defaults 5.0):
                Classifier-free guidance scale. Controls prompt adherence vs. creativity.
                If tuple, the first guide_scale will be used for low noise model and
                the second guide_scale will be used for high noise model.
            n_prompt (`str`, *optional*, defaults to ""):
                Negative prompt for content exclusion. If not given, use `config.sample_neg_prompt`
            seed (`int`, *optional*, defaults to -1):
                Random seed for noise generation. If -1, use random seed
            offload_model (`bool`, *optional*, defaults to True):
                If True, offloads models to CPU during generation to save VRAM
            init_first_frame (`bool`, *optional*, defaults to False):
                Whether to use the reference image as the first frame (i.e., standard image-to-video generation)
            dubbing_noise_strength (`float`, *optional*, defaults to 0.5):
                Noise strength for dubbing mode (only used when input is video). 
                0.0 = no noise (preserve video content), 1.0 = full noise (generate from scratch).
                This parameter is ignored when input is an image.

        Returns:
            torch.Tensor:
                Generated video frames tensor. Dimensions: (C, N H, W) where:
                - C: Color channels (3 for RGB)
                - N: Number of frames (81)
                - H: Frame height (from max_area)
                - W: Frame width from max_area)
        """
        # preprocess
        size = self.get_gen_size(
            size=None,
            max_area=max_area,
            ref_image_path=ref_image_path,
            pre_video_path=None)
        HEIGHT, WIDTH = size
        channel = 3

        resize_opreat = transforms.Resize(min(HEIGHT, WIDTH))
        crop_opreat = transforms.CenterCrop((HEIGHT, WIDTH))
        tensor_trans = transforms.ToTensor()

        ref_image = None
        motion_frames = None

        if ref_image is None:
            # Check if ref_image_path is a video file
            video_extensions = ['.mp4', '.avi', '.mov', '.mkv', '.flv', '.wmv', '.webm', '.m4v']
            is_video = any(ref_image_path.lower().endswith(ext) for ext in video_extensions)
            
            if is_video:
                # Save the first frame as an image
                first_frame_path = ref_image_path.rsplit('.', 1)[0] + '_first_frame.png'

                if not os.path.exists(first_frame_path):
                    # Extract first frame from video and save as image for debugging
                    print(f"Detected video input: {ref_image_path}, extracting and saving first frame...")
                    first_frame = self.read_last_n_frames(ref_image_path, n_frames=1, target_fps=self.fps, reverse=True)[0]
                    
                    Image.fromarray(first_frame).save(first_frame_path)
                    print(f"Saved first frame to: {first_frame_path}")

                ref_image = np.array(Image.open(first_frame_path).convert('RGB'))
            else:
                # Load image directly
                ref_image = np.array(Image.open(ref_image_path).convert('RGB'))
        if motion_frames is None:
            motion_frames = torch.zeros(
                [1, channel, self.num_motion_frames, HEIGHT, WIDTH],
                dtype=self.param_dtype,
                device=self.device)

        # extract audio emb
        if enable_tts is True:
            audio_path = self.tts(tts_prompt_audio, tts_prompt_text, tts_text)
        audio_emb, nr = self.encode_audio(audio_path, infer_frames=infer_frames)
        print(f'audio_emb: {audio_emb.shape}, number of repeat: {nr}, infer_frames: {infer_frames}')
        if num_repeat is None or num_repeat > nr:
            num_repeat = nr

        lat_motion_frames = (self.num_motion_frames + 3) // 4
        model_pic = crop_opreat(resize_opreat(Image.fromarray(ref_image)))

        ref_pixel_values = tensor_trans(model_pic)
        ref_pixel_values = ref_pixel_values.unsqueeze(1).unsqueeze(
            0) * 2 - 1.0  # b c 1 h w
        ref_pixel_values = ref_pixel_values.to(
            dtype=self.vae.dtype, device=self.vae.device)
        ref_latents = torch.stack(self.vae.encode([ref_pixel_values[i] for i in range(ref_pixel_values.shape[0])]))

        # encode the motion latents
        videos_last_frames = motion_frames.detach()
        drop_first_motion = self.drop_first_motion
        if init_first_frame:    # We do not use the reference image as the first frame
            drop_first_motion = False
            motion_frames[:, :, -6:] = ref_pixel_values
            motion_latents = torch.stack(self.vae.encode([motion_frames[i] for i in range(motion_frames.shape[0])]))
        else:
            # Calculate latent shape manually to skip VAE computation
            # VAE 2.2 compresses by 4 in time, 16 in space
            lat_t = (self.num_motion_frames + 3) // 4
            lat_h = HEIGHT // 16
            lat_w = WIDTH // 16
            # Create zeros directly
            motion_latents = torch.zeros(
                (motion_frames.shape[0], 48, lat_t, lat_h, lat_w), 
                dtype=self.param_dtype, 
                device=self.device
            )
            print(f"Skipping VAE encoding for motion frames, using zeros latent shape {motion_latents.shape}.")

        # # get pose cond input if need (S2V 5B does not support pose_video)
        # COND = self.load_pose_cond(
        #     num_repeat=num_repeat,
        #     infer_frames=infer_frames,
        #     size=size)

        seed = seed if seed >= 0 else random.randint(0, sys.maxsize)

        if n_prompt == "":
            n_prompt = self.sample_neg_prompt

        # Check if ref_image_path is a video for dubbing mode
        video_extensions = ['.mp4', '.avi', '.mov', '.mkv', '.flv', '.wmv', '.webm', '.m4v']
        is_video_input = any(ref_image_path.lower().endswith(ext) for ext in video_extensions)

        # preprocess
        if not self.t5_cpu:
            self.text_encoder.model.to(self.device)
            context = self.text_encoder([input_prompt], self.device)
            context_null = self.text_encoder([n_prompt], self.device)
            if offload_model:
                self.text_encoder.model.cpu()
        else:
            context = self.text_encoder([input_prompt], torch.device('cpu'))
            context_null = self.text_encoder([n_prompt], torch.device('cpu'))
            context = [t.to(self.device) for t in context]
            context_null = [t.to(self.device) for t in context_null]

        out = []
        # evaluation mode
        with (
                torch.amp.autocast('cuda', dtype=self.param_dtype),
                torch.no_grad(),
        ):
            for r in range(num_repeat):
                seed_g = torch.Generator(device=self.device)
                seed_g.manual_seed(seed + r)

                lat_target_frames = (infer_frames + 3 + self.num_motion_frames
                                    ) // 4 - lat_motion_frames
                target_shape = [lat_target_frames, HEIGHT // 16, WIDTH // 16]
                
                # Initialize noise - either from video latents (dubbing) or pure noise (generation)
                if is_video_input:
                    print(f"Dubbing mode: Loading video frames and adding noise (strength={dubbing_noise_strength})")
                    # Load video frames for this clip
                    left_frame_idx = r * infer_frames
                    right_frame_idx = r * infer_frames + infer_frames
                    
                    # Read video frames
                    vr = VideoReader(ref_image_path)
                    total_frames = len(vr)
                    print(f"r: {r}, Total frames: {total_frames}, left_frame_idx: {left_frame_idx}, right_frame_idx: {right_frame_idx}")
                    
                    # Calculate frame indices to sample
                    video_frame_indices = []
                    for i in range(infer_frames):
                        frame_idx = min(left_frame_idx + i, total_frames - 1)
                        video_frame_indices.append(frame_idx)
                    print(f"r: {r}, video_frame_indices: {video_frame_indices}")
                    
                    # Load and process frames
                    video_frames = vr.get_batch(video_frame_indices).asnumpy()  # [T, H, W, C]
                    
                    # Randomly sample a frame from current segment to update ref_img
                    random_frame_idx = np.random.randint(0, len(video_frames))
                    sampled_ref_frame = video_frames[random_frame_idx]
                    print(f"r: {r}, Updating ref_img with randomly sampled frame {random_frame_idx} from current video segment")
                    
                    # Process sampled frame to create new ref_latents
                    sampled_ref_pil = Image.fromarray(sampled_ref_frame)
                    sampled_ref_pil = crop_opreat(resize_opreat(sampled_ref_pil))
                    sampled_ref_tensor = tensor_trans(sampled_ref_pil)
                    sampled_ref_pixel_values = sampled_ref_tensor.unsqueeze(1).unsqueeze(0) * 2 - 1.0  # [1, C, 1, H, W]
                    sampled_ref_pixel_values = sampled_ref_pixel_values.to(dtype=self.vae.dtype, device=self.vae.device)
                    ref_latents = torch.stack(self.vae.encode([sampled_ref_pixel_values[i] for i in range(sampled_ref_pixel_values.shape[0])]))
                    print(f"r: {r}, Updated ref_latents shape: {ref_latents.shape}")
                    
                    # Convert to tensor and preprocess
                    video_tensor = []
                    for frame in video_frames:
                        frame_pil = Image.fromarray(frame)
                        frame_pil = crop_opreat(resize_opreat(frame_pil))
                        frame_tensor = tensor_trans(frame_pil)
                        video_tensor.append(frame_tensor)
                    
                    video_tensor = torch.stack(video_tensor)  # [T, C, H, W]
                    video_tensor = video_tensor.permute(1, 0, 2, 3).unsqueeze(0)  # [1, C, T, H, W]
                    video_tensor = video_tensor * 2 - 1.0  # normalize to [-1, 1]
                    video_tensor = video_tensor.to(dtype=self.vae.dtype, device=self.vae.device)
                    #print(f"r: {r}, video_tensor: {video_tensor.shape}")
                
                    input_motion_frames = videos_last_frames.to(dtype=self.vae.dtype, device=self.vae.device)
                    
                    # Concatenate along time dimension: [1, 3, motion_frames + infer_frames, H, W]
                    full_video_tensor = torch.cat([input_motion_frames, video_tensor], dim=2)
                    #print(f"r: {r}, full_video_tensor (motion + video): {full_video_tensor.shape}")
                    
                    # Encode the full sequence (motion + video frames)
                    full_video_latents = torch.stack(self.vae.encode([full_video_tensor[i] for i in range(full_video_tensor.shape[0])]))  # [1, 48, T_lat, H_lat, W_lat]
                    #print(f"r: {r}, full_video_latents after encoding: {full_video_latents.shape}")
                    
                    # Split into motion latents and target latents
                    # Motion latents: first lat_motion_frames
                    # Target latents: remaining frames
                    video_latents = full_video_latents[0, :, lat_motion_frames:]  # [48, target_frames, H_lat, W_lat]
                    
                    print(f"Video latents shape: {video_latents.shape}, target shape: [48, {target_shape[0]}, {target_shape[1]}, {target_shape[2]}]")
                    
                    # Handle shape mismatch: pad or trim to match target_shape
                    actual_frames = video_latents.shape[1]
                    target_frames = target_shape[0]
                    
                    if actual_frames < target_frames:
                        # Video is shorter than requested - pad by repeating last frame
                        padding_frames = target_frames - actual_frames
                        last_frame = video_latents[:, -1:, :, :]  # [48, 1, H, W]
                        padding = last_frame.repeat(1, padding_frames, 1, 1)  # [48, padding_frames, H, W]
                        video_latents = torch.cat([video_latents, padding], dim=1)  # [48, target_frames, H, W]
                        print(f"Padded video latents from {actual_frames} to {target_frames} frames by repeating last frame")
                    elif actual_frames > target_frames:
                        # Video is longer than requested - trim to target length
                        video_latents = video_latents[:, :target_frames, :, :]
                        print(f"Trimmed video latents from {actual_frames} to {target_frames} frames")
                    
                    # Add noise to video latents based on dubbing_noise_strength
                    # dubbing_noise_strength controls the timestep: 0.0 = t=0 (no noise), 1.0 = t=1000 (full noise)
                    noise_timestep = int(dubbing_noise_strength * self.num_train_timesteps)
                    
                    # Generate random noise with the same shape as video_latents
                    random_noise = torch.randn(
                        video_latents.shape[0],
                        video_latents.shape[1],
                        video_latents.shape[2],
                        video_latents.shape[3],
                        dtype=self.param_dtype,
                        device=self.device,
                        generator=seed_g
                    )
                    
                    # Add noise to video latents using the scheduler's noise formula
                    # For flow matching: x_t = (1-t) * x_0 + t * noise
                    t_normalized = noise_timestep / self.num_train_timesteps  # normalize to [0, 1]
                    noisy_latents = (1 - t_normalized) * video_latents + t_normalized * random_noise
                    
                    noise = [noisy_latents]
                    print(f"Initialized with noisy video latents at timestep {noise_timestep} (t={t_normalized:.3f})")
                else:
                    # Original behavior: pure random noise for image-to-video generation
                    noise = [
                        torch.randn(
                            48,  # Use 48 channels to match model's in_dim
                            target_shape[0],
                            target_shape[1],
                            target_shape[2],
                            dtype=self.param_dtype,
                            device=self.device,
                            generator=seed_g)
                    ]
                    print("Generation mode: Starting from pure random noise")
                
                max_seq_len = np.prod(target_shape) // 4

                if sample_solver == 'unipc':
                    sample_scheduler = FlowUniPCMultistepScheduler(
                        num_train_timesteps=self.num_train_timesteps,
                        shift=1,
                        use_dynamic_shifting=False)
                    sample_scheduler.set_timesteps(
                        sampling_steps, device=self.device, shift=shift)
                    timesteps = sample_scheduler.timesteps
                elif sample_solver == 'dpm++':
                    sample_scheduler = FlowDPMSolverMultistepScheduler(
                        num_train_timesteps=self.num_train_timesteps,
                        shift=1,
                        use_dynamic_shifting=False)
                    sampling_sigmas = get_sampling_sigmas(sampling_steps, shift)
                    timesteps, _ = retrieve_timesteps(
                        sample_scheduler,
                        device=self.device,
                        sigmas=sampling_sigmas)
                else:
                    raise NotImplementedError("Unsupported solver.")
                
                # For dubbing mode: only denoise from the noise level we added
                print(f"Before dubbing filter: {len(timesteps)} timesteps")
                if is_video_input and dubbing_noise_strength < 1.0:
                    noise_timestep = int(dubbing_noise_strength * self.num_train_timesteps)
                    # Filter timesteps to only include those <= noise_timestep
                    original_timesteps = timesteps.clone()
                    timesteps = timesteps[timesteps <= noise_timestep]
                    print(f"Dubbing mode: Filtered from {len(original_timesteps)} to {len(timesteps)} denoising steps (noise_timestep={noise_timestep})")
                    print(f"Original timesteps: {original_timesteps.tolist()}")
                    print(f"Filtered timesteps: {timesteps.tolist()}")
                    if len(timesteps) == 0:
                        # If no timesteps remain, use at least one step from the noise level
                        timesteps = torch.tensor([noise_timestep], device=self.device)
                        print(f"Warning: No timesteps <= {noise_timestep}, using single step at {noise_timestep}")

                latents = deepcopy(noise)
                with torch.no_grad():
                    left_idx = r * infer_frames
                    right_idx = r * infer_frames + infer_frames
                    # cond_latents = COND[0] * 0  # S2V 5B does not support pose_video
                    # cond_latents = cond_latents.to(
                    #     dtype=self.param_dtype, device=self.device)
                    audio_input = audio_emb[..., left_idx:right_idx]
                input_motion_latents = motion_latents.clone()

                arg_c = {
                    'context': context[0:1],
                    'seq_len': max_seq_len,
                    # 'cond_states': cond_latents,
                    "motion_latents": input_motion_latents,    # [1, 48, 19, 36, 36]
                    'ref_latents': ref_latents,                # [1, 48, 1, 36, 36]
                    "audio_input": audio_input,                # [1, 25, 1024, 120]
                    "motion_frames": [self.num_motion_frames, lat_motion_frames],    # [73, 19]
                    "drop_motion_frames": drop_first_motion and r == 0,    # drop first motion = True
                }
                if guide_scale > 1:
                    arg_null = {
                        'context': context_null[0:1],
                        'seq_len': max_seq_len,
                        # 'cond_states': cond_latents,
                        "motion_latents": input_motion_latents,
                        'ref_latents': ref_latents,
                        "audio_input": 0.0 * audio_input,    # uncond audio input
                        "motion_frames": [
                            self.num_motion_frames, lat_motion_frames
                        ],
                        "drop_motion_frames": drop_first_motion and r == 0,
                    }
                if offload_model or self.init_on_cpu:
                    self.noise_model.to(self.device)
                    torch.cuda.empty_cache()

                for i, t in enumerate(tqdm(timesteps)):
                    latent_model_input = [latents[0]]  # [48, 30, 36, 36]
                    timestep = [t]

                    timestep = torch.stack(timestep).to(self.device)

                    noise_pred_cond = self.noise_model(
                        latent_model_input, t=timestep, **arg_c)

                    if guide_scale > 1:
                        noise_pred_uncond = self.noise_model(
                            latent_model_input, t=timestep, **arg_null)
                        noise_pred = [
                            u + guide_scale * (c - u)
                            for c, u in zip(noise_pred_cond, noise_pred_uncond)
                        ]
                    else:
                        noise_pred = noise_pred_cond

                    temp_x0 = sample_scheduler.step(
                        noise_pred[0].unsqueeze(0),
                        t,
                        latents[0].unsqueeze(0),
                        return_dict=False,
                        generator=seed_g)[0]
                    latents[0] = temp_x0.squeeze(0)

                if offload_model:
                    self.noise_model.cpu()
                    torch.cuda.synchronize()
                    torch.cuda.empty_cache()
                latents = torch.stack(latents)
                append_ref_latents=False
                if not (drop_first_motion and r == 0):
                    decode_latents = torch.cat([motion_latents, latents], dim=2)
                    print(f'r: {r}, motion_latents: {motion_latents.shape}, latents: {latents.shape}, decode_latents: {decode_latents.shape}')
                else:
                    decode_latents = torch.cat([ref_latents, latents], dim=2) if append_ref_latents else latents    # [1, 48, 31, 36, 36]
                    print(f'r: {r}, ref_latents: {ref_latents.shape}, latents: {latents.shape}, decode_latents: {decode_latents.shape}')
                print(f'decode_latents: {decode_latents.shape}')
                image = torch.stack(self.vae.decode([decode_latents[i] for i in range(decode_latents.shape[0])]))  # 31 -> 121;
                print(f'image: {image.shape}')
                image = image[:, :, -(infer_frames):]    # [1, 3, 120, 576, 576]
                print(f'image infer_frames: {image.shape}')
                print(f'drop_first_motion: {drop_first_motion}, r: {r}')
                if (drop_first_motion and r == 0):
                    image = image[:, :, 3:] if append_ref_latents else image    # [1, 3, 117, 576, 576]
                print(f'image after drop_first_motion: {image.shape}')

                overlap_frames_num = min(self.num_motion_frames, image.shape[2])    # 73
                print(f'overlap_frames_num: {overlap_frames_num}')
                videos_last_frames = torch.cat([    # [1, 3, 73, 576, 576]
                    videos_last_frames[:, :, overlap_frames_num:],
                    image[:, :, -overlap_frames_num:]
                ],
                                               dim=2)
                videos_last_frames = videos_last_frames.to(
                    dtype=motion_latents.dtype, device=motion_latents.device)    # [1, 3, 73, 576, 576]
                motion_latents = torch.stack(
                    self.vae.encode([videos_last_frames[i] for i in range(videos_last_frames.shape[0])]))    # [1, 48, 19, 36, 36]
                out.append(image.cpu())

        videos = torch.cat(out, dim=2)
        del noise, latents
        del sample_scheduler
        if offload_model:
            gc.collect()
            torch.cuda.synchronize()
        if dist.is_initialized():
            dist.barrier()

        return videos[0] if self.rank == 0 else None

    def tts(self, tts_prompt_audio, tts_prompt_text, tts_text):
        if not hasattr(self, 'cosyvoice'):
            self.load_tts()
        speech_list = []
        from cosyvoice.utils.file_utils import load_wav
        import torchaudio
        prompt_speech_16k = load_wav(tts_prompt_audio, 16000)
        if tts_prompt_text is not None:
            for i in self.cosyvoice.inference_zero_shot(tts_text, tts_prompt_text, prompt_speech_16k):
                speech_list.append(i['tts_speech'])
        else:
            for i in self.cosyvoice.inference_cross_lingual(tts_text, prompt_speech_16k):
                speech_list.append(i['tts_speech'])
        torchaudio.save('tts.wav', torch.concat(speech_list, dim=1), self.cosyvoice.sample_rate)
        return 'tts.wav'

    def load_tts(self):
        if not os.path.exists('CosyVoice'):
            from wan.utils.utils import download_cosyvoice_repo
            download_cosyvoice_repo('CosyVoice')
        if not os.path.exists('CosyVoice2-0.5B'):
            from wan.utils.utils import download_cosyvoice_model
            download_cosyvoice_model('CosyVoice2-0.5B', 'CosyVoice2-0.5B')
        sys.path.append('CosyVoice')
        sys.path.append('CosyVoice/third_party/Matcha-TTS')
        from cosyvoice.cli.cosyvoice import CosyVoice2
        self.cosyvoice = CosyVoice2('CosyVoice2-0.5B')
