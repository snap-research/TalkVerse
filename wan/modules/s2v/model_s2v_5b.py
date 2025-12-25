# Copyright 2024-2025 The Alibaba Wan Team Authors. All rights reserved.
import math
import types
from copy import deepcopy

import numpy as np
import torch
import torch.cuda.amp as amp
import torch.nn as nn
from torch.utils.checkpoint import checkpoint
from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.models.modeling_utils import ModelMixin
from einops import rearrange

from ...distributed.sequence_parallel import (
    distributed_attention,
    gather_forward,
    get_rank,
    get_world_size,
)
from ..model import (
    Head,
    WanAttentionBlock,
    WanLayerNorm,
    WanSelfAttention,
    flash_attention,
    rope_params,
    sinusoidal_embedding_1d,
)
from .audio_utils import AudioInjector_WAN, CausalAudioEncoder
from .s2v_utils import rope_precompute


def zero_module(module):
    """
    Zero out the parameters of a module and return it.
    """
    for p in module.parameters():
        p.detach().zero_()
    return module


def torch_dfs(model: nn.Module, parent_name='root'):
    module_names, modules = [], []
    current_name = parent_name if parent_name else 'root'
    module_names.append(current_name)
    modules.append(model)

    for name, child in model.named_children():
        if parent_name:
            child_name = f'{parent_name}.{name}'
        else:
            child_name = name
        child_modules, child_names = torch_dfs(child, child_name)
        module_names += child_names
        modules += child_modules
    return modules, module_names


@torch.amp.autocast('cuda', enabled=False)
def rope_apply(x, grid_sizes, freqs, start=None):
    n, c = x.size(2), x.size(3) // 2
    # loop over samples
    output = []
    for i, _ in enumerate(x):
        s = x.size(1)
        x_i = torch.view_as_complex(x[i, :s].to(torch.float64).reshape(
            s, n, -1, 2))
        freqs_i = freqs[i, :s]
        # apply rotary embedding
        x_i = torch.view_as_real(x_i * freqs_i).flatten(2)
        x_i = torch.cat([x_i, x[i, s:]])
        # append to collection
        output.append(x_i)
    return torch.stack(output).float()


@torch.amp.autocast('cuda', enabled=False)
def rope_apply_usp(x, grid_sizes, freqs):
    s, n, c = x.size(1), x.size(2), x.size(3) // 2
    # loop over samples
    output = []
    for i, _ in enumerate(x):
        s = x.size(1)
        # precompute multipliers
        x_i = torch.view_as_complex(x[i, :s].to(torch.float64).reshape(
            s, n, -1, 2))
        freqs_i = freqs[i]
        freqs_i_rank = freqs_i
        x_i = torch.view_as_real(x_i * freqs_i_rank).flatten(2)
        x_i = torch.cat([x_i, x[i, s:]])
        # append to collection
        output.append(x_i)
    return torch.stack(output).float()


def sp_attn_forward_s2v_5b(self,
                           x,
                           seq_lens,
                           grid_sizes,
                           freqs,
                           dtype=torch.bfloat16):
    b, s, n, d = *x.shape[:2], self.num_heads, self.head_dim
    half_dtypes = (torch.float16, torch.bfloat16)

    def half(x):
        return x if x.dtype in half_dtypes else x.to(dtype)

    # query, key, value function
    def qkv_fn(x):
        q = self.norm_q(self.q(x)).view(b, s, n, d)
        k = self.norm_k(self.k(x)).view(b, s, n, d)
        v = self.v(x).view(b, s, n, d)
        return q, k, v

    q, k, v = qkv_fn(x)
    q = rope_apply_usp(q, grid_sizes, freqs)
    k = rope_apply_usp(k, grid_sizes, freqs)

    x = distributed_attention(
        half(q),
        half(k),
        half(v),
        seq_lens,
        window_size=self.window_size,
    )

    # output
    x = x.flatten(2)
    x = self.o(x)
    return x


class Head_S2V_5B(Head):

    def forward(self, x, e):
        """
        Args:
            x(Tensor): Shape [B, L1, C]
            e(Tensor): Shape [B, L1, C]
        """
        assert e.dtype == torch.float32
        with amp.autocast(dtype=torch.float32):
            e = (self.modulation + e.unsqueeze(1)).chunk(2, dim=1)
            x = (self.head(self.norm(x) * (1 + e[1]) + e[0]))
        return x


class WanS2V5BSelfAttention(WanSelfAttention):

    def forward(self, x, seq_lens, grid_sizes, freqs):
        """
        Args:
            x(Tensor): Shape [B, L, num_heads, C / num_heads]
            seq_lens(Tensor): Shape [B]
            grid_sizes(Tensor): Shape [B, 3], the second dimension contains (F, H, W)
            freqs(Tensor): Rope freqs, shape [1024, C / num_heads / 2]
        """
        b, s, n, d = *x.shape[:2], self.num_heads, self.head_dim

        # query, key, value function
        def qkv_fn(x):
            q = self.norm_q(self.q(x)).view(b, s, n, d)
            k = self.norm_k(self.k(x)).view(b, s, n, d)
            v = self.v(x).view(b, s, n, d)
            return q, k, v

        q, k, v = qkv_fn(x)

        x = flash_attention(
            q=rope_apply(q, grid_sizes, freqs),
            k=rope_apply(k, grid_sizes, freqs),
            v=v,
            k_lens=seq_lens,
            window_size=self.window_size)

        # output
        x = x.flatten(2)
        x = self.o(x)
        return x


class WanS2V5BAttentionBlock(WanAttentionBlock):

    def __init__(self,
                 dim,
                 ffn_dim,
                 num_heads,
                 window_size=(-1, -1),
                 qk_norm=True,
                 cross_attn_norm=False,
                 eps=1e-6):
        super().__init__(dim, ffn_dim, num_heads, window_size, qk_norm,
                         cross_attn_norm, eps)
        self.self_attn = WanS2V5BSelfAttention(dim, num_heads, window_size,
                                               qk_norm, eps)

    def forward(self, x, e, seq_lens, grid_sizes, freqs, context, context_lens):
        assert e[0].dtype == torch.float32
        seg_idx = e[1].item()
        seg_idx = min(max(0, seg_idx), x.size(1))
        seg_idx = [0, seg_idx, x.size(1)]
        e = e[0]
        modulation = self.modulation.unsqueeze(2)
        with amp.autocast(dtype=torch.float32):
            e = (modulation + e).chunk(6, dim=1)
        assert e[0].dtype == torch.float32

        e = [element.squeeze(1) for element in e]
        norm_x = self.norm1(x).float()
        parts = []
        for i in range(2):
            parts.append(norm_x[:, seg_idx[i]:seg_idx[i + 1]] *
                         (1 + e[1][:, i:i + 1]) + e[0][:, i:i + 1])
        norm_x = torch.cat(parts, dim=1)
        # self-attention
        y = self.self_attn(norm_x, seq_lens, grid_sizes, freqs)
        with amp.autocast(dtype=torch.float32):
            z = []
            for i in range(2):
                z.append(y[:, seg_idx[i]:seg_idx[i + 1]] * e[2][:, i:i + 1])
            y = torch.cat(z, dim=1)
            x = x + y
        # cross-attention & ffn function
        def cross_attn_ffn(x, context, context_lens, e):
            x = x + self.cross_attn(self.norm3(x), context, context_lens)
            norm2_x = self.norm2(x).float()
            parts = []
            for i in range(2):
                parts.append(norm2_x[:, seg_idx[i]:seg_idx[i + 1]] *
                             (1 + e[4][:, i:i + 1]) + e[3][:, i:i + 1])
            norm2_x = torch.cat(parts, dim=1)
            y = self.ffn(norm2_x)
            with amp.autocast(dtype=torch.float32):
                z = []
                for i in range(2):
                    z.append(y[:, seg_idx[i]:seg_idx[i + 1]] * e[5][:, i:i + 1])
                y = torch.cat(z, dim=1)
                x = x + y
            return x

        x = cross_attn_ffn(x, context, context_lens, e)
        return x


class WanModel_S2V_5B(ModelMixin, ConfigMixin):
    ignore_for_config = [
        'args', 'kwargs', 'patch_size', 'cross_attn_norm', 'qk_norm',
        'text_dim', 'window_size'
    ]
    _no_split_modules = ['WanS2V5BAttentionBlock']

    @register_to_config
    def __init__(
            self,
            cond_dim=0,
            audio_dim=1024,  # Match S2V 14B audio dimension
            num_audio_token=4,
            enable_adain=False,
            adain_mode="attn_norm",
            audio_inject_layers=[0, 3, 6, 9, 12, 15, 18, 21, 24, 27, 29],  # Adapted for 30 layers
            zero_init=False,
            zero_timestep=False,
            enable_motioner=False,  # Disabled for TI2V base
            add_last_motion=True,
            enable_tsm=False,
            trainable_token_pos_emb=False,
            motion_token_num=1024,
            enable_framepack=True,  # Use framepack instead of motioner
            framepack_drop_mode="padd",
            model_type='s2v',
            patch_size=(1, 2, 2),
            text_len=512,
            in_dim=48,
            dim=3072,  # TI2V 5B dimension
            ffn_dim=14336,  # TI2V 5B ffn dimension
            freq_dim=256,
            text_dim=4096,
            out_dim=16,
            num_heads=24,  # TI2V 5B num_heads
            num_layers=30,  # TI2V 5B num_layers
            window_size=(-1, -1),
            qk_norm=True,
            cross_attn_norm=True,
            eps=1e-6,
            v_scale=1.0,
            *args,
            **kwargs):
        super().__init__()

        assert model_type == 's2v'
        self.model_type = model_type

        self.patch_size = patch_size
        self.text_len = text_len
        self.in_dim = in_dim
        self.dim = dim
        self.ffn_dim = ffn_dim
        self.freq_dim = freq_dim
        self.text_dim = text_dim
        self.out_dim = out_dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.window_size = window_size
        self.qk_norm = qk_norm
        self.cross_attn_norm = cross_attn_norm
        self.eps = eps
        self.v_scale = v_scale

        # embeddings
        self.patch_embedding = nn.Conv3d(
            in_dim, dim, kernel_size=patch_size, stride=patch_size)
        self.text_embedding = nn.Sequential(
            nn.Linear(text_dim, dim), nn.GELU(approximate='tanh'),
            nn.Linear(dim, dim))

        self.time_embedding = nn.Sequential(
            nn.Linear(freq_dim, dim), nn.SiLU(), nn.Linear(dim, dim))
        self.time_projection = nn.Sequential(nn.SiLU(), nn.Linear(dim, dim * 6))

        # blocks - Use TI2V 5B architecture with S2V attention blocks
        self.blocks = nn.ModuleList([
            WanS2V5BAttentionBlock(dim, ffn_dim, num_heads, window_size, qk_norm,
                                   cross_attn_norm, eps)
            for _ in range(num_layers)
        ])

        # head
        self.head = Head_S2V_5B(dim, out_dim, patch_size, eps)

        # buffers (don't use register_buffer otherwise dtype will be changed in to())
        assert (dim % num_heads) == 0 and (dim // num_heads) % 2 == 0
        d = dim // num_heads
        self.freqs = torch.cat([
            rope_params(1024, d - 4 * (d // 6)),
            rope_params(1024, 2 * (d // 6)),
            rope_params(1024, 2 * (d // 6))
        ],
                               dim=1)

        # initialize weights
        self.init_weights()

        self.use_context_parallel = False  # will modify in _configure_model func

        # Audio injection components
        # Commented out cond_encoder since it's not used in the forward pass
        # if cond_dim > 0:
        #     self.cond_encoder = nn.Conv3d(
        #         cond_dim,
        #         self.dim,
        #         kernel_size=self.patch_size,
        #         stride=self.patch_size)
        
        self.enable_adain = enable_adain

        self.casual_audio_encoder = CausalAudioEncoder(
            dim=audio_dim,  # Input dimension from wav2vec
            out_dim=self.dim,  # Output to transformer dimension directly
            num_token=num_audio_token,
            need_global=enable_adain)
        
        all_modules, all_modules_names = torch_dfs(
            self.blocks, parent_name="root.transformer_blocks")

        self.audio_injector = AudioInjector_WAN(
            all_modules,
            all_modules_names,
            dim=self.dim,  # Use transformer dimension directly
            num_heads=self.num_heads,  # Use transformer num_heads
            inject_layer=audio_inject_layers,
            root_net=self,
            enable_adain=enable_adain,
            adain_dim=self.dim,  # Use transformer dimension
            need_adain_ont=adain_mode != "attn_norm",
        )
        
        # Add projection layer for audio injector output to transformer dimension
        self.adain_mode = adain_mode

        if enable_framepack:
            self.trainable_cond_mask = nn.Embedding(3, self.dim)
        else:
            self.trainable_cond_mask = nn.Embedding(2, self.dim)

        if zero_init:
            self.zero_init_weights()

        self.zero_timestep = zero_timestep  # Whether to assign 0 value timestep to ref/motion

        # Motion handling - simplified for TI2V base
        self.enable_motioner = enable_motioner
        self.add_last_motion = add_last_motion
        self.enable_framepack = enable_framepack

        if enable_framepack:
            from .motioner import FramePackMotioner
            self.frame_packer = FramePackMotioner(
                inner_dim=self.dim,
                num_heads=self.num_heads,
                zip_frame_buckets=[1, 2, 16],
                drop_mode=framepack_drop_mode)

    def zero_init_weights(self):
        print(f"************ [DEBUG] Zeroing init weights ************")
        with torch.no_grad():
            self.trainable_cond_mask = zero_module(self.trainable_cond_mask)
            # Commented out cond_encoder since it's not used
            # if hasattr(self, "cond_encoder"):
            #     self.cond_encoder = zero_module(self.cond_encoder)

            for i in range(self.audio_injector.injector.__len__()):
                self.audio_injector.injector[i].o = zero_module(
                    self.audio_injector.injector[i].o)
                if self.enable_adain:
                    self.audio_injector.injector_adain_layers[
                        i].linear = zero_module(
                            self.audio_injector.injector_adain_layers[i].linear)
            
            # Zero initialize the CausalAudioEncoder components
            self.casual_audio_encoder.encoder = zero_module(self.casual_audio_encoder.encoder)
            # Zero out the weights parameter directly (it's a Parameter, not a Module)
            with torch.no_grad():
                self.casual_audio_encoder.weights.zero_()
                    

    def after_transformer_block(self, block_idx, hidden_states):
        if block_idx in self.audio_injector.injected_block_id.keys():
            audio_attn_id = self.audio_injector.injected_block_id[block_idx]
            audio_emb = self.merged_audio_emb  # b f n c       [1, 30, 5, 3072]   30x4=120 frames, 5 layers feature
            num_frames = audio_emb.shape[1]

            if self.use_context_parallel:
                hidden_states = gather_forward(hidden_states, dim=1)

            input_hidden_states = hidden_states[:, :self.
                                                original_seq_len].clone(
                                                )  # b (f h w) c    [1, 12090, 3072]     12090= 31x30x52/4; frames=31, width=52, height=30, t_downsample=4

            input_hidden_states = rearrange(
                input_hidden_states, "b (t n) c -> (b t) n c", t=num_frames)

            # Use input hidden states directly for injector (no projection needed)
            input_hidden_states_audio = input_hidden_states

            if self.enable_adain and self.adain_mode == "attn_norm":
                audio_emb_global = self.audio_emb_global
                audio_emb_global = rearrange(audio_emb_global,
                                             "b t n c -> (b t) n c")
                adain_hidden_states = self.audio_injector.injector_adain_layers[
                    audio_attn_id](
                        input_hidden_states_audio, temb=audio_emb_global[:, 0])
                attn_hidden_states = adain_hidden_states
            else:
                attn_hidden_states = self.audio_injector.injector_pre_norm_feat[
                    audio_attn_id](
                        input_hidden_states_audio)
            audio_emb = rearrange(
                audio_emb, "b t n c -> (b t) n c", t=num_frames)
            attn_audio_emb = audio_emb
            
            residual_out = self.audio_injector.injector[audio_attn_id](
                x=attn_hidden_states,
                context=attn_audio_emb,
                v_scale=1.0,                  # DEBUG: Increase the audio attention strength
                context_lens=torch.ones(
                    attn_hidden_states.shape[0],
                    dtype=torch.long,
                    device=attn_hidden_states.device) * attn_audio_emb.shape[1])
            
            # Use residual output directly (no projection needed)
            # residual_out = residual_out
            residual_out = rearrange(
                residual_out, "(b t) n c -> b (t n) c", t=num_frames)
            hidden_states[:, :self.
                          original_seq_len] = hidden_states[:, :self.
                                                            original_seq_len] + residual_out

            if self.use_context_parallel:
                hidden_states = torch.chunk(
                    hidden_states, get_world_size(), dim=1)[get_rank()]

        return hidden_states

    def process_motion_frame_pack(self,
                                  motion_latents,
                                  drop_motion_frames=False,
                                  add_last_motion=2):
        flattern_mot, mot_remb = self.frame_packer(motion_latents,
                                                   add_last_motion)
        if drop_motion_frames:
            return [m[:, :0] for m in flattern_mot
                   ], [m[:, :0] for m in mot_remb]
        else:
            return flattern_mot, mot_remb

    def inject_motion(self,
                      x,
                      seq_lens,
                      rope_embs,
                      mask_input,
                      motion_latents,
                      drop_motion_frames=False,
                      add_last_motion=True):
        # Use frame packing path for 5B
        if self.enable_framepack:
            mot, mot_remb = self.process_motion_frame_pack(
                motion_latents,
                drop_motion_frames=drop_motion_frames,
                add_last_motion=add_last_motion)
        else:
            mot, mot_remb = [], []

        if len(mot) > 0:
            x = [torch.cat([u, m], dim=1) for u, m in zip(x, mot)]
            seq_lens = seq_lens + torch.tensor([r.size(1) for r in mot],
                                               dtype=torch.long)
            rope_embs = [
                torch.cat([u, m], dim=1) for u, m in zip(rope_embs, mot_remb)
            ]
            mask_input = [
                torch.cat([
                    m, 2 * torch.ones([1, u.shape[1] - m.shape[1]],
                                      device=m.device,
                                      dtype=m.dtype)
                ],
                          dim=1) for m, u in zip(mask_input, x)
            ]
        return x, seq_lens, rope_embs, mask_input

    def forward(
            self,
            x,
            t,
            context,
            seq_len,
            ref_latents=None,
            motion_latents=None,
            cond_states=None,
            audio_input=None,
            motion_frames=[17, 5],
            add_last_motion=2,
            drop_motion_frames=False,
            y=None,
            *extra_args,
            **extra_kwargs):
        """
        Forward pass consistent with WanModel_S2V (14B) preprocessing, audio handling, and motion injection,
        adapted to the 5B base architecture.
        """
        # Audio processing (match 14B)
        if audio_input is not None:
            audio_input = torch.cat([
                audio_input[..., 0:1].repeat(1, 1, 1, motion_frames[0]),
                audio_input
            ], dim=-1)
            audio_emb_res = self.casual_audio_encoder(audio_input)
            if self.enable_adain:
                audio_emb_global, audio_emb = audio_emb_res
                self.audio_emb_global = audio_emb_global[:,
                                                         motion_frames[1]:].clone()
            else:
                audio_emb = audio_emb_res
            
            # Use audio embedding directly (no projection needed)
            # audio_emb = audio_emb
            self.merged_audio_emb = audio_emb[:, motion_frames[1]:, :]
        else:
            batch_size = len(x)
            frame_num = x[0].shape[1] if len(x) > 0 else 81
            self.merged_audio_emb = torch.zeros(batch_size, frame_num, 4,
                                                self.dim,  # Already projected to transformer dimension
                                                device=next(self.parameters()).device)

        device = self.patch_embedding.weight.device
        if self.freqs.device != device:
            self.freqs = self.freqs.to(device)

        # embeddings for noisy latents and cond states
        x = [self.patch_embedding(u.unsqueeze(0)) for u in x]
        # cond = [self.cond_encoder(c.unsqueeze(0)) for c in cond_states]
        # TODO: skipping cond_states for now
        # x = [x_ + pose for x_, pose in zip(x, cond)]

        grid_sizes = torch.stack(
            [torch.tensor(u.shape[2:], dtype=torch.long) for u in x])
        x = [u.flatten(2).transpose(1, 2) for u in x]
        seq_lens = torch.tensor([u.size(1) for u in x], dtype=torch.long)

        # reference latents path
        self.lat_motion_frames = motion_latents[0].shape[1] if motion_latents is not None else 0
        ref = [self.patch_embedding(r.unsqueeze(0)) for r in ref_latents]
        batch_size = len(ref)
        height, width = ref[0].shape[3], ref[0].shape[4]
        ref_grid_sizes = [[
            torch.tensor([40, 0, 0]).unsqueeze(0).repeat(batch_size, 1),
            torch.tensor([41, height, width]).unsqueeze(0).repeat(batch_size, 1),
            torch.tensor([1, height, width]).unsqueeze(0).repeat(batch_size, 1),
        ]]
        ref = [r.flatten(2).transpose(1, 2) for r in ref]
        self.original_seq_len = seq_lens[0]
        seq_lens = seq_lens + torch.tensor([r.size(1) for r in ref],
                                           dtype=torch.long)
        original_grid_sizes = deepcopy(grid_sizes)
        grid_sizes = [[torch.zeros_like(grid_sizes), grid_sizes, grid_sizes]] + ref_grid_sizes
        x = [torch.cat([u, r], dim=1) for u, r in zip(x, ref)]

        # mask initialization (0: noisy, 1: ref, 2: motion)
        mask_input = [
            torch.zeros([1, u.shape[1]], dtype=torch.long, device=x[0].device)
            for u in x
        ]
        for i in range(len(mask_input)):
            mask_input[i][:, self.original_seq_len:] = 1

        # compute rope embeddings
        x_cat = torch.cat(x)
        b, s, n, d = x_cat.size(0), x_cat.size(1), self.num_heads, self.dim // self.num_heads
        self.pre_compute_freqs = rope_precompute(
            x_cat.detach().view(b, s, n, d), grid_sizes, self.freqs, start=None)

        x = [u.unsqueeze(0) for u in x_cat]
        self.pre_compute_freqs = [u.unsqueeze(0) for u in self.pre_compute_freqs]

        # inject motion via framepack
        # print(f"x shape before inject motion: {x[0].shape}")
        x, seq_lens, self.pre_compute_freqs, mask_input = self.inject_motion(
            x,
            seq_lens,
            self.pre_compute_freqs,
            mask_input,
            motion_latents,
            drop_motion_frames=drop_motion_frames,
            add_last_motion=add_last_motion)
        # print(f"x shape after inject motion: {x[0].shape}")

        x = torch.cat(x, dim=0)
        self.pre_compute_freqs = torch.cat(self.pre_compute_freqs, dim=0)
        mask_input = torch.cat(mask_input, dim=0)

        x = x + self.trainable_cond_mask(mask_input).to(x.dtype)

        # time embeddings
        if self.zero_timestep:
            t = torch.cat([t, torch.zeros([1], dtype=t.dtype, device=t.device)])
        with amp.autocast(dtype=torch.float32):
            e = self.time_embedding(
                sinusoidal_embedding_1d(self.freq_dim, t).float())
            e0 = self.time_projection(e).unflatten(1, (6, self.dim))
            assert e.dtype == torch.float32 and e0.dtype == torch.float32

        if self.zero_timestep:
            e = e[:-1]
            zero_e0 = e0[-1:]
            e0 = e0[:-1]
            e0 = torch.cat([
                e0.unsqueeze(2),
                zero_e0.unsqueeze(2).repeat(e0.size(0), 1, 1, 1)
            ], dim=2)
            e0 = [e0, self.original_seq_len]
        else:
            e0 = e0.unsqueeze(2).repeat(1, 1, 2, 1)
            e0 = [e0, 0]

        # context
        context_lens = None
        context = self.text_embedding(
            torch.stack([
                torch.cat(
                    [u, u.new_zeros(self.text_len - u.size(0), u.size(1))])
                for u in context
            ]))

        if self.use_context_parallel:
            sp_rank = get_rank()
            x = torch.chunk(x, get_world_size(), dim=1)
            sq_size = [u.shape[1] for u in x]
            sq_start_size = sum(sq_size[:sp_rank])
            x = x[sp_rank]
            seg_idx = e0[1] - sq_start_size
            e0[1] = seg_idx
            self.pre_compute_freqs = torch.chunk(
                self.pre_compute_freqs, get_world_size(), dim=1)
            self.pre_compute_freqs = self.pre_compute_freqs[sp_rank]

        kwargs = dict(
            e=e0,
            seq_lens=seq_lens,
            grid_sizes=grid_sizes,
            freqs=self.pre_compute_freqs,
            context=context,
            context_lens=context_lens)
        
        # Apply gradient checkpointing to transformer blocks if enabled
        if hasattr(self, 'use_gradient_checkpointing') and self.use_gradient_checkpointing:
            # print(f"[DEBUG] Using gradient checkpointing for {len(self.blocks)} transformer blocks")
            # Use gradient checkpointing for transformer blocks
            for idx, block in enumerate(self.blocks):
                def checkpointed_forward(hidden_states, block_idx, block, **block_kwargs):
                    x = block(hidden_states, **block_kwargs)
                    return self.after_transformer_block(block_idx, x)
                
                x = checkpoint(
                    checkpointed_forward, 
                    x, 
                    idx, 
                    block, 
                    use_reentrant=False,
                    **kwargs
                )
        else:
            # Standard forward pass without gradient checkpointing
            # print(f"[DEBUG] Using standard forward pass for {len(self.blocks)} transformer blocks")
            for idx, block in enumerate(self.blocks):
                x = block(x, **kwargs)
                x = self.after_transformer_block(idx, x)

        # Context Parallel gather
        if self.use_context_parallel:
            x = gather_forward(x.contiguous(), dim=1)

        # keep only original seq, head and unpatchify
        x = x[:, :self.original_seq_len]     # [1, 12643, 3072] -> [1, 11700, 3072]    11700= 30x30x52/4; frames=30, width=52, height=30, t_downsample=4
        x = self.head(x, e)                 # [1, 11700, 3072] -> [1, 11700, 192]
        x = self.unpatchify(x, original_grid_sizes)
        return [u.float() for u in x]

    def unpatchify(self, x, grid_sizes):
        """
        Reconstruct video tensors from patch embeddings.
        """
        c = self.out_dim
        out = []
        for u, v in zip(x, grid_sizes.tolist()):
            u = u[:math.prod(v)].view(*v, *self.patch_size, c)
            u = torch.einsum('fhwpqrc->cfphqwr', u)
            u = u.reshape(c, *[i * j for i, j in zip(v, self.patch_size)])
            out.append(u)
        return out

    def init_weights(self):
        """
        Initialize model parameters using Xavier initialization.
        """
        # basic init
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

        # init embeddings
        nn.init.xavier_uniform_(self.patch_embedding.weight.flatten(1))
        for m in self.text_embedding.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=.02)
        for m in self.time_embedding.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=.02)

        # init output layer
        nn.init.zeros_(self.head.head.weight)
