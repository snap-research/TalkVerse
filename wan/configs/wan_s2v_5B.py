# Copyright 2024-2025 The Alibaba Wan Team Authors. All rights reserved.
from easydict import EasyDict

from .shared_config import wan_shared_cfg

#------------------------ Wan S2V 5B ------------------------#

s2v_5B = EasyDict(__name__='Config: Wan S2V 5B')
s2v_5B.update(wan_shared_cfg)

# t5
s2v_5B.t5_checkpoint = 'models_t5_umt5-xxl-enc-bf16.pth'
s2v_5B.t5_tokenizer = 'google/umt5-xxl'

# vae - Use VAE2.2 from TI2V 5B (larger compression)
s2v_5B.vae_checkpoint = 'Wan2.2_VAE.pth'
s2v_5B.vae_stride = (4, 16, 16)

# wav2vec
s2v_5B.wav2vec = "wav2vec2-large-xlsr-53-english"

# transformer - Based on TI2V 5B architecture with audio components from S2V 14B
s2v_5B.transformer = EasyDict(
    __name__="Config: Transformer config for WanModel_S2V 5B")
s2v_5B.transformer.patch_size = (1, 2, 2)
s2v_5B.transformer.dim = 3072  # From TI2V 5B
s2v_5B.transformer.ffn_dim = 14336  # From TI2V 5B
s2v_5B.transformer.freq_dim = 256
s2v_5B.transformer.num_heads = 24  # From TI2V 5B
s2v_5B.transformer.num_layers = 30  # From TI2V 5B
s2v_5B.transformer.window_size = (-1, -1)
s2v_5B.transformer.qk_norm = True
s2v_5B.transformer.cross_attn_norm = True
s2v_5B.transformer.eps = 1e-6

# Audio injection settings - Adapted from S2V 14B for smaller model
s2v_5B.transformer.enable_adain = True
s2v_5B.transformer.adain_mode = "attn_norm"
# Reduced number of injection layers for 5B model (30 layers vs 40 in 14B)
s2v_5B.transformer.audio_inject_layers = [
    0, 3, 6, 9, 12, 15, 18, 21, 24, 27, 29
]
s2v_5B.transformer.zero_init = False
s2v_5B.transformer.zero_timestep = True
s2v_5B.transformer.enable_motioner = False
s2v_5B.transformer.add_last_motion = True
s2v_5B.transformer.trainable_token = False
s2v_5B.transformer.enable_tsm = False
s2v_5B.transformer.enable_framepack = True
s2v_5B.transformer.framepack_drop_mode = 'padd'
s2v_5B.transformer.audio_dim = 1024  # Match S2V 14B audio dimension

# Motion settings - Adapted for 5B model
s2v_5B.transformer.motion_frames = 73
s2v_5B.transformer.cond_dim = 48

# v_scale
s2v_5B.transformer.v_scale = 1.0

# inference
s2v_5B.sample_neg_prompt = "画面模糊，最差质量，画面模糊，细节模糊不清，情绪激动剧烈，手快速抖动，字幕，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走"
s2v_5B.drop_first_motion = True
s2v_5B.sample_fps = 25
s2v_5B.sample_shift = 5
s2v_5B.sample_steps = 50
s2v_5B.sample_guide_scale = 4.5
s2v_5B.frame_num = 120

