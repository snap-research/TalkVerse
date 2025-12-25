#!/bin/bash

export CUDA_HOME=software/cuda-12.4
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
source software/anaconda3/etc/profile.d/conda.sh
conda activate talking

CUDA_VISIBLE_DEVICES=7 \
python generate.py \
  --task s2v-5B \
  --ckpt_dir ckpts/Wan2.2-TI2V-5B \
  --batch_file ./outputs/dubbing_input/batch_gpu_0.json \
  --output_dir ./outputs/dubbing \
  --size 1280*704 \
  --infer_frames 120 \
  --num_clip 2 \
  --sample_steps 50 \
  --sample_guide_scale 6.5 \
  --base_seed -1 \
  --v_scale 1.0
