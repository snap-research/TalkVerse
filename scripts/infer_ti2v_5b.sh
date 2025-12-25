export CUDA_HOME=software/cuda-12.4
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
export CUDA_VISIBLE_DEVICES='1'

python generate.py \
  --task ti2v-5B \
  --size 1280*704 \
  --ckpt_dir ckpts/Wan2.2-TI2V-5B \
  --offload_model True \
  --convert_model_dtype \
  --t5_cpu \
  --image ./examples/sota_cases/music_6.png \
  --prompt "A young woman is singing a song."
