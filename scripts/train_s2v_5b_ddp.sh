#!/bin/bash
export CUDA_HOME=software/cuda-12.4
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH

export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128,roundup_power2_divisions:16

DEBUG=false

export CUDA_VISIBLE_DEVICES='0,1,2,3,4,5,6,7'
NUM_GPUS=$(echo $CUDA_VISIBLE_DEVICES | awk -F',' '{print NF}')

MODEL_DIR="ckpts/Wan2.2-TI2V-5B"
WAV2VEC_DIR="ckpts/Wan2.2-S2V-14B/wav2vec2-large-xlsr-53-english"
BASE_DATA_PATH=""
OUTPUT_DIR="./s2v_5b_output/training"

DEFAULT_METADATA_PATH="openhumanvid_meta_info.csv panda70m_meta_info.csv"
FRAMEPACK_METADATA_PATH="openhumanvid_meta_info.csv panda70m_meta_info.csv"

SIZE_BUCKET="fasttalk-720"
NUM_FRAMES=72
LEARNING_RATE=1e-5
NUM_EPOCHS=100
BATCH_SIZE=1
GRADIENT_ACCUMULATION_STEPS=8
SAVE_STEPS=50
ENABLE_FRAMEPACK=true
UNFREEZE_STRATEGY="lora"
ENABLE_ROI_LOSS="true"
always_use_motion_frames="false"

while [[ $# -gt 0 ]]; do
    case $1 in
        --model_dir)
            MODEL_DIR="$2"
            shift 2
            ;;
        --output_dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --height)
            HEIGHT="$2"
            shift 2
            ;;
        --width)
            WIDTH="$2"
            shift 2
            ;;
        --num_frames)
            NUM_FRAMES="$2"
            shift 2
            ;;
        --learning_rate)
            LEARNING_RATE="$2"
            shift 2
            ;;
        --num_epochs)
            NUM_EPOCHS="$2"
            shift 2
            ;;
        --batch_size)
            BATCH_SIZE="$2"
            shift 2
            ;;
        --gradient_accumulation_steps)
            GRADIENT_ACCUMULATION_STEPS="$2"
            shift 2
            ;;
        --save_steps)
            SAVE_STEPS="$2"
            shift 2
            ;;
        --finetune_all)
            FINETUNE_ALL="--finetune_all"
            shift
            ;;
        --debug)
            DEBUG=true
            shift
            ;;
        --enable_framepack)
            ENABLE_FRAMEPACK="$2"
            shift 2
            ;;
        --unfreeze_strategy)
            UNFREEZE_STRATEGY="$2"
            shift 2
            ;;
        --enable_roi_loss)
            ENABLE_ROI_LOSS="$2"
            shift 2
            ;;
        -h|--help)
            echo "Usage: $0 [OPTIONS]"
            echo "Options:"
            echo "  --model_dir PATH           Path to TI2V 5B model checkpoint directory"
            echo "  --output_dir PATH          Output directory for training checkpoints"
            echo "  --height INT               Video height (default: 256)"
            echo "  --width INT                Video width (default: 256)"
            echo "  --num_frames INT           Number of frames (default: 80)"
            echo "  --learning_rate FLOAT      Learning rate (default: 1e-4)"
            echo "  --num_epochs INT           Number of epochs (default: 10)"
            echo "  --batch_size INT           Batch size per device (default: 1)"
            echo "  --gradient_accumulation_steps INT  Gradient accumulation steps (default: 8)"
            echo "  --save_steps INT           Save checkpoint every N steps (default: 500)"
            echo "  --finetune_all             Finetune all parameters instead of just audio components"
            echo "  --debug                    Enable debug mode (use 2 GPUs, debug dataset, debug output dir)"
            echo "  --enable_framepack BOOL    Enable framepack feature (default: true)"
            echo "  --unfreeze_strategy STR    Unfreeze strategy (default: moderate)"
            echo "  --enable_roi_loss BOOL     Enable ROI loss feature (default: false)"
            echo "  -h, --help                 Show this help message"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use -h or --help for usage information"
            exit 1
            ;;
    esac
done

if [ "$DEBUG" = true ]; then
    echo "Debug mode enabled - using 2 GPUs and debug dataset"
    export CUDA_VISIBLE_DEVICES='0'
    NUM_GPUS=1
    SAVE_STEPS=10
    OUTPUT_DIR="./s2v_5b_output/training_debug"
    DATASET_METADATA_PATH=("./examples/dataset_debug.csv")
    BATCH_SIZE=1
    GRADIENT_ACCUMULATION_STEPS=4
else
    if [ "$ENABLE_FRAMEPACK" = "true" ]; then
        DATASET_METADATA_PATH="$FRAMEPACK_METADATA_PATH"
        echo "Framepack enabled: Using metadata path: $DATASET_METADATA_PATH"
    else
        DATASET_METADATA_PATH="$DEFAULT_METADATA_PATH"
        echo "Framepack disabled: Using default metadata path: $DATASET_METADATA_PATH"
    fi
fi

if [ ! -d "$MODEL_DIR" ]; then
    echo "Error: Model directory does not exist: $MODEL_DIR"
    exit 1
fi

mkdir -p "$OUTPUT_DIR"

echo "Starting S2V 5B training..."
if [ "$DEBUG" = true ]; then
    echo "*** DEBUG MODE ENABLED ***"
fi
echo "Model Directory: $MODEL_DIR"
echo "Wav2vec Model: $WAV2VEC_DIR"
echo "Output Directory: $OUTPUT_DIR"
echo "Dataset Path: $DATASET_METADATA_PATH"
echo "Video Resolution: $SIZE_BUCKET"
echo "Number of Frames: $NUM_FRAMES"
echo "Learning Rate: $LEARNING_RATE"
echo "Number of Epochs: $NUM_EPOCHS"
echo "Batch Size: $BATCH_SIZE"
echo "Gradient Accumulation Steps: $GRADIENT_ACCUMULATION_STEPS"
echo "Save Steps: $SAVE_STEPS"
echo "Number of GPUs: $NUM_GPUS"
echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
echo "Enable Framepack: $ENABLE_FRAMEPACK"
echo "Enable ROI Loss: $ENABLE_ROI_LOSS"
echo "Unfreeze Strategy: $UNFREEZE_STRATEGY"
echo "Always Use Motion Frames: $always_use_motion_frames"

python -m torch.distributed.run train_s2v_5b.py \
  --ckpt_dir "${MODEL_DIR}" \
  --wav2vec_dir "${WAV2VEC_DIR}" \
  --dataset_metadata_path ${DATASET_METADATA_PATH} \
  --dataset_base_path "${BASE_DATA_PATH}" \
  --output_path "${OUTPUT_DIR}" \
  --distributed_policy "ddp" \
  --size_bucket ${SIZE_BUCKET} \
  --num_frames ${NUM_FRAMES} \
  --frame_interval 1 \
  --learning_rate ${LEARNING_RATE} \
  --num_epochs ${NUM_EPOCHS} \
  --batch_size ${BATCH_SIZE} \
  --gradient_accumulation_steps ${GRADIENT_ACCUMULATION_STEPS} \
  --save_steps ${SAVE_STEPS} \
  --num_workers 2 \
  --text_dropout_prob 0.1 \
  --img_dropout_prob 0.1 \
  --audio_dropout_prob 0.1 \
  --audio_zero_init_lr_scaling 1 \
  --use_gradient_checkpointing \
  --use_tensorboard \
  --log_interval 100 \
  --unfreeze_strategy ${UNFREEZE_STRATEGY} \
  --enable_framepack ${ENABLE_FRAMEPACK} \
  --enable_roi_loss ${ENABLE_ROI_LOSS} \
  --always_use_motion_frames ${always_use_motion_frames}

echo "Training completed!"