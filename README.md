# TalkVerse: Democratizing Minute-Long Audio-Driven Video Generation

<p align="center">
    <a href="https://zhenzhiwang.github.io/talkverse/"><b>TalkVerse Website</b></a> &nbsp&nbsp ｜ &nbsp&nbsp 🖥️ <a href="https://github.com/Snap-Research/TalkVerse">GitHub</a> &nbsp&nbsp  | &nbsp&nbsp🤗 <a href="https://huggingface.co/datasets/zhenzhiwang/TalkVerse">Hugging Face</a>&nbsp&nbsp | &nbsp&nbsp📑 <a href="https://arxiv.org/abs/2512.14938">arXiv</a>&nbsp&nbsp
    <br>
</p>

-----

We introduce **TalkVerse**, a large-scale, open corpus for single-person, audio-driven talking video generation designed to enable fair, reproducible comparison across methods, with **2.1M video clips(6.3K hours)** from public T2V source (OpenHumanVid, Panda70M). We also release all the training and inference code.

## 🔥 Latest News
* **[2025-12-31]** 🚀 We release the Training and Inference code for **TalkVerse-5B** model, a lightweight baseline capable of minute-long generation.

## 🛠️ Installation

Install dependencies:
```bash
# Basic requirements for Wan2.2 backbone
pip install -r requirements.txt

# Additional requirements for Speech-to-Video (S2V) and Audio processing
pip install -r requirements_s2v.txt
```

## 📥 Model Weights

| Model | Description | Download |
|-------|-------------|----------|
| **TalkVerse-5B** | Audio-Driven LoRA weights trained on TalkVerse | [HuggingFace](https://huggingface.co/snap-research/talkverse-s2v-5b) |
| **Wan2.2-TI2V-5B** | Base Text/Image-to-Video Model (Backbone) | [HuggingFace](https://huggingface.co/Wan-AI/Wan2.2-TI2V-5B) |
| **Wav2Vec2** | Audio Encoder (wav2vec2-large-xlsr-53-english) | [HuggingFace](https://huggingface.co/facebook/wav2vec2-large-xlsr-53) |

Download the models and place them in a `ckpts/` directory (or adjust paths in scripts accordingly).

## 🚀 Inference

We support both single-video generation and batch generation for the S2V-5B model.

### 1. Single Video Generation

Use `generate.py` to generate a single video from an image and audio file.

```bash
python generate.py \
    --task s2v-5B \
    --ckpt_dir ./ckpts/Wan2.2-TI2V-5B \
    --lora_ckpt ./ckpts/talkverse_5b_lora.safetensors \
    --image examples/input_face.jpg \
    --audio examples/input_audio.wav \
    --prompt "A person talking naturally." \
    --offload_model True \
    --t5_cpu
```

**Arguments:**
* `--task s2v-5B`: Selects the TalkVerse 5B model.
* `--ckpt_dir`: Path to the base Wan2.2-TI2V-5B checkpoint.
* `--lora_ckpt`: (Optional) Path to the trained S2V LoRA checkpoint if not merged.
* `--offload_model True` & `--t5_cpu`: Saves VRAM for consumer GPUs (e.g., RTX 4090).

### 2. Batch Generation

For generating videos for a large dataset (e.g., testing set), use the provided shell script `run_batch_generation.sh`. This script automatically shards the input JSON file across available GPUs.

1. **Prepare a batch config JSON file** (e.g., `batch_config.json`):
   ```json
   [
     {
       "image": "path/to/img1.jpg",
       "audio": "path/to/audio1.wav",
       "prompt": "A man talking..."
     },
     {
       "image": "path/to/img2.jpg",
       "audio": "path/to/audio2.wav",
       "prompt": "A woman singing..."
     }
   ]
   ```

2. **Run the script**:
   ```bash
   # Edit the script to set CKPT_DIR and BATCH_FILE paths
   bash run_batch_generation.sh
   ```

   The script allows configuration of:
   - `INFER_FRAMES=120` (Length of video)
   - `GUIDE_SCALE=6.5` (Classifier-free guidance scale)

## 🏋️ Training

We provide the training script `train_s2v_5b.py` to train the 5B model on the TalkVerse dataset (or your own data).

### Data Preparation

The training script expects a `VideoDataset` that provides video, audio, and text pairs. Please refer to the `Wan.src.dataset` module or `create_batch_config.py` for data format details.

### Running Training

You can run training using `torchrun` for distributed training (DDP).

```bash
# Example: Train on 8 GPUs with DDP
torchrun --nproc_per_node=8 train_s2v_5b.py \
    --distributed_policy ddp \
    --ckpt_dir ./ckpts/Wan2.2-TI2V-5B \
    --output_path ./output_s2v_5b \
    --batch_size 1 \
    --gradient_accumulation_steps 4 \
    --learning_rate 1e-5 \
    --unfreeze_strategy lora \
    --enable_roi_loss true \
    --enable_framepack true
```

**Key Arguments:**
* `--unfreeze_strategy`: Controls which parts of the model to train.
  * `minimal`: Unfreezes only time embeddings and new headers.
  * `framepack`: Unfreezes frame packing modules.
  * `lora`: Uses LoRA for efficient fine-tuning.
  * `full`: Full parameter fine-tuning.
* `--enable_roi_loss`: Applies higher loss weight to face/body regions (requires pre-computed masks).
* `--enable_framepack`: Enables the context frame packing for long-video generation consistency.

## 🎥 Gradio Demo

We provide a Gradio interface for easy interaction with the model. The demo requires two backend APIs to be running: the **Qwen API** (for prompt enhancement) and the **Wan API** (for video generation).

1.  **Start the Qwen API** (Terminal 1):
    ```bash
    # Runs on GPU 0
    python gradio/qwen_api.py
    ```

2.  **Start the Wan API** (Terminal 2):
    ```bash
    # Runs on GPU 1
    export CUDA_VISIBLE_DEVICES='1'
    python gradio/wan_api_trim.py
    ```

3.  **Launch the Gradio App** (Terminal 3):
    ```bash
    # Connects to APIs
    python gradio/gradio_demo.py
    ```

    The app allows you to:
    - Upload an image and audio file.
    - Enhance the input prompt using Qwen (optional).
    - Generate a talking face video.
    - View the result directly in the browser.

## ⚖️ License

The code, dataset and model weights are released under the **Snap Inc. Non-Commercial License**.

## 📚 Citation

If you find TalkVerse useful for your research, please cite:

```bibtex
@article{wang2025talkverse,
  title={TalkVerse: Democratizing Minute-Long Audio-Driven Video Generation},
  author={Wang, Zhenzhi and Wang, Jian and Ma, Ke and Lin, Dahua and Zhou, Bing},
  journal={arXiv preprint arXiv:2512.14938},
  year={2025}
}
```
