# qwen_api.py
# FastAPI server that exposes Qwen3-Omni "prompt enhancing" as an HTTP API.
# Accepts multipart form-data: image, audio, user_prompt
# Returns JSON: { "visual_caption": str, "audio_caption": str, "combined": str }

import os
import argparse
import warnings
import hashlib
import subprocess
import tempfile
from pathlib import Path
from typing import Optional, List, Tuple

warnings.filterwarnings("ignore")

from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import uvicorn

# ===================== Utilities =====================
def safe_mkdir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

def sh(cmd: list):
    # Raise CalledProcessError on failure so we can 500 cleanly
    subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

def extract_audio_ffmpeg_from_wav_or_video(audio_or_video_path: str, out_dir: Path) -> str:
    safe_mkdir(out_dir)
    key = hashlib.md5(audio_or_video_path.encode("utf-8")).hexdigest()
    out_wav = out_dir / f"{key}.wav"
    if out_wav.exists():
        try: out_wav.unlink()
        except FileNotFoundError: pass
    sh(["ffmpeg", "-y", "-i", audio_or_video_path, "-vn", "-ac", "1", "-ar", "16000", "-f", "wav", str(out_wav)])
    return str(out_wav)

def build_message_for_audio(audio_path: str) -> list:
    return [
        {
            "role": "user",
            "content": [
                {"type": "audio", "audio": audio_path},
                {"type": "text", "text": (
                    "Write one natural paragraph (40-80 words) that brief describe the audio. "
                    "State whether it's singing or talking and how many voices, then summarize the dominant emotion and its intensity. "
                    "Describe pitch and loudness only as they affect facial/mouth movement and energy (e.g., high/low, rising question, whisper/shout, crescendo). "
                    "Note timbre in visual terms (warm/raspy/bright), pace and articulation, and infer likely age/gender presentation. "
                    "If obvious, mention standout techniques (vibrato, falsetto, growl). Avoid lists and technical jargon."
                )},
            ],
        }
    ]

def build_message_for_image(image_path: str, user_prompt: str) -> list:
    return [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image_path},
                {"type": "text", "text": (
                    "You are an professional expert of video descriptioning. "
                    "Use the following user-intended video description as additional context for what will happen in this video: "
                    f"<<USER_PROMPT>> {user_prompt} <<USER_PROMPT_END>> "
                    "Describe the image while considering the user-intended video description above in one natural paragraph (60-90 words) in a documentary tone. "
                    "Cover camera framing (e.g., medium close-up, static camera), subject appearance (skin tone, hair, clothing), "
                    "notable objects (e.g., microphone with label), and the environment (studio/room, walls, panels, lighting). "
                    "State what the person appears to be doing and where they are looking. "
                    "Avoid lists and technical jargon. Do not mention unknown brands. "
                )},
            ],
        }
    ]

# ===================== Qwen Wrapper =====================
class QwenEnhancer:
    def __init__(self, model_path: str, backend: str, flash_attn2: bool, gpu_mem_util: float, max_model_len: int):
        # Keep vLLM state inside this process only
        os.environ["CUDA_VISIBLE_DEVICES"] = os.environ.get("CUDA_VISIBLE_DEVICES", "0")
        os.environ["VLLM_USE_V1"] = "0"
        os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"
        os.environ["VLLM_LOGGING_LEVEL"] = "ERROR"

        self.backend = backend.lower()
        from transformers import Qwen3OmniMoeProcessor
        self.processor = Qwen3OmniMoeProcessor.from_pretrained(model_path)
        if self.backend == "transformers":
            from transformers import Qwen3OmniMoeForConditionalGeneration
            if flash_attn2:
                self.model = Qwen3OmniMoeForConditionalGeneration.from_pretrained(
                    model_path, dtype="auto", attn_implementation="flash_attention_2", device_map="auto"
                )
            else:
                self.model = Qwen3OmniMoeForConditionalGeneration.from_pretrained(
                    model_path, dtype="auto", device_map="auto"
                )
            self._llm = None
        else:
            from vllm import LLM
            self._llm = LLM(
                model=model_path,
                trust_remote_code=True,
                tensor_parallel_size=1,
                gpu_memory_utilization=gpu_mem_util,
                max_num_seqs=1,
                max_model_len=max_model_len,
                limit_mm_per_prompt={"image": 3, "video": 0, "audio": 3},
                seed=1234,
            )
            self.model = None

        try:
            from qwen_omni_utils import process_mm_info  # noqa: F401
            self.process_mm_info = __import__("qwen_omni_utils").process_mm_info
        except Exception as e:
            raise ImportError("Missing qwen_omni_utils.process_mm_info in PYTHONPATH.") from e

    def _gen(self, messages: list) -> str:
        if self.backend == "transformers":
            text = self.processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
            audios, images, videos = self.process_mm_info(messages, use_audio_in_video=True)
            inputs = self.processor(
                text=text, audio=audios, images=images, videos=videos,
                return_tensors="pt", padding=True, use_audio_in_video=True
            )
            inputs = inputs.to(self.model.device).to(self.model.dtype)
            text_ids, _ = self.model.generate(
                **inputs,
                thinker_return_dict_in_generate=True,
                thinker_max_new_tokens=2048,
                thinker_do_sample=False,
                speaker="Ethan",
                use_audio_in_video=True,
                return_audio=False,
            )
            out = self.processor.batch_decode(
                text_ids.sequences[:, inputs["input_ids"].shape[1]:],
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )[0]
            return out.strip()
        else:
            from vllm import SamplingParams
            sampling_params = SamplingParams(temperature=1e-2, top_p=0.1, top_k=1, max_tokens=2048)
            text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            audios, images, videos = self.process_mm_info(messages, use_audio_in_video=True)
            inputs = {
                "prompt": text,
                "multi_modal_data": {},
                "mm_processor_kwargs": {"use_audio_in_video": True},
            }
            if audios is not None: inputs["multi_modal_data"]["audio"] = audios
            if images is not None: inputs["multi_modal_data"]["image"] = images
            if videos is not None: inputs["multi_modal_data"]["video"] = videos
            outputs = self._llm.generate(inputs, sampling_params=sampling_params)
            return outputs[0].outputs[0].text.strip()

    def enhance(self, image_path: str, audio_path: str, user_prompt: str, cache_dir: Path) -> Tuple[str, str, str]:
        if not Path(image_path).exists():
            raise FileNotFoundError(f"Image missing: {image_path}")
        if not Path(audio_path).exists():
            raise FileNotFoundError(f"Audio missing: {audio_path}")

        wav = extract_audio_ffmpeg_from_wav_or_video(audio_path, cache_dir)
        try:
            visual_caption = self._gen(build_message_for_image(image_path, user_prompt or ""))
            audio_caption  = self._gen(build_message_for_audio(wav))
        finally:
            try: Path(wav).unlink(missing_ok=True)
            except Exception: pass

        combined = f"{visual_caption.strip()} {audio_caption.strip()}"
        """if user_prompt and user_prompt.strip():
            combined = f"{combined} {user_prompt.strip()}" """
        return visual_caption.strip(), audio_caption.strip(), combined.strip()

# ===================== CLI =====================
def parse_args():
    ap = argparse.ArgumentParser(description="Qwen3-Omni API server (prompt enhancing).")
    ap.add_argument("--qwen_model_path", type=str, default="ckpts/Qwen3-Omni-30B-A3B-Instruct",
                    help="Path to Qwen3-Omni (e.g., Qwen3-Omni-30B-A3B-Instruct)")
    ap.add_argument("--qwen_backend", choices=["vllm", "transformers"], default="vllm")
    ap.add_argument("--flash_attn2", action="store_true", help="(transformers) flash_attention_2")
    ap.add_argument("--gpu_mem_util", type=float, default=0.95, help="(vLLM) GPU memory utilization")
    ap.add_argument("--max_model_len", type=int, default=32768, help="(vLLM) max model len")
    ap.add_argument("--host", type=str, default="0.0.0.0")
    ap.add_argument("--port", type=int, default=8000)
    return ap.parse_args()

# ===================== App =====================
def make_app(qwen: QwenEnhancer) -> FastAPI:
    app = FastAPI(title="Qwen Prompt Enhancer API", version="1.0")

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"], allow_credentials=True,
        allow_methods=["*"], allow_headers=["*"],
    )

    @app.get("/health")
    def health():
        return {"ok": True}

    @app.post("/enhance")
    async def enhance(
        image: UploadFile = File(...),
        audio: UploadFile = File(...),
        user_prompt: str = Form(default="")
    ):
        try:
            with tempfile.TemporaryDirectory(prefix="qwen_api_") as td:
                tdir = Path(td)
                img_path = tdir / f"img{Path(image.filename).suffix or '.png'}"
                aud_path = tdir / f"aud{Path(audio.filename).suffix or '.wav'}"

                with img_path.open("wb") as f:
                    f.write(await image.read())
                with aud_path.open("wb") as f:
                    f.write(await audio.read())

                vcap, acap, combo = qwen.enhance(
                    image_path=str(img_path),
                    audio_path=str(aud_path),
                    user_prompt=user_prompt or "",
                    cache_dir=tdir / "_audio_cache"
                )

            return JSONResponse({
                "visual_caption": vcap,
                "audio_caption": acap,
                "combined": combo,
            })
        except Exception as e:
            return JSONResponse({"error": repr(e)}, status_code=500)

    return app

def main():
    args = parse_args()
    qwen = QwenEnhancer(
        model_path=args.qwen_model_path,
        backend=args.qwen_backend,
        flash_attn2=args.flash_attn2,
        gpu_mem_util=args.gpu_mem_util,
        max_model_len=args.max_model_len,
    )
    app = make_app(qwen)
    uvicorn.run(app, host=args.host, port=args.port)

if __name__ == "__main__":
    main()
