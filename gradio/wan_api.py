#!/usr/bin/env python3
"""
WAN S2V-5B Inference API
- Loads the WAN S2V-5B model once on startup
- POST /infer accepts image+audio + generation params
- Returns either:
  (A) JSON: {"video_url": "http://host:port/static/outputs/....mp4"}
  (B) direct video stream (when form field stream=1)
"""

import os, io, time, argparse, random
from pathlib import Path
from typing import Optional

import torch
from fastapi import FastAPI, File, Form, UploadFile, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
import uvicorn

# ---- WAN imports (adjust import path if needed) ----
from wan.configs import WAN_CONFIGS
from wan.utils.utils import save_video, merge_video_audio
from wan import WanS2V_5B as BaseWanS2V_5B
from wan.s2v_5b import merge_lora_into_base


def parse_args():
    ap = argparse.ArgumentParser("WAN S2V-5B Inference API")
    ap.add_argument("--ckpt_dir", type=str, default="ckpts/Wan2.2-TI2V-5B",
                    help="TI2V-5B base checkpoint dir or .pth")
    ap.add_argument("--s2v_ckpt_path", type=str,
                    default="./s2v_5b_output/ckpt_backup/step-173750.safetensors",
                    help="Path to S2V-5B audio components .safetensors")
    ap.add_argument("--device_id", type=int, default=2)
    ap.add_argument("--v_scale", type=float, default=1.0)
    ap.add_argument("--port", type=int, default=8001)
    ap.add_argument("--host", type=str, default="0.0.0.0")
    ap.add_argument("--static_root", type=str, default="./gradio_outputs",
                    help="Directory to serve /static from (outputs are saved here)")
    ap.add_argument("--allow_origins", type=str, default="*")
    return ap.parse_args()


# ---- Helpers ----
def _bool(x: Optional[str], default=False):
    if x is None: return default
    if isinstance(x, bool): return x
    return str(x).strip().lower() in {"1", "true", "t", "yes", "y", "on"}

def _int(x, default):
    try: return int(x)
    except: return default

def _float(x, default):
    try: return float(x)
    except: return default


# ---- Optional subclass to merge audio and LoRA into base ----
class WanS2V_5B(BaseWanS2V_5B):
    def __init__(self, *args, s2v_ckpt_path: Optional[str] = None, **kwargs):
        self.s2v_ckpt_path = s2v_ckpt_path
        super().__init__(*args, **kwargs)

    def _load_pretrained_weights(self, checkpoint_dir):
        from safetensors.torch import load_file

        print(f"[WAN API] Loading TI2V-5B base from {checkpoint_dir}")
        # Base weights
        if checkpoint_dir.endswith(".pth"):
            ti2v_state = torch.load(checkpoint_dir, map_location="cpu")
        else:
            sts = [f for f in os.listdir(checkpoint_dir) if f.endswith(".safetensors")]
            if not sts:
                raise ValueError(f"No .safetensors under {checkpoint_dir}")
            ti2v_state = load_file(os.path.join(checkpoint_dir, sts[0]))
            for fn in sts[1:]:
                ti2v_state.update(load_file(os.path.join(checkpoint_dir, fn)))

        # Audio components
        audio_state = {}
        if self.s2v_ckpt_path and os.path.exists(self.s2v_ckpt_path):
            print(f"[WAN API] Loading audio components from: {self.s2v_ckpt_path}")
            audio_state = load_file(self.s2v_ckpt_path)
        else:
            print("[WAN API] No valid --s2v_ckpt_path provided; skipping audio components.")

        current = self.noise_model.state_dict()
        merged, n_base, n_audio = {}, 0, 0
        for k, v in current.items():
            if k in audio_state:
                merged[k] = audio_state[k]; n_audio += 1
            elif k in ti2v_state:
                merged[k] = ti2v_state[k]; n_base += 1
            else:
                merged[k] = v.clone()
        missing, unexpected = self.noise_model.load_state_dict(merged, strict=False)
        print(f"[WAN API] Merged base={n_base}, audio={n_audio}, "
              f"params={len(current)}, missing={len(missing)}, unexpected={len(unexpected)}")

        # Merge LoRA if present
        def _has_lora_keys(sd: dict):
            for k in sd.keys():
                if k.endswith("lora_A.weight") or k.endswith("lora_B.weight"):
                    return True
            return False
        if _has_lora_keys(audio_state):
            print(f"[WAN API] Found LoRA tensors in S2V ckpt. Merging (r={self.lora_rank}, alpha={self.lora_alpha})")
            merge_lora_into_base(self.noise_model, audio_state,
                                 r=self.lora_rank, alpha=self.lora_alpha)
        self._freeze_all_weights()


def build_app(args):
    app = FastAPI(title="WAN S2V-5B API", version="1.0")

    # CORS (optional)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=[o.strip() for o in args.allow_origins.split(",") if o.strip()],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Prepare static dir and mount for serving outputs
    static_root = Path(args.static_root).resolve()
    out_dir = static_root / "outputs"
    out_dir.mkdir(parents=True, exist_ok=True)
    app.mount("/static", StaticFiles(directory=str(static_root)), name="static")

    # ---- Load WAN once ----
    print("[WAN API] Initializing model...")
    task = "s2v-5B"
    cfg = WAN_CONFIGS[task]
    cfg.transformer.v_scale = float(args.v_scale)

    WAN = WanS2V_5B(
        config=cfg,
        checkpoint_dir=args.ckpt_dir,
        device_id=args.device_id,
        rank=0,
        t5_fsdp=False,
        dit_fsdp=False,
        use_sp=False,
        t5_cpu=False,
        init_on_cpu=True,
        convert_model_dtype=False,
        size_bucket="fasttalk-720",           # default; can be overridden per request
        s2v_ckpt_path=args.s2v_ckpt_path,
    )
    print("[WAN API] Model ready.")

    @app.get("/health")
    def health():
        return {"ok": True, "model": "Wan S2V-5B", "device_id": args.device_id}

    @app.post("/infer")
    async def infer(
        request: Request,
        image: UploadFile = File(...),
        audio: UploadFile = File(...),
        prompt: str = Form("A person speaking to the camera."),

        # Generation params (optional form fields)
        size_bucket: str = Form("fasttalk-720"),
        max_area: str = Form("902,400"),  # 1280*704
        infer_frames: str = Form("120"),
        num_clip: str = Form("2"),
        sample_steps: str = Form("50"),
        sample_guide_scale: str = Form("6.5"),
        base_seed: str = Form("-1"),
        offload_model: str = Form("true"),
        init_first_frame: str = Form("false"),
        sample_solver: str = Form("unipc"),

        # Response behavior
        stream: str = Form("0"),   # "1" to stream video bytes directly
    ):
        try:
            # Write uploads to temp files
            tmp_dir = Path("./_tmp_uploads")
            tmp_dir.mkdir(parents=True, exist_ok=True)

            img_suffix = Path(image.filename or "image.png").suffix or ".png"
            aud_suffix = Path(audio.filename or "audio.wav").suffix or ".wav"
            img_path = tmp_dir / f"img_{int(time.time()*1000)}{img_suffix}"
            aud_path = tmp_dir / f"aud_{int(time.time()*1000)}{aud_suffix}"

            with open(img_path, "wb") as f:
                f.write(await image.read())
            with open(aud_path, "wb") as f:
                f.write(await audio.read())

            # Parse params
            WAN.size_bucket = size_bucket or "fasttalk-720"
            max_area_i = _int(str(max_area).replace(",", ""), 1280 * 704)
            infer_frames_i = _int(infer_frames, 120)
            num_clip_i = _int(num_clip, 2)
            sample_steps_i = _int(sample_steps, 50)
            sample_guide_scale_f = _float(sample_guide_scale, 6.5)
            seed_i = _int(base_seed, -1)
            if seed_i < 0:
                seed_i = random.randint(0, 2**31 - 1)
            offload_b = _bool(offload_model, True)
            init_first_b = _bool(init_first_frame, False)
            solver_s = sample_solver if sample_solver in {"unipc", "dpm++"} else "unipc"

            # Output file path
            ts = time.strftime("%Y%m%d_%H%M%S")
            out_path = out_dir / f"{img_path.stem}_{WAN.size_bucket}_{infer_frames_i}f_{ts}.mp4"

            with torch.inference_mode():
                vid = WAN.generate(
                    input_prompt=prompt,
                    ref_image_path=str(img_path),
                    audio_path=str(aud_path),
                    enable_tts=False,
                    tts_prompt_audio=None, tts_prompt_text=None, tts_text=None,
                    num_repeat=num_clip_i,
                    max_area=max_area_i,
                    infer_frames=infer_frames_i,
                    shift=5.0,
                    sample_solver=solver_s,
                    sampling_steps=sample_steps_i,
                    guide_scale=sample_guide_scale_f,
                    n_prompt="",
                    seed=seed_i,
                    offload_model=offload_b,
                    init_first_frame=init_first_b,
                )

            # Save video + mux audio
            save_video(
                tensor=vid[None],
                save_file=str(out_path),
                fps=WAN.config.sample_fps,  # or cfg.sample_fps
                nrow=1,
                normalize=True,
                value_range=(-1, 1),
            )
            merge_video_audio(video_path=str(out_path), audio_path=str(aud_path))

            # Stream?
            if _bool(stream, False):
                def _iterfile():
                    with open(out_path, "rb") as f:
                        while True:
                            chunk = f.read(1024 * 512)
                            if not chunk:
                                break
                            yield chunk
                return StreamingResponse(_iterfile(), media_type="video/mp4")

            # JSON with fully-qualified URL
            base_url = str(request.base_url).rstrip("/")
            rel = f"/static/outputs/{out_path.name}"
            return JSONResponse({"video_url": f"{base_url}{rel}",
                                 "seed": seed_i,
                                 "size_bucket": WAN.size_bucket})

        except Exception as e:
            return JSONResponse({"error": repr(e)}, status_code=500)

    return app


if __name__ == "__main__":
    args = parse_args()
    app = build_app(args)
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")
