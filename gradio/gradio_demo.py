# app_gradio_s2v5b_enhance_client.py
# Gradio app that calls external APIs for BOTH: Qwen enhancement & Wan S2V-5B inference.

import os
import time
import argparse
import warnings
import base64
from pathlib import Path
from typing import Optional, List

warnings.filterwarnings("ignore")

import gradio as gr
import requests

import safehttpx
safehttpx.ALLOW_PRIVATE = True  # (if the version exposes this flag)


# ===================== PRESETS =====================
PRESETS: List[dict] = [
  {"name":"2025-07-14-7825","image":"./examples/sota_cases/2025-07-14-7825.png","audio":"./examples/sota_cases/2025-07-14-7825.wav"},
  {"name":"2025-07-14-2371","image":"./examples/sota_cases/2025-07-14-2371.png","audio":"./examples/sota_cases/2025-07-14-2371.wav"},
  {"name":"2025-07-14-6032","image":"./examples/sota_cases/2025-07-14-6032.png","audio":"./examples/sota_cases/2025-07-14-6032.wav"},
  {"name":"case-43","image":"./examples/sota_cases/case-43.png","audio":"./examples/sota_cases/case-43.wav"},
  {"name":"snap-32","image":"./examples/human_data/female_720p.jpg","audio":"./examples/music_data/4batz-act ii_ date @ 8-6807820226.mp3"},
]

def _uniq(seq):
    seen = set()
    out = []
    for x in seq:
        if x and x not in seen:
            seen.add(x)
            out.append(x)
    return out

IMAGE_PRESETS: List[str] = _uniq([p.get("image") for p in PRESETS if p.get("image")])
AUDIO_PRESETS: List[str] = _uniq([p.get("audio") for p in PRESETS if p.get("audio")])

# ===================== CONSTANTS =====================
FIXED_BUCKET = "fasttalk-720"
FIXED_MAX_AREA = 1280 * 704
CLIENT_TMP_DIR = Path("./_client_tmp_videos")  # temp save when server streams/b64
DEFAULT_PROMPT = "A person is speaking to the camera"

# ===================== CLI =====================
def parse_args():
    ap = argparse.ArgumentParser(description="Gradio: Wan S2V-5B app (Qwen enhance + Wan infer via HTTP APIs).")
    ap.add_argument("--qwen_api_url", type=str, default="http://localhost:8000/enhance",
                    help="HTTP endpoint for prompt enhancing")
    ap.add_argument("--qwen_timeout", type=int, default=180, help="Seconds to wait for Qwen API")

    ap.add_argument("--wan_api_url", type=str, default="http://localhost:8002/infer",
                    help="HTTP endpoint for Wan S2V-5B inference")
    ap.add_argument("--wan_timeout", type=int, default=3600, help="Seconds to wait for WAN API")

    ap.add_argument("--server_port", type=int, default=8888)
    return ap.parse_args()

# ===================== Globals =====================
ARGS = None

# ===================== API helpers =====================
def call_qwen_api(image_path: str, audio_path: str, user_prompt: str, api_url: str, timeout: int):
    if not image_path or not os.path.exists(image_path):
        return False, "❌ Image not found.", None
    if not audio_path or not os.path.exists(audio_path):
        return False, "❌ Audio not found.", None

    try:
        with open(image_path, "rb") as fimg, open(audio_path, "rb") as faud:
            files = {
                "image": ("image"+Path(image_path).suffix, fimg, "application/octet-stream"),
                "audio": ("audio"+Path(audio_path).suffix, faud, "application/octet-stream"),
            }
            # Use DEFAULT_PROMPT when user provided nothing
            data = {"user_prompt": (user_prompt or DEFAULT_PROMPT)}
            r = requests.post(api_url, files=files, data=data, timeout=timeout)
        if r.status_code != 200:
            return False, f"❌ Enhance failed (HTTP {r.status_code}): {r.text[:300]}", None
        pay = r.json()
        if "error" in pay:
            return False, f"❌ Enhance failed: {pay['error']}", None
        return True, "✅ Prompt enhanced.", (pay.get("combined", "") or "")
    except Exception as e:
        return False, f"❌ Enhance failed: {repr(e)}", None

def _write_stream_to_file(resp: requests.Response, out_path: Path):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "wb") as f:
        for chunk in resp.iter_content(chunk_size=1024 * 512):
            if chunk:
                f.write(chunk)
    return str(out_path)

def call_wan_api(image_path: str,
                 audio_path: str,
                 prompt: str,
                 params: dict,
                 api_url: str,
                 timeout: int):
    """
    POST image+audio binaries + params to WAN API.
    Accepts:
      1) raw video/mp4 stream
      2) JSON with {video_url} or {video_b64}
    Returns (ok, status, url_or_local_path|None).
    """
    if not image_path or not os.path.exists(image_path):
        return False, "❌ Image not found.", None
    if not audio_path or not os.path.exists(audio_path):
        return False, "❌ Audio not found.", None

    ts = time.strftime("%Y%m%d_%H%M%S")
    # Local fallback path only if server streams bytes or returns base64
    local_out = CLIENT_TMP_DIR / f"{Path(image_path).stem}_{FIXED_BUCKET}_{int(params.get('infer_frames',120))}f_{ts}.mp4"

    try:
        with open(image_path, "rb") as fimg, open(audio_path, "rb") as faud:
            files = {
                "image": ("image"+Path(image_path).suffix, fimg, "application/octet-stream"),
                "audio": ("audio"+Path(audio_path).suffix, faud, "application/octet-stream"),
            }
            data = {"prompt": prompt or DEFAULT_PROMPT}
            for k, v in (params or {}).items():
                data[k] = str(v)

            r = requests.post(api_url, files=files, data=data, timeout=timeout, stream=True)

        ctype = r.headers.get("content-type", "")
        if r.status_code != 200:
            # try to read JSON error if available
            if "application/json" in ctype:
                try:
                    j = r.json()
                    msg = j.get("error") or j
                except Exception:
                    msg = r.text[:300]
            else:
                msg = r.text[:300]
            return False, f"❌ WAN inference failed (HTTP {r.status_code}): {msg}", None

        # Case 1: direct video bytes
        if "video/" in ctype or "application/octet-stream" in ctype:
            CLIENT_TMP_DIR.mkdir(parents=True, exist_ok=True)
            local_path = _write_stream_to_file(r, local_out)
            return True, f"✅ Done: {local_path}", local_path

        # Case 2: JSON contract
        if "application/json" in ctype:
            pay = r.json()
            if "error" in pay:
                return False, f"❌ WAN inference failed: {pay['error']}", None

            if "video_url" in pay and pay["video_url"]:
                # Let gr.Video read from the server URL directly
                return True, "✅ Done.", pay["video_url"]

            if "video_b64" in pay and pay["video_b64"]:
                CLIENT_TMP_DIR.mkdir(parents=True, exist_ok=True)
                with open(local_out, "wb") as f:
                    f.write(base64.b64decode(pay["video_b64"]))
                return True, f"✅ Saved: {str(local_out)}", str(local_out)

            return False, f"❌ WAN inference returned JSON without video_url/video_b64.", None

        # Unexpected content-type
        return False, f"❌ Unexpected response content-type: {ctype}", None

    except Exception as e:
        return False, f"❌ WAN inference failed: {repr(e)}", None

# ===================== Gradio callbacks =====================
def on_image_preset_select(evt: gr.SelectData):
    idx = evt.index
    if isinstance(idx, (list, tuple)):
        idx = idx[0]
    idx = max(0, min(int(idx), len(IMAGE_PRESETS) - 1))
    return gr.update(value=IMAGE_PRESETS[idx])

def on_audio_preset_select(evt: gr.SelectData):
    idx = evt.index
    if isinstance(idx, (list, tuple)):
        idx = idx[0]
    idx = max(0, min(int(idx), len(AUDIO_PRESETS) - 1))
    return gr.update(value=AUDIO_PRESETS[idx])

def cb_enhance(uploaded_image, uploaded_audio, user_prompt):
    try:
        img = uploaded_image or (IMAGE_PRESETS[0] if IMAGE_PRESETS else None)
        aud = uploaded_audio or (AUDIO_PRESETS[0] if AUDIO_PRESETS else None)
        ok, status, combo = call_qwen_api(
            image_path=img, audio_path=aud, user_prompt=user_prompt or DEFAULT_PROMPT,
            api_url=ARGS.qwen_api_url, timeout=ARGS.qwen_timeout
        )
        return status, (combo or "")
    except Exception as e:
        return f"❌ Enhance failed: {repr(e)}", ""

def cb_run(
    uploaded_image, uploaded_audio,
    enhanced_prompt, user_prompt,
    infer_frames, num_clip, sample_steps, sample_guide_scale, base_seed,
    offload_model, init_first_frame, sample_solver
):
    try:
        img = uploaded_image or (IMAGE_PRESETS[0] if IMAGE_PRESETS else None)
        aud = uploaded_audio or (AUDIO_PRESETS[0] if AUDIO_PRESETS else None)
        # Use enhanced prompt if provided; else user input; else DEFAULT_PROMPT
        prompt = (enhanced_prompt or user_prompt or DEFAULT_PROMPT).strip()

        params = dict(
            size_bucket=FIXED_BUCKET,
            max_area=FIXED_MAX_AREA,
            infer_frames=int(infer_frames),
            num_clip=int(num_clip),
            sample_steps=int(sample_steps),
            sample_guide_scale=float(sample_guide_scale),
            base_seed=int(base_seed),
            offload_model=bool(offload_model),
            init_first_frame=bool(init_first_frame),
            sample_solver=str(sample_solver),
        )

        ok, status, media_ref = call_wan_api(
            image_path=img,
            audio_path=aud,
            prompt=prompt,
            params=params,
            api_url=ARGS.wan_api_url,
            timeout=ARGS.wan_timeout,
        )
        return status, (media_ref if ok else None)
    except Exception as e:
        return f"❌ Generation failed: {repr(e)}", None

# ===================== UI =====================
def build_ui():
    with gr.Blocks(title="Wan 2.2 – S2V-5B (APIs: Qwen Enhance → Wan Infer)") as demo:
        gr.Markdown(
            "## Wan 2.2 – S2V-5B (APIs: Qwen Enhance → Wan Inference)\n"
            "- Resolution fixed to area equal to **720p**, with adaptive aspect ratios (closest resolution bin to the input image)\n"
            "- Click an **image preset** or an **audio preset** to fill the inputs (or upload your own).\n"
            "- Then **Prompt enhancing** → **Run video generation**"
        )

        with gr.Row():
            # LEFT: inputs + enhance
            with gr.Column(scale=1):
                upload_image = gr.Image(
                    type="filepath",
                    label="Image input",
                    value=IMAGE_PRESETS[0] if IMAGE_PRESETS else None
                )
                upload_audio = gr.Audio(
                    type="filepath",
                    label="Audio input",
                    value=AUDIO_PRESETS[0] if AUDIO_PRESETS else None
                )

                user_prompt = gr.Textbox(
                    label="Your prompt (intent for the video)",
                    lines=4,
                    placeholder=DEFAULT_PROMPT
                )

                enhance_btn = gr.Button("✨ Prompt enhancing via Qwen3-Omni")
                enhance_status = gr.Textbox(label="Enhance status", interactive=False)
                enhanced_prompt = gr.Textbox(label="Enhanced prompt (visual + audio)", lines=8)

            # RIGHT: generation controls
            with gr.Column(scale=1):
                gr.Markdown("**Generation hyper-params:**")

                infer_frames = gr.Slider(minimum=80, maximum=160, step=4, value=120, label="infer_frames")
                num_clip = gr.Slider(minimum=1, maximum=30, step=1, value=2, label="num_clip")
                sample_steps = gr.Slider(minimum=1, maximum=50, step=1, value=10, label="sample_steps")
                sample_guide_scale = gr.Slider(minimum=3.5, maximum=7.0, step=0.5, value=6.5, label="sample_guide_scale")
                base_seed = gr.Slider(minimum=-1, maximum=1_000_000_000, step=1, value=-1, label="base_seed (-1=random)")
                offload_model = gr.Checkbox(value=False, label="offload_model (VRAM saver)")
                init_first_frame = gr.Checkbox(value=False, label="init_first_frame")
                sample_solver = gr.Dropdown(choices=["unipc","dpm++"], value="unipc", label="sample_solver")

                run_btn = gr.Button("🎬 Run video generation")
                run_status = gr.Textbox(label="Run status", interactive=False)
                video_out = gr.Video(label="Output (.mp4)")

        # ====== Independent preset pickers ======
        gr.Markdown("### Presets")
        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("**Image presets (click to use)**")
                image_gallery = gr.Gallery(
                    value=IMAGE_PRESETS,
                    columns=[4],
                    height="auto",
                    label="Images"
                )
            with gr.Column(scale=1):
                gr.Markdown("**Audio presets (click to use)**")
                audio_proto = gr.Audio(type="filepath", label="Preset", interactive=False)
                audio_ds = gr.Dataset(
                    components=[audio_proto],
                    samples=[[p] for p in AUDIO_PRESETS],
                    samples_per_page=6,
                    label="Audios"
                )

        # === Wiring ===
        image_gallery.select(
            fn=on_image_preset_select,
            inputs=None,
            outputs=[upload_image],
        )
        audio_ds.select(
            fn=on_audio_preset_select,
            inputs=None,
            outputs=[upload_audio],
        )
        enhance_btn.click(
            fn=cb_enhance,
            inputs=[upload_image, upload_audio, user_prompt],
            outputs=[enhance_status, enhanced_prompt]
        )
        run_btn.click(
            fn=cb_run,
            inputs=[upload_image, uploaded_audio := upload_audio,
                    enhanced_prompt, user_prompt,
                    infer_frames, num_clip, sample_steps, sample_guide_scale, base_seed,
                    offload_model, init_first_frame, sample_solver],
            outputs=[run_status, video_out]
        )
    return demo

# ===================== Startup =====================
def main():
    global ARGS
    ARGS = parse_args()

    asset_dirs = sorted(
        {
            *(str(Path(p).parent) for p in IMAGE_PRESETS),
            *(str(Path(p).parent) for p in AUDIO_PRESETS),
            str(CLIENT_TMP_DIR.resolve()),
        }
    )
    print("[Gradio] allowed_paths =", asset_dirs)

    demo = build_ui()
    demo.launch(server_name="0.0.0.0",
                server_port=ARGS.server_port,
                show_error=True,
                allowed_paths=asset_dirs)

if __name__ == "__main__":
    main()
