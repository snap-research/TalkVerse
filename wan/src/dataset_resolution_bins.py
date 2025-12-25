import os
import random
import warnings
import importlib
import numpy as np
import pandas as pd
from PIL import Image
import torch
import av
import torchvision
from pathlib import Path
from PIL import ImageDraw, ImageFont

error_text_file = "./misc/error_text.txt"

def main_print(msg, rank):
    """Prints a message only on the main process."""
    if rank == 0:
        print(msg)

# Fallback aspect-ratio tables (used if wan.utils.multitalk_utils is unavailable)
ASPECT_RATIO_627_FALLBACK = {
     '0.26': ([320, 1216], 1), '0.38': ([384, 1024], 1), '0.50': ([448, 896], 1), '0.67': ([512, 768], 1),
     '0.82': ([576, 704], 1),  '1.00': ([640, 640], 1),  '1.22': ([704, 576], 1), '1.50': ([768, 512], 1),
     '1.86': ([832, 448], 1),  '2.00': ([896, 448], 1),  '2.50': ([960, 384], 1), '2.83': ([1088, 384], 1),
     '3.60': ([1152, 320], 1), '3.80': ([1216, 320], 1), '4.00': ([1280, 320], 1)
}

ASPECT_RATIO_960_FALLBACK = {
     '0.22': ([448, 2048], 1), '0.29': ([512, 1792], 1), '0.36': ([576, 1600], 1), '0.45': ([640, 1408], 1),
     '0.55': ([704, 1280], 1), '0.63': ([768, 1216], 1), '0.76': ([832, 1088], 1), '0.88': ([896, 1024], 1),
     '1.00': ([960, 960], 1), '1.14': ([1024, 896], 1), '1.31': ([1088, 832], 1), '1.50': ([1152, 768], 1),
     '1.58': ([1216, 768], 1), '1.82': ([1280, 704], 1), '1.91': ([1344, 704], 1), '2.20': ([1408, 640], 1),
     '2.30': ([1472, 640], 1), '2.67': ([1536, 576], 1), '2.89': ([1664, 576], 1), '3.62': ([1856, 512], 1),
     '3.75': ([1920, 512], 1)
}


class VideoDatasetWithContextDynamicReso(torch.utils.data.Dataset):
    """
    Single-class dataset:
    - Two resolution bins: fasttalk-480 / fasttalk-720
    - Picks closest aspect-ratio bucket per sample (no dynamic/fixed resolution keys)
    - Optional motion-context frames
    - VAE-stride based conditional image sampling window after the sequence end
    - Audio aligned to the sampled video segment
    """

    def __init__(
        self,
        base_path="",
        metadata_path=[""],
        frame_interval=1,
        num_frames=121,
        # context options
        num_motion_frames=73,
        always_use_motion_frames=True,
        # common
        audio_file_key="audio",
        repeat=1,
        args=None,
        debug=False,
        text_only=False,
        rank=0,
        text_embedding_path="openhumanvid/text_embeddings",
        panda_text_embedding_path="panda70m/text_embeddings",
        random_sample_cond_image=True,
        # two-bin policy
        size_bucket="fasttalk-480",  # or "fasttalk-720"
        # temporal sampling knobs for conditional image window
        vae_t_stride=4,
        sample_offset=15,
        sample_window_size=5,
    ):
        self.random_sample_cond_image = random_sample_cond_image

        # args overrides if provided
        if args is not None:
            base_path = getattr(args, "dataset_base_path", base_path)
            if isinstance(getattr(args, "dataset_metadata_path", None), (list, str)):
                metadata_path = args.dataset_metadata_path
            frame_interval = getattr(args, "frame_interval", frame_interval)
            num_frames = getattr(args, "num_frames", num_frames)
            num_motion_frames = getattr(args, "num_motion_frames", num_motion_frames)
            always_use_motion_frames = getattr(args, "always_use_motion_frames", always_use_motion_frames)
            repeat = getattr(args, "dataset_repeat", repeat)
            if getattr(args, "task", "") in ("fasttalk-1.3B", "phantom-1.3B"):
                self.random_sample_cond_image = True
            size_bucket = getattr(args, "size_bucket", size_bucket)
            vae_t_stride = getattr(args, "vae_t_stride", vae_t_stride)
            sample_offset = getattr(args, "sample_offset", sample_offset)
            sample_window_size = getattr(args, "sample_window_size", sample_window_size)

        if self.random_sample_cond_image:
            main_print("Sampling reference image from offset window after the segment.", rank)
        else:
            main_print("Using the first frame of the sequence as the reference image.", rank)

        if isinstance(metadata_path, str):
            metadata_path = [metadata_path]

        # Load metadata
        all_dfs = []
        for path in metadata_path:
            main_print(f"Loading metadata from: {path}", rank)
            try:
                df = pd.read_csv(path)
                all_dfs.append(df)
            except FileNotFoundError:
                warnings.warn(f"Metadata file not found: {path}. Skipping.")
        if not all_dfs:
            raise FileNotFoundError("No valid metadata files were found. Please check the paths.")
        self.metadata = pd.concat(all_dfs, ignore_index=True)
        main_print(f"Successfully loaded and combined {len(all_dfs)} metadata file(s) with a total of {len(self.metadata)} records.", rank)

        # Core attributes
        self.data_list = [self.metadata.iloc[i].to_dict() for i in range(len(self.metadata))]
        self.base_path = base_path
        self.frame_interval = frame_interval
        self.num_frames = num_frames
        self.num_motion_frames = num_motion_frames
        self.always_use_motion_frames = always_use_motion_frames
        self.audio_file_key = audio_file_key
        self.repeat = repeat
        self.debug = debug
        self.text_only = text_only
        self.text_embedding_path = text_embedding_path
        self.panda_text_embedding_path = panda_text_embedding_path
        self.rank = rank

        # temporal sampling knobs
        self.vae_t_stride = vae_t_stride
        self.sample_offset = sample_offset
        self.sample_window_size = sample_window_size

        # bins only
        self.size_bucket = size_bucket
        self.buckets = self._load_buckets()
        main_print(f"Using resolution bucket: {self.size_bucket}", rank)

    # ----- Buckets -----
    def _load_buckets(self):
        """
        Try to import bucket dicts from wan.utils.multitalk_utils, fallback to included tables.
        """
        try:
            mod = importlib.import_module("wan.utils.multitalk_utils")
            ar627 = getattr(mod, "ASPECT_RATIO_627", ASPECT_RATIO_627_FALLBACK)
            ar960 = getattr(mod, "ASPECT_RATIO_960", ASPECT_RATIO_960_FALLBACK)
        except Exception:
            ar627 = ASPECT_RATIO_627_FALLBACK
            ar960 = ASPECT_RATIO_960_FALLBACK
        return {
            "fasttalk-480": ar627,
            "fasttalk-720": ar960,
        }

    def _closest_bucket_hw(self, ref_img: Image.Image):
        """
        Pick (h, w) from the chosen size bucket that best matches the ref aspect ratio.
        """
        bucket = self.buckets[self.size_bucket]
        ratio = ref_img.height / ref_img.width
        best_key = min(bucket.keys(), key=lambda k: abs(float(k) - ratio))
        target_h, target_w = bucket[best_key][0]  # [h, w]
        return target_h, target_w

    def compute_target_hw(self, image: Image.Image):
        """
        Always pick (h, w) from the selected bucket (two-bin policy).
        """
        return self._closest_bucket_hw(image)

    # ----- Image ops -----
    def crop_and_resize(self, image, target_height, target_width):
        width, height = image.size
        scale = max(target_width / width, target_height / height)
        image = torchvision.transforms.functional.resize(
            image,
            (round(height * scale), round(width * scale)),
            interpolation=torchvision.transforms.InterpolationMode.BILINEAR,
        )
        return torchvision.transforms.functional.center_crop(image, (target_height, target_width))

    # ----- Core -----
    def __getitem__(self, data_id):
        data_row = self.data_list[data_id % len(self.data_list)]
        video_relative_path = data_row.get('video_path')
        if not video_relative_path:
            random_id = random.randint(0, len(self.data_list) - 1)
            with open(error_text_file, "a") as f:
                f.write(f"No video path provided for {data_row}\n")
            return self.__getitem__(random_id)

        file_path_str = os.path.join(self.base_path, video_relative_path)
        data = data_row.copy()
        video_id = video_relative_path.split('/')[-1].split('.')[0]
        data['video_id'] = video_id

        if self.text_only:
            return data

        # text embedding cache
        if "panda70m" in video_relative_path:
            cur_text_cache_path = os.path.join(self.panda_text_embedding_path, f"{video_id}.pt")
        else:
            cur_text_cache_path = os.path.join(self.text_embedding_path, f"{video_id}.pt")
        if not os.path.exists(cur_text_cache_path):
            random_id = random.randint(0, len(self.data_list) - 1)
            with open(error_text_file, "a") as f:
                f.write(f"{video_relative_path} has no text cache path\n")
            return self.__getitem__(random_id)

        try:
            with av.open(file_path_str) as container:
                # Require both video and audio
                if not container.streams.video:
                    warnings.warn(f"Skipping {file_path_str}: no video stream found.")
                    random_id = random.randint(0, len(self.data_list) - 1)
                    with open(error_text_file, "a") as f:
                        f.write(f"{video_relative_path} has no video stream\n")
                    return self.__getitem__(random_id)

                if not container.streams.audio:
                    warnings.warn(f"Skipping {file_path_str}: no audio stream found.")
                    random_id = random.randint(0, len(self.data_list) - 1)
                    with open(error_text_file, "a") as f:
                        f.write(f"{video_relative_path} has no audio stream\n")
                    return self.__getitem__(random_id)

                video_stream = container.streams.video[0]
                audio_stream = container.streams.audio[0]
                video_fps = video_stream.average_rate or video_stream.rate
                assert float(video_fps) == 25.00, f"Invalid video FPS: {video_fps}"

                # Decode all frames
                all_video_frames = [frame.to_image() for frame in container.decode(video=0)]
                total_frames = len(all_video_frames)

                # Determine whether to use motion context
                if self.always_use_motion_frames:
                    required_frames = (self.num_frames + self.num_motion_frames) * self.frame_interval
                    if total_frames < required_frames:
                        random_id = random.randint(0, len(self.data_list) - 1)
                        return self.__getitem__(random_id)
                    actual_video_frames = self.num_frames + self.num_motion_frames
                else:
                    long_required_frames = (self.num_frames + self.num_motion_frames) * self.frame_interval
                    short_required_frames = self.num_frames * self.frame_interval
                    if total_frames >= long_required_frames:
                        actual_video_frames = self.num_frames + self.num_motion_frames
                    elif total_frames >= short_required_frames:
                        actual_video_frames = self.num_frames
                    else:
                        random_id = random.randint(0, len(self.data_list) - 1)
                        return self.__getitem__(random_id)

                # Additional margin for conditional image window (exact user formula)
                margin = self.vae_t_stride * (self.sample_offset - self.sample_window_size)
                max_start = total_frames - (actual_video_frames * self.frame_interval) - margin
                if max_start < 0:
                    random_id = random.randint(0, len(self.data_list) - 1)
                    with open(error_text_file, "a") as f:
                        f.write(f"{video_relative_path} insufficient frames for sampling window (max_start={max_start}).\n")
                    return self.__getitem__(random_id)

                # Target size from bucket using first frame
                first_frame_for_res = all_video_frames[0]
                h, w = self.compute_target_hw(first_frame_for_res)

                # Choose random start within valid range
                random_start_frame = random.randint(0, max_start)

                # Extract frames
                frames = []
                for frame_id in range(actual_video_frames):
                    frame_index = random_start_frame + frame_id * self.frame_interval
                    pil_image = all_video_frames[frame_index]
                    processed_frame = self.crop_and_resize(pil_image, h, w)
                    frames.append(processed_frame)

                # Conditional/reference image sampled from the post-window
                random_end_frame = random_start_frame + actual_video_frames * self.frame_interval
                if self.random_sample_cond_image:
                    # lower = random_end_frame + self.vae_t_stride * (self.sample_offset - self.sample_window_size)
                    # upper = min(total_frames, random_end_frame + self.vae_t_stride * (self.sample_offset + self.sample_window_size))
                    lower = random_start_frame
                    upper = random_end_frame
                    # ensure indices are valid increasing range
                    lower = max(0, int(lower))
                    upper = max(0, int(upper))
                    valid_region = list(range(lower, upper)) if upper > lower else []
                    if not valid_region:
                        random_id = random.randint(0, len(self.data_list) - 1)
                        with open(error_text_file, "a") as f:
                            f.write(f"{video_relative_path} insufficient frames for sampling window (lower={lower}, upper={upper}).\n")
                        return self.__getitem__(random_id)

                    random_frame_index = random.choice(valid_region)
                    random_frame = self.crop_and_resize(all_video_frames[random_frame_index], h, w)
                else:
                    random_frame = frames[0]

                data['video'] = torch.stack([torchvision.transforms.functional.to_tensor(frame) for frame in frames], dim=0)
                data['input_image'] = torchvision.transforms.functional.to_tensor(random_frame)
                data['actual_video_frames'] = actual_video_frames
                data['has_motion_context'] = (actual_video_frames == self.num_frames + self.num_motion_frames)

                # ----- Audio processing -----
                audio_frames = []
                container.seek(0)
                for frame in container.decode(audio=0):
                    audio_frames.append(frame.to_ndarray())
                if not audio_frames:
                    warnings.warn(f"Skipping {file_path_str}: audio stream found but could not be decoded.")
                    random_id = random.randint(0, len(self.data_list) - 1)
                    with open(error_text_file, "a") as f:
                        f.write(f"{video_relative_path} has no audio frames\n")
                    return self.__getitem__(random_id)

                audio_data_np = np.concatenate(audio_frames, axis=1)
                audio_data = torch.from_numpy(audio_data_np).float()
                if audio_data.ndim > 1:
                    audio_data = torch.mean(audio_data, dim=0)  # mono
                audio_sample_rate = audio_stream.rate

                # Resample to 16k if needed
                if audio_sample_rate != 16000:
                    try:
                        import torchaudio
                        resampler = torchaudio.transforms.Resample(orig_freq=audio_sample_rate, new_freq=16000)
                        audio_data = resampler(audio_data)
                        audio_sample_rate = 16000
                        if self.debug:
                            print(f"Resampled audio from {audio_stream.rate} Hz to 16000 Hz for {file_path_str}")
                    except Exception as e:
                        warnings.warn(f"Failed to resample audio for {file_path_str} from {audio_sample_rate} Hz to 16000 Hz: {e}")
                        random_id = random.randint(0, len(self.data_list) - 1)
                        with open(error_text_file, "a") as f:
                            f.write(f"{video_relative_path} failed to resample audio from {audio_sample_rate} Hz to 16000 Hz: {e}\n")
                        return self.__getitem__(random_id)

                # Align audio to the chosen video segment (first to last used video frame)
                start_time_sec = random_start_frame / float(video_fps)
                end_frame_index = random_start_frame + (actual_video_frames - 1) * self.frame_interval
                end_time_sec = end_frame_index / float(video_fps)

                start_sample = int(start_time_sec * audio_sample_rate)
                end_sample = int(end_time_sec * audio_sample_rate)

                video_segment_duration_frames = (actual_video_frames - 1) * self.frame_interval
                required_audio_len = int((video_segment_duration_frames / float(video_fps)) * audio_sample_rate)

                # Force exact length
                end_sample = start_sample + required_audio_len

                if end_sample > audio_data.shape[0]:
                    audio_segment = audio_data[start_sample:]
                    audio_segment = torch.nn.functional.pad(
                        audio_segment,
                        (0, required_audio_len - audio_segment.shape[0]),
                        mode='constant',
                        value=audio_segment[-1] if audio_segment.shape[0] > 0 else 0.0
                    )
                else:
                    audio_segment = audio_data[start_sample:end_sample]

                if audio_segment.shape[0] != required_audio_len:
                    if audio_segment.shape[0] < required_audio_len:
                        padding_needed = required_audio_len - audio_segment.shape[0]
                        audio_segment = torch.nn.functional.pad(audio_segment, (0, padding_needed), mode='constant', value=0.0)
                    else:
                        audio_segment = audio_segment[:required_audio_len]

                assert audio_segment.shape[0] == required_audio_len, f"Audio segment length mismatch for {file_path_str}: expected {required_audio_len}, got {audio_segment.shape[0]}"
                assert audio_segment.abs().sum() > 1e-6, "Audio segment is all zeros!"
                assert not (torch.isnan(audio_segment).any() or torch.isinf(audio_segment).any()), "Audio segment contains NaN or Inf values!"

                data[self.audio_file_key] = audio_segment

                # Text embeddings
                data["context"] = torch.load(cur_text_cache_path, map_location='cpu', weights_only=True).squeeze(0)
                data["context_null"] = torch.load(
                    os.path.join(self.text_embedding_path, "null_embedding.pt"), map_location='cpu', weights_only=True
                ).unsqueeze(0)

                # Optional debug visualization
                if self.debug:
                    visualize_video_and_audio(
                        video_frames_tensor=data['video'],
                        audio_signal_tensor=data[self.audio_file_key],
                        output_path=f"./debug_output/{video_id}_context.mp4",
                        video_fps=video_fps,
                        audio_sample_rate=audio_sample_rate,
                        num_motion_frames=self.num_motion_frames if data['has_motion_context'] else 0,
                        num_main_frames=self.num_frames,
                    )
                    output_image = torchvision.transforms.functional.to_pil_image(data['input_image'])
                    img_output_path = f"./debug_output/{video_id}_context.jpg"
                    img_output_path = Path(img_output_path)
                    img_output_path.parent.mkdir(parents=True, exist_ok=True)
                    output_image.save(img_output_path)

                return data

        except av.AVError as e:
            warnings.warn(f"PyAV Error processing file {file_path_str}: {e}")
            random_id = random.randint(0, len(self.data_list) - 1)
            with open(error_text_file, "a") as f:
                f.write(f"{file_path_str} reports PyAV Error: {e}\n")
            return self.__getitem__(random_id)
        except Exception as e:
            warnings.warn(f"{file_path_str} reports unexpected error: {e}")
            random_id = random.randint(0, len(self.data_list) - 1)
            with open(error_text_file, "a") as f:
                f.write(f"{file_path_str} reports unexpected error: {e}\n")
            return self.__getitem__(random_id)

    def __len__(self):
        return len(self.data_list) * self.repeat


def custom_collate_fn(batch):
    """
    Custom collate function to handle filtering of None values and
    batching of data with variable-length tensors.
    """
    batch = [item for item in batch if item is not None]
    if not batch:
        return None

    collated_batch = {}
    elem = batch[0]
    keys = elem.keys()

    for key in keys:
        if key == 'context' or key == 'context_null':
            collated_batch[key] = [d[key] for d in batch]
        elif isinstance(elem[key], torch.Tensor):
            collated_batch[key] = torch.stack([d[key] for d in batch], 0)
        else:
            collated_batch[key] = [d[key] for d in batch]
    return collated_batch


def visualize_video_and_audio(
    video_frames_tensor,
    audio_signal_tensor,
    output_path,
    video_fps=25,
    audio_sample_rate=16000,
    num_motion_frames=0,
    num_main_frames=None,
):
    """
    Saves a batch of video frames and an audio signal to an MP4 file.
    Annotates each frame as MOTION CTX or MAIN (real frames to be processed).
    """
    # --- FS prep ---
    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    # --- Convert tensors to numpy ---
    video_frames_np = (video_frames_tensor.permute(0, 2, 3, 1) * 255).byte().cpu().numpy()
    audio_signal_np = (audio_signal_tensor.cpu().numpy() * 32767).astype(np.int16)
    if audio_signal_np.ndim > 1:
        audio_signal_np = audio_signal_np.mean(axis=0, dtype=np.int16)

    T, H, W, _ = video_frames_np.shape
    if num_main_frames is None:
        num_main_frames = max(0, T - int(num_motion_frames))
    num_motion_frames = int(num_motion_frames)

    # --- Helpers for drawing ---
    def annotate_frame(frame_np, label_text, color_rgb=(50, 205, 50), border_px=6):
        # color_rgb as (R,G,B), e.g., green for MAIN, orange for MOTION
        img = Image.fromarray(frame_np, mode='RGB').convert('RGBA')
        overlay = Image.new('RGBA', img.size, (0, 0, 0, 0))
        draw = ImageDraw.Draw(overlay)
        font = ImageFont.load_default()

        # border
        if border_px > 0:
            # four sides
            draw.rectangle([0, 0, W-1, border_px-1], fill=color_rgb + (255,))
            draw.rectangle([0, H-border_px, W-1, H-1], fill=color_rgb + (255,))
            draw.rectangle([0, 0, border_px-1, H-1], fill=color_rgb + (255,))
            draw.rectangle([W-border_px, 0, W-1, H-1], fill=color_rgb + (255,))

        # label box
        pad = 8
        tw, th = draw.textbbox((0, 0), label_text, font=font)[2:]
        box_w, box_h = tw + pad * 2, th + pad * 2
        # translucent black background for readability
        draw.rectangle([border_px, border_px, border_px + box_w, border_px + box_h], fill=(0, 0, 0, 140))
        # text
        draw.text((border_px + pad, border_px + pad), label_text, fill=(255, 255, 255, 255), font=font)

        # compose
        out = Image.alpha_composite(img, overlay).convert('RGB')
        return np.asarray(out, dtype=np.uint8)

    # --- Encode with annotations ---
    try:
        with av.open(output_path, mode='w') as container:
            video_stream = container.add_stream('libx264', rate=video_fps)
            video_stream.width = W
            video_stream.height = H
            video_stream.pix_fmt = 'yuv420p'

            audio_stream = container.add_stream('aac', rate=audio_sample_rate)
            audio_stream.layout = 'mono'

            for i, frame_np in enumerate(video_frames_np):
                if i < num_motion_frames:
                    # MOTION CTX (orange)
                    label = f"MOTION CTX [{i+1}/{num_motion_frames}]"
                    color = (255, 140, 0)
                else:
                    # MAIN (green)
                    main_idx = i - num_motion_frames + 1
                    label = f"MAIN [{main_idx}/{num_main_frames}]"
                    color = (50, 205, 50)
                annotated = annotate_frame(frame_np, label, color_rgb=color, border_px=6)

                frame = av.VideoFrame.from_ndarray(annotated, format='rgb24')
                for packet in video_stream.encode(frame):
                    container.mux(packet)
            for packet in video_stream.encode():
                container.mux(packet)

            audio_frame = av.AudioFrame.from_ndarray(audio_signal_np[np.newaxis, :], format='s16', layout='mono')
            audio_frame.sample_rate = audio_sample_rate
            for packet in audio_stream.encode(audio_frame):
                container.mux(packet)
            for packet in audio_stream.encode():
                container.mux(packet)
    except av.AVError as e:
        print(f"Failed to save video to {output_path} due to PyAV error: {e}")
    except Exception as e:
        print(f"An unexpected error occurred while saving video: {e}")



if __name__ == "__main__":
    # Example usage
    dataset = VideoDatasetWithContextDynamicReso(
        debug=True,
        random_sample_cond_image=True,
        size_bucket="fasttalk-720",  # or "fasttalk-480"
        vae_t_stride=4,
        sample_offset=20,
        sample_window_size=10,
        num_motion_frames=73,
        always_use_motion_frames=True,
    )
    dataloader = torch.utils.data.DataLoader(
        dataset,
        shuffle=True,
        num_workers=0,
        pin_memory=True,
        batch_size=2,
        collate_fn=custom_collate_fn
    )

    for i, data in enumerate(dataloader):
        if i >= 10:
            break
        if data is not None:
            print(
                f"Sample {i}: video_id {data['video_id']}, "
                f"video shape {tuple(data['video'].shape)}, "
                f"image shape {tuple(data['input_image'].shape)}, "
                f"audio shape {tuple(data.get(dataset.audio_file_key, torch.empty(0)).shape)}, "
                f"text cache shape {[c.shape for c in data['context']] if 'context' in data else 'N/A'} "
                f"null text shape {[c.shape for c in data['context_null']] if 'context_null' in data else 'N/A'}"
            )
        else:
            print(f"Sample {i} is None.")
    print("Dataset loading complete.")
    print(f"Total samples: {len(dataset)}")
