#!/usr/bin/env bash
set -euo pipefail

echo "Starting S2V 5B Batch Generation (multi-GPU sharded)..."
echo "======================================================="

# -------------------------
# Configuration
# -------------------------
TASK="s2v-5B"
CKPT_DIR="ckpts/Wan2.2-TI2V-5B"
BATCH_FILE="json_batch_inference_config/batch_config.json"
OUTPUT_DIR="outputs/s2v_5b_batch_generation-6.5-seed420"
SIZE="1280*704"
INFER_FRAMES=120
NUM_CLIP=15
SAMPLING_STEPS=50
GUIDE_SCALE=6.5
SEED=420

# -------------------------
# Ensure batch config exists
# -------------------------
if [ ! -f "$BATCH_FILE" ]; then
  echo "Creating batch configuration..."
  python3 create_batch_config.py
fi

mkdir -p "$OUTPUT_DIR"

# -------------------------
# Detect available GPUs
# Priority:
# 1) CUDA_VISIBLE_DEVICES
# 2) nvidia-smi
# Fallback: 1 GPU
# -------------------------
GPU_ENV="${CUDA_VISIBLE_DEVICES:-}"
declare -a GPU_IDS=()
if [[ -n "$GPU_ENV" ]]; then
  IFS=',' read -ra TOKS <<< "$GPU_ENV"
  for t in "${TOKS[@]}"; do
    t="$(echo "$t" | xargs)"
    if [[ -n "$t" && "$t" != "-1" ]]; then
      GPU_IDS+=("$t")
    fi
  done
else
  if command -v nvidia-smi >/dev/null 2>&1; then
    COUNT=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | grep -c . || true)
    if [[ "$COUNT" -gt 0 ]]; then
      for ((i=0; i<COUNT; i++)); do GPU_IDS+=("$i"); done
    fi
  fi
fi

# Fallback to single GPU if detection failed
if [[ "${#GPU_IDS[@]}" -eq 0 ]]; then
  GPU_IDS=("0")
fi

NUM_GPUS="${#GPU_IDS[@]}"
echo "Using ${NUM_GPUS} GPU(s): ${GPU_IDS[*]}"

# -------------------------
# Split JSON into NUM_GPUS shards (preserving order)
# Accepts JSON array (preferred). If JSONL, it will also work.
# -------------------------
SHARDS_DIR="${OUTPUT_DIR}/shards"
mkdir -p "$SHARDS_DIR"

# We capture shard index, path, and count from Python
mapfile -t SHARD_INFO < <(python3 - "$BATCH_FILE" "$NUM_GPUS" "$SHARDS_DIR" <<'PY'
import sys, json, os
from json import JSONDecodeError

in_path = sys.argv[1]
n = int(sys.argv[2])
out_dir = sys.argv[3]
base = os.path.splitext(os.path.basename(in_path))[0]

# Load items: prefer JSON array; fallback to JSONL
items = []
with open(in_path, 'r') as f:
    data = f.read().strip()
    try:
        parsed = json.loads(data)
        if isinstance(parsed, list):
            items = parsed
        else:
            raise JSONDecodeError("Top-level JSON is not a list", data, 0)
    except JSONDecodeError:
        # Try JSON Lines
        items = [json.loads(line) for line in data.splitlines() if line.strip()]

length = len(items)
# Split into n roughly-equal contiguous shards
base_sz = length // n
rem = length % n

start = 0
for i in range(n):
    extra = 1 if i < rem else 0
    end = start + base_sz + extra
    part = items[start:end]
    out_path = os.path.join(out_dir, f"{base}_shard_{i}.json")
    with open(out_path, 'w') as o:
        json.dump(part, o, indent=2)
    # Print: index, path, count (tab-separated)
    print(f"{i}\t{out_path}\t{len(part)}")
    start = end
PY
)

echo "Sharded batch files:"
printf '  %s\n' "${SHARD_INFO[@]}"

# -------------------------
# Launch one process per GPU (skip empty shards)
# -------------------------
pids=()
for ((i=0; i<NUM_GPUS; i++)); do
  # Parse the i-th line from SHARD_INFO
  IFS=$'\t' read -r shard_idx shard_path shard_count <<< "${SHARD_INFO[$i]}"

  if [[ "$shard_count" -eq 0 ]]; then
    echo "GPU ${GPU_IDS[$i]}: shard ${shard_idx} is empty; skipping."
    continue
  fi

  GPU="${GPU_IDS[$i]}"
  OUT_DIR_GPU="${OUTPUT_DIR}"
  mkdir -p "$OUT_DIR_GPU"

  echo "Launching shard ${shard_idx} (${shard_count} items) on GPU ${GPU}..."
  (
    export CUDA_VISIBLE_DEVICES="${GPU}"
    python3 generate.py \
      --task "$TASK" \
      --ckpt_dir "$CKPT_DIR" \
      --batch_file "$shard_path" \
      --output_dir "$OUT_DIR_GPU" \
      --size "$SIZE" \
      --infer_frames "$INFER_FRAMES" \
      --num_clip "$NUM_CLIP" \
      --sample_steps "$SAMPLING_STEPS" \
      --sample_guide_scale "$GUIDE_SCALE" \
      --base_seed "$SEED" \
      --v_scale 1.0
  ) & pids+=($!)
done

# -------------------------
# Wait for all workers
# -------------------------
if [[ "${#pids[@]}" -eq 0 ]]; then
  echo "No shards launched (input was empty?). Exiting."
  exit 0
fi

echo "Waiting for ${#pids[@]} worker(s) to finish..."
fail=0
for pid in "${pids[@]}"; do
  if ! wait "$pid"; then
    fail=$((fail+1))
  fi
done

if [[ "$fail" -eq 0 ]]; then
  echo "Batch generation completed successfully on all GPUs."
else
  echo "Batch generation finished with ${fail} failure(s)."
fi

echo "Results saved under: $OUTPUT_DIR"
