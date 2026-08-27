#!/usr/bin/env bash
set -euo pipefail

task_root="${1:-data/pairs_2048}"
task_stamp="$(date +%Y%m%d-%H%M%S)"
task_output="outputs/smoke_train_${task_stamp}"

if [[ ! -f "${task_root}/manifest.jsonl" ]]; then
  echo "Missing pair manifest: ${task_root}/manifest.jsonl" >&2
  exit 1
fi

if ! python -c 'import torch, torchvision' >/dev/null 2>&1; then
  echo "PyTorch and torchvision are required. Install the CUDA wheel on the training machine first." >&2
  exit 1
fi

task_device="$(python - <<'PY'
import torch
print("cuda" if torch.cuda.is_available() else "cpu")
PY
)"

echo "Smoke training on ${task_device}; root=${task_root}; output=${task_output}"
python train.py \
  --root "${task_root}" \
  --out "${task_output}" \
  --epochs 1 \
  --batch 2 \
  --crop-size 256 \
  --workers 0 \
  --amp auto \
  --output-stride 8 \
  --device "${task_device}" \
  --no-pretrained \
  --disable-augment \
  --max-train-batches 1 \
  --max-validation-batches 1
