#!/usr/bin/env bash
# P1.2: extract DINOv2-S and DINOv2-L (224 px, CLS of layers -1/-2/-4, one forward pass) for the frozen 12,000-object
# development manifest MIMO-HS3-DEV12K-v1, with the unchanged, already-validated extractor reports/dinov2/source/dinov2_embed.py.
# DINOv2-B is NOT re-extracted: its validated cache (reports/dinov2/embeddings/dev_dinov2_b_224.json) is reused.
# Resumable: dinov2_embed.py keeps 16-image chunks in data/dinov2/cache/<fingerprint>/ keyed by the (image_id, sha256) list;
# re-running skips finished chunks. Peak RSS / wall time are captured with GNU time.
# Usage (repo root):  bash reports/roadmap/phases/P1/source/run_extraction.sh [dinov2_s|dinov2_l ...]  [BATCH=16 THREADS=6]
set -euo pipefail
cd "$(dirname "$0")/../../../../.."
LOG=reports/roadmap/phases/P1/artifacts/logs; mkdir -p "$LOG"
BATCH=${BATCH:-16}; THREADS=${THREADS:-6}
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1   # weights are local (pinned sha256); never download
for m in "${@:-dinov2_s dinov2_l}"; do
  for model in $m; do
    echo "== $model  batch=$BATCH threads=$THREADS  $(date -u +%FT%TZ)" | tee -a "$LOG/extract_${model}.log"
    /usr/bin/time -v -o "$LOG/extract_${model}.time" \
      .venv/bin/python reports/dinov2/source/dinov2_embed.py --set dev --model "$model" --size 224 --threads "$THREADS" --batch "$BATCH" \
      2>&1 | tee -a "$LOG/extract_${model}.log"
  done
done
