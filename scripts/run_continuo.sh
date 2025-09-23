#!/usr/bin/env bash
set -euo pipefail
VIDEO_PATH="${1:-data/raw_videos/VIDEO.mp4}"

PYTHONPATH=src python3 -m rdm_monitor \
  --src "${VIDEO_PATH}" \
  --engine diff \
  --idle_stop_sec 20 \
  --width 960 \
  --out "outputs/annotated/$(basename "${VIDEO_PATH%.*}")_annotated.mp4" \
  --csv "outputs/csv/operacion_sesion.csv"
