#!/bin/bash
# Pre-fetch the Cityscapes-pretrained YOLO26 semantic weights on a login node (jobs run offline).
set -euo pipefail
cd "$(dirname "$0")/.."
python - <<'PY'
from ultralytics import YOLO

for name in ("yolo26n-sem.pt", "yolo26s-sem.pt"):
    YOLO(name)
    print("cached", name)
PY
