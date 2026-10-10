#!/bin/bash
# Semantic Drone Dataset (TU Graz ICG): 400 public training images, 6000 x 4000 px, with
# colour-coded masks. Non-commercial use with attribution; the data may not be redistributed,
# so it is downloaded here and never committed. Run on a login node (needs internet).
#   bash scripts/download_data.sh            # into $DATA (cluster) or ./data, then prepared
set -euo pipefail
cd "$(dirname "$0")/.."
DATA="${DATA:-data}"
mkdir -p "$DATA"
ZIP="$DATA/SemanticDroneDatasetPublic.zip"
if [[ ! -s "$ZIP" ]]; then
    curl -fL -C - -o "$ZIP.part" "https://cloud.tugraz.at/index.php/s/csxYfaKmie6LyqA/download"
    mv "$ZIP.part" "$ZIP"
fi
# The archive holds two more archives; only the semantic one is needed (not the 13 GB 3D set).
RAW="$DATA/SemanticDroneDatasetPublic"
if [[ ! -d "$RAW/semantics" ]]; then
    unzip -q -o "$ZIP" "*semantics*.zip" -d "$RAW"
    unzip -q -o "$(find "$RAW" -name "semantic_drone_dataset_semantics*.zip" | head -1)" -d "$RAW/semantics"
fi
echo "extracted to $RAW/semantics; prepare with: python3 slurm/generate.py prepare --submit"
