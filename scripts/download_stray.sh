#!/bin/bash
#
# Download the example StrayScanner (iPhone RGB-D) scene from VSLAM-LAB HuggingFace.
# Unpacks raw data under datasets/stray_raw/<scene_id>/. Does not preprocess.
#
# For ingest / SLAM preparation, follow README (Stray Scanner):
#   python scripts/stray_ingest.py <scene_id>
#
# Usage:
#   bash scripts/download_stray.sh

set -e

SCENE="4e41d0a7da"
RAW_DIR="datasets/stray_raw"
HF_REPO="vslamlab/strayscanner"

dst="$RAW_DIR/$SCENE"
if [ -d "$dst" ]; then
    echo "$SCENE already exists, skipping download."
else
    echo "Downloading $SCENE from HuggingFace ($HF_REPO)..."
    mkdir -p "$RAW_DIR"

    if command -v huggingface-cli &> /dev/null; then
        huggingface-cli download "$HF_REPO" "${SCENE}.zip" \
            --repo-type dataset --local-dir "$RAW_DIR"
    else
        wget -q --show-progress -O "$RAW_DIR/${SCENE}.zip" \
            "https://huggingface.co/datasets/${HF_REPO}/resolve/main/${SCENE}.zip"
    fi
    unzip -q "$RAW_DIR/${SCENE}.zip" -d "$RAW_DIR"
    rm -f "$RAW_DIR/${SCENE}.zip"
    echo "Done: $dst"
fi

echo ""
echo "Raw data: $RAW_DIR/$SCENE"
echo ""
echo "Next (README, Stray Scanner):"
echo "  python scripts/stray_ingest.py $SCENE"
