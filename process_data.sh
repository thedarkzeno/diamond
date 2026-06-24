#!/bin/bash
# Convert TeaPearce HDF5 episodes to DIAMOND format for SANA training.
#
# Usage:
#   bash process_data.sh
#   HDF5_DIR=./data OUT_DIR=~/processed_data bash process_data.sh
#   OVERWRITE=true bash process_data.sh   # rebuild after adding more HDF5 files

set -e

PROJECT_DIR="/mnt/c/Users/Adalberto/Documents/code/diamond"
cd "$PROJECT_DIR"

if [ ! -f ".venv/bin/activate" ]; then
    echo "ERROR: Virtual environment not found. Run setup.sh first."
    exit 1
fi
source .venv/bin/activate

HDF5_DIR="${HDF5_DIR:-./data}"
OUT_DIR="${OUT_DIR:-./processed_data}"
RESOLUTION="${RESOLUTION:-256}"
OVERWRITE="${OVERWRITE:-false}"

ARGS=("$HDF5_DIR" "$OUT_DIR" --resolution "$RESOLUTION")
if [ "$OVERWRITE" = "true" ]; then
    ARGS+=(--overwrite)
fi

echo "HDF5 input : $HDF5_DIR"
echo "Output     : $OUT_DIR"
echo "Resolution : ${RESOLUTION}px"
echo "Overwrite  : $OVERWRITE"
echo ""

python scripts/process_csgo_sana_dataset.py "${ARGS[@]}"

echo ""
echo "Next: encode latents (required after reprocessing)"
echo "  python scripts/encode_dataset.py --dataset-dir $OUT_DIR"
