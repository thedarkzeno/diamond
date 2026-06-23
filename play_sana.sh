#!/bin/bash
# Play the world model from the most recent SANA training run.
#
# Usage:
#   bash play_sana.sh                    # Atari (latest run)
#   bash play_sana_csgo.sh --windowed    # CS:GO SANA world model (WSL-friendly)
#   bash play_sana.sh --run-dir outputs/2026-06-21/21-43-00
#
# Controls in the game window:
#   m       - switch human / policy control
#   Tab     - switch env: wm (world model) / test / train
#   Up/Down - imagination horizon

set -e

PROJECT_DIR="/mnt/c/Users/Adalberto/Documents/code/diamond"
cd "$PROJECT_DIR"

if [ ! -f ".venv/bin/activate" ]; then
    echo "ERROR: Virtual environment not found. Run setup.sh first."
    exit 1
fi
source .venv/bin/activate

python src/play.py --latest-run "$@"
