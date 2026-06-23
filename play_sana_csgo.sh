#!/bin/bash
# Play a trained SANA CS:GO world model (human control in imagination).
#
# Usage:
#   bash play_sana_csgo.sh
#   bash play_sana_csgo.sh --run-dir outputs/2026-06-23/11-12-42
#   bash play_sana_csgo.sh --windowed          # recommended on WSL
#
# Controls:
#   WASD, space, mouse  - CS:GO actions
#   m                   - toggle human / replay dataset actions
#   ↑ / ↓               - imagination horizon
#   Esc                 - quit

set -e

PROJECT_DIR="/mnt/c/Users/Adalberto/Documents/code/diamond"
cd "$PROJECT_DIR"

if [ ! -f ".venv/bin/activate" ]; then
    echo "ERROR: Virtual environment not found. Run setup.sh first."
    exit 1
fi
source .venv/bin/activate

python src/play.py --latest-run --path-data "$HOME/processed_data" --windowed "$@"
