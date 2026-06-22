#!/bin/bash
# Train DIAMOND on an Atari game
# Usage: bash train.sh [GAME] [DEVICE]
# Examples:
#   bash train.sh                                   # Breakout on GPU 0
#   bash train.sh PongNoFrameskip-v4                # Pong on GPU 0
#   bash train.sh BreakoutNoFrameskip-v4 cpu        # Breakout on CPU
#   bash train.sh BreakoutNoFrameskip-v4 0,1        # Breakout on GPUs 0 and 1

set -e

PROJECT_DIR="/mnt/c/Users/Adalberto/Documents/code/diamond"
cd "$PROJECT_DIR"

# Activate virtual environment
if [ ! -f ".venv/bin/activate" ]; then
    echo "ERROR: Virtual environment not found. Run setup.sh first."
    exit 1
fi
source .venv/bin/activate

# --- Arguments ---
GAME="${1:-BreakoutNoFrameskip-v4}"
DEVICE="${2:-0}"

# Supported Atari 100k games (same as paper)
SUPPORTED_GAMES=(
    "AlienNoFrameskip-v4"
    "AmidarNoFrameskip-v4"
    "AssaultNoFrameskip-v4"
    "AsterixNoFrameskip-v4"
    "BankHeistNoFrameskip-v4"
    "BattleZoneNoFrameskip-v4"
    "BoxingNoFrameskip-v4"
    "BreakoutNoFrameskip-v4"
    "ChopperCommandNoFrameskip-v4"
    "CrazyClimberNoFrameskip-v4"
    "DemonAttackNoFrameskip-v4"
    "FreewayNoFrameskip-v4"
    "FrostbiteNoFrameskip-v4"
    "GopherNoFrameskip-v4"
    "HeroNoFrameskip-v4"
    "JamesbondNoFrameskip-v4"
    "KangarooNoFrameskip-v4"
    "KrullNoFrameskip-v4"
    "KungFuMasterNoFrameskip-v4"
    "MsPacmanNoFrameskip-v4"
    "PongNoFrameskip-v4"
    "PrivateEyeNoFrameskip-v4"
    "QbertNoFrameskip-v4"
    "RoadRunnerNoFrameskip-v4"
    "SeaquestNoFrameskip-v4"
    "UpNDownNoFrameskip-v4"
)

echo "=========================================="
echo "  DIAMOND Training"
echo "=========================================="
echo "  Game  : $GAME"
echo "  Device: $DEVICE"
echo "  Output: outputs/<date>/<time>/"
echo "=========================================="
echo ""

# Verify GPU if not using CPU
if [ "$DEVICE" != "cpu" ]; then
    python -c "
import torch
if not torch.cuda.is_available():
    print('WARNING: CUDA not available. Training will be slow on CPU.')
    print('         Run: bash train.sh $GAME cpu  to use CPU explicitly.')
else:
    n = torch.cuda.device_count()
    print(f'GPU(s) available: {n}')
    for i in range(n):
        print(f'  [{i}] {torch.cuda.get_device_name(i)}')
"
fi

echo ""
echo "Starting training..."
echo ""

python src/main.py \
    env.train.id="$GAME" \
    common.devices="$DEVICE" \
    wandb.mode=disabled
