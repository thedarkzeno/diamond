#!/bin/bash
# Train SANA world model on CS:GO static dataset (no RL).
#
# Usage:
#   bash train_sana_csgo.sh --path-data /path/to/processed/dataset
#   bash train_sana_csgo.sh --path-data /path/to/dataset --resolution 256
#
# Prepare data first:
#   python scripts/process_csgo_sana_dataset.py /path/to/hdf5 /path/to/out --resolution 256

set -e

PROJECT_DIR="/mnt/c/Users/Adalberto/Documents/code/diamond"
cd "$PROJECT_DIR"

if [ ! -f ".venv/bin/activate" ]; then
    echo "ERROR: Virtual environment not found. Run setup.sh first."
    exit 1
fi
source .venv/bin/activate

PATH_DATA="${PATH_DATA:-}"
RESOLUTION="${RESOLUTION:-256}"
DEVICE="${DEVICE:-0}"
BATCH_SIZE="${BATCH_SIZE:-4}"
GRAD_ACC="${GRAD_ACC:-2}"
STEPS_FIRST="${STEPS_FIRST:-5000}"
STEPS_EPOCH="${STEPS_EPOCH:-400}"
FINAL_EPOCHS="${FINAL_EPOCHS:-50}"
AUTO_ENCODE="${AUTO_ENCODE:-true}"
COMPILE_DENOISER="${COMPILE_DENOISER:-true}"
WANDB_MODE="${WANDB_MODE:-disabled}"
EXTRA_ARGS=()

usage() {
    cat <<'EOF'
SANA + CS:GO static dataset training

Options:
  --path-data PATH         Processed dataset root (train/ + test/) [required]
  --resolution N           Frame size (default: 256)
  --device DEVICE          CUDA device id (default: 0)
  --batch-size N           Denoiser batch size (default: 4)
  --grad-acc N             Gradient accumulation (default: 2)
  --steps-first N          Steps on first epoch (default: 5000)
  --steps-epoch N          Steps per later epoch (default: 400)
  --final-epochs N         Training epochs (default: 50)
  --auto-encode BOOL       Cache VAE latents (default: true)
  --wandb MODE             wandb mode (default: disabled)
  --extra "k=v ..."        Extra Hydra overrides
  -h, --help               Show this help
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --path-data)      PATH_DATA="$2"; shift 2 ;;
        --resolution)     RESOLUTION="$2"; shift 2 ;;
        --device)         DEVICE="$2"; shift 2 ;;
        --batch-size)     BATCH_SIZE="$2"; shift 2 ;;
        --grad-acc)       GRAD_ACC="$2"; shift 2 ;;
        --steps-first)    STEPS_FIRST="$2"; shift 2 ;;
        --steps-epoch)    STEPS_EPOCH="$2"; shift 2 ;;
        --final-epochs)   FINAL_EPOCHS="$2"; shift 2 ;;
        --auto-encode)    AUTO_ENCODE="$2"; shift 2 ;;
        --wandb)          WANDB_MODE="$2"; shift 2 ;;
        --extra)          EXTRA_ARGS+=("$2"); shift 2 ;;
        -h|--help)        usage; exit 0 ;;
        *)
            echo "Unknown option: $1"
            usage
            exit 1
            ;;
    esac
done

if [ -z "$PATH_DATA" ]; then
    echo "ERROR: --path-data is required."
    usage
    exit 1
fi

if [ ! -f "$PATH_DATA/train/info.pt" ]; then
    echo "ERROR: $PATH_DATA/train/info.pt not found. Run process_csgo_sana_dataset.py first."
    exit 1
fi

echo "=========================================="
echo "  DIAMOND + SANA — CS:GO (static dataset)"
echo "=========================================="
echo "  Dataset        : $PATH_DATA"
echo "  Resolution     : ${RESOLUTION}px"
echo "  Device         : $DEVICE"
echo "  Batch size     : $BATCH_SIZE (grad_acc=$GRAD_ACC)"
echo "  Steps          : first=$STEPS_FIRST, per_epoch=$STEPS_EPOCH"
echo "  Final epochs   : $FINAL_EPOCHS"
echo "  Auto encode    : $AUTO_ENCODE"
echo "=========================================="
echo ""

HYDRA_OVERRIDES=(
    "env.path_data=$PATH_DATA"
    "env.train.size=$RESOLUTION"
    "common.devices=$DEVICE"
    "wandb.mode=$WANDB_MODE"
    "auto_encode_latents=$AUTO_ENCODE"
    "use_cached_latents=$AUTO_ENCODE"
    "agent.use_cached_latents=$AUTO_ENCODE"
    "denoiser.training.batch_size=$BATCH_SIZE"
    "denoiser.training.grad_acc_steps=$GRAD_ACC"
    "denoiser.training.steps_first_epoch=$STEPS_FIRST"
    "denoiser.training.steps_per_epoch=$STEPS_EPOCH"
    "training.num_final_epochs=$FINAL_EPOCHS"
    "training.compile_denoiser=$COMPILE_DENOISER"
)

if [ ${#EXTRA_ARGS[@]} -gt 0 ]; then
    HYDRA_OVERRIDES+=("${EXTRA_ARGS[@]}")
fi

python src/main.py --config-name trainer_sana_csgo "${HYDRA_OVERRIDES[@]}"
