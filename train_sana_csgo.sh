#!/bin/bash
# Train SANA world model on CS:GO static dataset (no RL).
#
# Usage:
#   bash train_sana_csgo.sh --path-data ~/processed_data
#   bash train_sana_csgo.sh --path-data ~/processed_data --batch-size 12
#
# Prepare data first:
#   bash process_data.sh
#   python scripts/encode_dataset.py --dataset-dir ~/processed_data

set -e

PROJECT_DIR="/mnt/c/Users/Adalberto/Documents/code/diamond"
cd "$PROJECT_DIR"

if [ ! -f ".venv/bin/activate" ]; then
    echo "ERROR: Virtual environment not found. Run setup.sh first."
    exit 1
fi
source .venv/bin/activate

PATH_DATA="${PATH_DATA:-$HOME/processed_data}"
RESOLUTION="${RESOLUTION:-256}"
DEVICE="${DEVICE:-0}"
BATCH_SIZE="${BATCH_SIZE:-8}"
GRAD_ACC="${GRAD_ACC:-1}"
NUM_WORKERS="${NUM_WORKERS:-6}"
PREFETCH="${PREFETCH:-4}"
CACHE_RAM="${CACHE_RAM:-true}"
STEPS_FIRST="${STEPS_FIRST:-5000}"
STEPS_EPOCH="${STEPS_EPOCH:-400}"
FINAL_EPOCHS="${FINAL_EPOCHS:-50}"
AUTO_ENCODE="${AUTO_ENCODE:-true}"
COMPILE_DENOISER="${COMPILE_DENOISER:-true}"
WANDB_MODE="${WANDB_MODE:-online}"
WANDB_PROJECT="${WANDB_PROJECT:-diamond-sana-csgo}"
SAMPLE_EVERY="${SAMPLE_EVERY:-500}"
EXTRA_ARGS=()

usage() {
    cat <<'EOF'
SANA + CS:GO static dataset training

Options:
  --path-data PATH         Processed dataset root (default: ~/processed_data)
  --resolution N           Frame size (default: 256)
  --device DEVICE          CUDA device id (default: 0)
  --batch-size N           Denoiser batch size (default: 8)
  --grad-acc N             Gradient accumulation (default: 1)
  --num-workers N          DataLoader workers (default: 6)
  --prefetch N             Batches prefetched per worker (default: 4)
  --cache-ram BOOL         Cache latent episodes in RAM (default: true)
  --steps-first N          Steps on first epoch (default: 5000)
  --steps-epoch N          Steps per later epoch (default: 400)
  --final-epochs N         Training epochs (default: 50)
  --auto-encode BOOL       Cache VAE latents (default: true)
  --wandb MODE             wandb mode (default: online)
  --wandb-project NAME     wandb project (default: diamond-sana-csgo)
  --sample-every N         Visual sample every N denoiser steps (default: 500)
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
        --num-workers)    NUM_WORKERS="$2"; shift 2 ;;
        --prefetch)       PREFETCH="$2"; shift 2 ;;
        --cache-ram)      CACHE_RAM="$2"; shift 2 ;;
        --steps-first)    STEPS_FIRST="$2"; shift 2 ;;
        --steps-epoch)    STEPS_EPOCH="$2"; shift 2 ;;
        --final-epochs)   FINAL_EPOCHS="$2"; shift 2 ;;
        --auto-encode)    AUTO_ENCODE="$2"; shift 2 ;;
        --wandb)          WANDB_MODE="$2"; shift 2 ;;
        --wandb-project)  WANDB_PROJECT="$2"; shift 2 ;;
        --sample-every)   SAMPLE_EVERY="$2"; shift 2 ;;
        --extra)          EXTRA_ARGS+=("$2"); shift 2 ;;
        -h|--help)        usage; exit 0 ;;
        *)
            echo "Unknown option: $1"
            usage
            exit 1
            ;;
    esac
done

if [ ! -f "$PATH_DATA/train/info.pt" ]; then
    echo "ERROR: $PATH_DATA/train/info.pt not found. Run process_data.sh first."
    exit 1
fi

PATH_DATA="$(cd "$PATH_DATA" && pwd)"

echo "Validating CS:GO dataset..."
python scripts/validate_csgo_dataset.py --path-data "$PATH_DATA" --samples 0

echo "Upgrading latent sidecars for fast loading (one-time per legacy episode)..."
python scripts/upgrade_latent_sidecars.py --path-data "$PATH_DATA"

echo "=========================================="
echo "  DIAMOND + SANA — CS:GO (static dataset)"
echo "=========================================="
echo "  Dataset        : $PATH_DATA"
echo "  Resolution     : ${RESOLUTION}px"
echo "  Device         : $DEVICE"
echo "  Batch size     : $BATCH_SIZE (grad_acc=$GRAD_ACC)"
echo "  DataLoader     : workers=$NUM_WORKERS prefetch=$PREFETCH cache_ram=$CACHE_RAM"
echo "  Steps          : first=$STEPS_FIRST, per_epoch=$STEPS_EPOCH"
echo "  Final epochs   : $FINAL_EPOCHS"
echo "  Auto encode    : $AUTO_ENCODE"
echo "  WandB          : $WANDB_MODE ($WANDB_PROJECT)"
echo "  Sample every   : $SAMPLE_EVERY steps"
echo "=========================================="
echo ""

HYDRA_OVERRIDES=(
    "env.path_data=$PATH_DATA"
    "static_dataset.path=$PATH_DATA"
    "env.train.size=$RESOLUTION"
    "common.devices=$DEVICE"
    "wandb.mode=$WANDB_MODE"
    "wandb.project=$WANDB_PROJECT"
    "visual_sampling.should=true"
    "visual_sampling.every_steps=$SAMPLE_EVERY"
    "auto_encode_latents=$AUTO_ENCODE"
    "use_cached_latents=$AUTO_ENCODE"
    "agent.use_cached_latents=$AUTO_ENCODE"
    "denoiser.training.batch_size=$BATCH_SIZE"
    "denoiser.training.grad_acc_steps=$GRAD_ACC"
    "denoiser.training.steps_first_epoch=$STEPS_FIRST"
    "denoiser.training.steps_per_epoch=$STEPS_EPOCH"
    "training.num_final_epochs=$FINAL_EPOCHS"
    "training.compile_denoiser=$COMPILE_DENOISER"
    "training.cache_in_ram=$CACHE_RAM"
    "training.num_workers_data_loaders=$NUM_WORKERS"
    "training.dataloader_prefetch_factor=$PREFETCH"
)

if [ ${#EXTRA_ARGS[@]} -gt 0 ]; then
    HYDRA_OVERRIDES+=("${EXTRA_ARGS[@]}")
fi

python src/main.py --config-name trainer_sana_csgo "${HYDRA_OVERRIDES[@]}"
