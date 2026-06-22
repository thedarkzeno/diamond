#!/bin/bash
# Train DIAMOND+SANA on an Atari game at 512px resolution.
#
# Usage:
#   bash train_sana.sh [OPTIONS]
#
# Quick examples:
#   bash train_sana.sh
#   bash train_sana.sh --fast
#   bash train_sana.sh --game PongNoFrameskip-v4 --batch-size 4
#   bash train_sana.sh --vae-device cpu --batch-size 1          # low VRAM
#   bash train_sana.sh --cached-latents true                    # latent cache (default)
#   python scripts/encode_dataset.py --latest-run               # optional pre-encode before train
#   bash train_sana.sh --help
#
# Env var overrides (same names as flags, uppercase with underscores):
#   GAME=PongNoFrameskip-v4 BATCH_SIZE=4 bash train_sana.sh

set -e

PROJECT_DIR="/mnt/c/Users/Adalberto/Documents/code/diamond"
cd "$PROJECT_DIR"

if [ ! -f ".venv/bin/activate" ]; then
    echo "ERROR: Virtual environment not found. Run setup.sh first."
    exit 1
fi
source .venv/bin/activate

# --- Defaults (override via flags or env vars) ---
GAME="${GAME:-BreakoutNoFrameskip-v4}"
DEVICE="${DEVICE:-0}"
BATCH_SIZE="${BATCH_SIZE:-4}"
GRAD_ACC="${GRAD_ACC:-1}"
RESOLUTION="${RESOLUTION:-512}"
RL_IMG_SIZE="${RL_IMG_SIZE:-64}"
VAE_DEVICE="${VAE_DEVICE:-cuda}"
VAE_MICRO_BATCH="${VAE_MICRO_BATCH:-8}"
VAE_TILING="${VAE_TILING:-true}"
AUTO_ENCODE="${AUTO_ENCODE:-true}"
CACHED_LATENTS="${CACHED_LATENTS:-true}"
FLOW_SHIFT="${FLOW_SHIFT:-3.0}"
RL_DENOISE_STEPS="${RL_DENOISE_STEPS:-4}"
WM_PRELOAD="${WM_PRELOAD:-32}"
STEPS_FIRST="${STEPS_FIRST:-10000}"
STEPS_EPOCH="${STEPS_EPOCH:-400}"
FINAL_EPOCHS="${FINAL_EPOCHS:-50}"
DENOISER_LR="${DENOISER_LR:-1e-4}"
COMPILE_WM="${COMPILE_WM:-true}"
COMPILE_DENOISER="${COMPILE_DENOISER:-true}"
USE_AMP="${USE_AMP:-true}"
WANDB_MODE="${WANDB_MODE:-disabled}"
CONTEXT_SCALE="${CONTEXT_SCALE:-0.1}"
FAST_MODE="${FAST_MODE:-false}"
EXTRA_ARGS=()

usage() {
    cat <<'EOF'
DIAMOND + SANA training launcher

Options:
  --game GAME              Atari env id (default: BreakoutNoFrameskip-v4)
  --device DEVICE          CUDA device id, "cpu", or "all" (default: 0)
  --batch-size N           Batch size for all models (default: 4)
  --grad-acc N             Gradient accumulation steps (default: 1)
  --resolution N           World model / VAE frame size in pixels (default: 512)
  --rl-img-size N          rew_end_model + actor_critic resolution (default: 64)
  --vae-device DEV         VAE device: cuda | cpu (default: cuda)
  --vae-micro-batch N      Frames per VAE encode chunk (default: 8)
  --vae-tiling BOOL        VAE tiling: true | false (default: true)
  --auto-encode BOOL       Encode missing latents after each collection (default: true)
  --cached-latents BOOL    Train on cached latents when available (default: true)
  --use-amp BOOL           BF16 autocast for denoiser (default: true)
  --compile-denoiser BOOL  torch.compile SANA transformer (default: true)
  --flow-shift F           Flow matching shift (default: 3.0)
  --rl-denoise-steps N     Denoise steps in RL imagination (default: 4; lower = faster)
  --wm-preload N           Batches to preload for WM env resets (default: 32)
  --denoise-steps N        Alias for --rl-denoise-steps (deprecated name)
  --steps-first N          Denoiser steps on first training epoch (default: 10000)
  --steps-epoch N          Denoiser steps per later epoch (default: 400)
  --final-epochs N         Training epochs after collection (default: 50)
  --denoiser-lr LR         Denoiser learning rate (default: 1e-4)
  --compile-wm BOOL        torch.compile WM rollout: true | false (default: true)
  --wandb MODE             wandb mode: disabled | online | offline (default: disabled)
  --context-scale F        Patch-embed context frame weight scale (default: 0.1)
  --fast                   Quick experiment preset (256px, fewer steps/epochs)
  --extra "k=v ..."        Extra Hydra overrides (quoted string)
  -h, --help               Show this help

Examples:
  bash train_sana.sh
  bash train_sana.sh --fast
  bash train_sana.sh --batch-size 4 --vae-micro-batch 16
  bash train_sana.sh --vae-device cpu --batch-size 2 --grad-acc 2
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --game)              GAME="$2"; shift 2 ;;
        --device)            DEVICE="$2"; shift 2 ;;
        --batch-size)        BATCH_SIZE="$2"; shift 2 ;;
        --grad-acc)          GRAD_ACC="$2"; shift 2 ;;
        --resolution)        RESOLUTION="$2"; shift 2 ;;
        --rl-img-size)        RL_IMG_SIZE="$2"; shift 2 ;;
        --vae-device)        VAE_DEVICE="$2"; shift 2 ;;
        --vae-micro-batch)   VAE_MICRO_BATCH="$2"; shift 2 ;;
        --vae-tiling)        VAE_TILING="$2"; shift 2 ;;
        --auto-encode)       AUTO_ENCODE="$2"; shift 2 ;;
        --cached-latents)    CACHED_LATENTS="$2"; shift 2 ;;
        --use-amp)           USE_AMP="$2"; shift 2 ;;
        --compile-denoiser)  COMPILE_DENOISER="$2"; shift 2 ;;
        --flow-shift)        FLOW_SHIFT="$2"; shift 2 ;;
        --rl-denoise-steps)  RL_DENOISE_STEPS="$2"; shift 2 ;;
        --wm-preload)        WM_PRELOAD="$2"; shift 2 ;;
        --denoise-steps)     RL_DENOISE_STEPS="$2"; shift 2 ;;
        --steps-first)       STEPS_FIRST="$2"; shift 2 ;;
        --steps-epoch)       STEPS_EPOCH="$2"; shift 2 ;;
        --final-epochs)      FINAL_EPOCHS="$2"; shift 2 ;;
        --denoiser-lr)       DENOISER_LR="$2"; shift 2 ;;
        --compile-wm)        COMPILE_WM="$2"; shift 2 ;;
        --wandb)             WANDB_MODE="$2"; shift 2 ;;
        --context-scale)     CONTEXT_SCALE="$2"; shift 2 ;;
        --fast)              FAST_MODE=true; shift ;;
        --extra)             EXTRA_ARGS+=("$2"); shift 2 ;;
        -h|--help)           usage; exit 0 ;;
        *)
            echo "Unknown option: $1"
            usage
            exit 1
            ;;
    esac
done

if [ "$FAST_MODE" = true ]; then
    RESOLUTION=256
    STEPS_FIRST=2000
    STEPS_EPOCH=100
    FINAL_EPOCHS=10
    BATCH_SIZE=4
    EXTRA_ARGS+=("collection.train.num_steps_total=20000")
fi

echo "=========================================="
echo "  DIAMOND + SANA Training"
echo "=========================================="
echo "  Game           : $GAME"
echo "  Device         : $DEVICE"
echo "  Resolution     : ${RESOLUTION}px (WM / VAE)"
echo "  RL resolution  : ${RL_IMG_SIZE}px (rew_end + actor_critic)"
echo "  Batch size     : $BATCH_SIZE (grad_acc=$GRAD_ACC)"
echo "  VAE device     : $VAE_DEVICE (micro_batch=$VAE_MICRO_BATCH, tiling=$VAE_TILING)"
echo "  Cached latents : $CACHED_LATENTS (auto_encode=$AUTO_ENCODE)"
echo "  AMP / Compile  : amp=$USE_AMP, denoiser=$COMPILE_DENOISER"
echo "  Flow shift     : $FLOW_SHIFT"
echo "  RL denoise     : $RL_DENOISE_STEPS steps (WM imagination)"
echo "  WM preload     : $WM_PRELOAD batches"
echo "  Steps          : first=$STEPS_FIRST, per_epoch=$STEPS_EPOCH"
echo "  Final epochs   : $FINAL_EPOCHS"
echo "  Denoiser LR    : $DENOISER_LR"
echo "  Compile WM     : $COMPILE_WM"
echo "  WandB          : $WANDB_MODE"
echo "=========================================="
echo ""

if [ "$DEVICE" != "cpu" ]; then
    python -c "
import torch
if torch.cuda.is_available():
    for i in range(torch.cuda.device_count()):
        print(f'GPU [{i}] {torch.cuda.get_device_name(i)}')
else:
    print('WARNING: CUDA not available.')
"
    echo ""
fi

HYDRA_OVERRIDES=(
    "env.train.id=$GAME"
    "env.train.size=$RESOLUTION"
    "rl_img_size=$RL_IMG_SIZE"
    "agent.rew_end_model.img_size=$RL_IMG_SIZE"
    "agent.actor_critic.img_size=$RL_IMG_SIZE"
    "common.devices=$DEVICE"
    "wandb.mode=$WANDB_MODE"
    "auto_encode_latents=$AUTO_ENCODE"
    "use_cached_latents=$CACHED_LATENTS"
    "agent.use_cached_latents=$CACHED_LATENTS"
    "agent.denoiser.use_amp=$USE_AMP"
    "agent.vae.device=$VAE_DEVICE"
    "agent.vae.encode_micro_batch=$VAE_MICRO_BATCH"
    "agent.vae.enable_tiling=$VAE_TILING"
    "agent.denoiser.flow_shift=$FLOW_SHIFT"
    "agent.denoiser.inner_model.context_weight_scale=$CONTEXT_SCALE"
    "world_model_env.latent_sampler.num_steps_denoising=$RL_DENOISE_STEPS"
    "world_model_env.latent_sampler.flow_shift=$FLOW_SHIFT"
    "world_model_env.num_batches_to_preload=$WM_PRELOAD"
    "denoiser.training.batch_size=$BATCH_SIZE"
    "denoiser.training.grad_acc_steps=$GRAD_ACC"
    "denoiser.training.steps_first_epoch=$STEPS_FIRST"
    "denoiser.training.steps_per_epoch=$STEPS_EPOCH"
    "denoiser.optimizer.lr=$DENOISER_LR"
    "rew_end_model.training.batch_size=$BATCH_SIZE"
    "rew_end_model.training.grad_acc_steps=$GRAD_ACC"
    "rew_end_model.training.steps_first_epoch=$STEPS_FIRST"
    "rew_end_model.training.steps_per_epoch=$STEPS_EPOCH"
    "actor_critic.training.batch_size=$BATCH_SIZE"
    "actor_critic.training.grad_acc_steps=$GRAD_ACC"
    "actor_critic.training.steps_first_epoch=$((STEPS_FIRST / 2))"
    "actor_critic.training.steps_per_epoch=$STEPS_EPOCH"
    "training.num_final_epochs=$FINAL_EPOCHS"
    "training.compile_wm=$COMPILE_WM"
    "training.compile_denoiser=$COMPILE_DENOISER"
)

if [ ${#EXTRA_ARGS[@]} -gt 0 ]; then
    # shellcheck disable=SC2206
    HYDRA_OVERRIDES+=(${EXTRA_ARGS[@]})
fi

python src/main.py --config-name trainer_sana "${HYDRA_OVERRIDES[@]}"
