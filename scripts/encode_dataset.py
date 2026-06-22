#!/usr/bin/env python3
"""Pre-encode dataset episodes into DC-AE latents for SANA world model training."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import torch

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR / "src"))

from data.latent_encoding import encode_datasets
from data.dataset import Dataset
from models.vae_wrapper import DCVAEWrapper, VAEConfig


def find_latest_run_dataset(root: Path) -> Path | None:
    outputs = root / "outputs"
    if not outputs.is_dir():
        return None
    candidates = []
    for date_dir in outputs.iterdir():
        if not date_dir.is_dir():
            continue
        for time_dir in date_dir.iterdir():
            dataset_dir = time_dir / "dataset"
            if (dataset_dir / "train" / "info.pt").is_file():
                candidates.append(dataset_dir)
    if not candidates:
        return None
    return max(candidates, key=lambda p: p.stat().st_mtime)


def resolve_dataset_dir(root: Path, dataset_dir: Path | None, latest_run: bool) -> Path:
    if latest_run:
        found = find_latest_run_dataset(root)
        if found is None:
            raise FileNotFoundError("No Hydra run with dataset/train/info.pt found under outputs/.")
        return found
    if dataset_dir is not None:
        return dataset_dir
    return root / "dataset"


def main() -> None:
    parser = argparse.ArgumentParser(description="Cache DC-AE latents for DIAMOND+SANA training.")
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        default=None,
        help="Dataset root containing train/ and test/ (default: ./dataset)",
    )
    parser.add_argument(
        "--latest-run",
        action="store_true",
        help="Encode the most recent Hydra output dataset under outputs/",
    )
    parser.add_argument("--model-id", type=str, default="Efficient-Large-Model/Sana_600M_512px_diffusers")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--enable-tiling", action="store_true")
    args = parser.parse_args()

    dataset_dir = resolve_dataset_dir(ROOT_DIR, args.dataset_dir, args.latest_run)
    print(f"Dataset directory: {dataset_dir.resolve()}")
    print("Note: training with Hydra writes to outputs/<date>/<time>/dataset — use --latest-run for that path.")

    device = torch.device(args.device)
    vae = DCVAEWrapper(
        VAEConfig(model_id=args.model_id, freeze=True, enable_tiling=args.enable_tiling, device=str(device))
    )
    vae.apply_device_policy(device)

    train_dataset = Dataset(dataset_dir / "train", cache_in_ram=False)
    test_dataset = Dataset(dataset_dir / "test", cache_in_ram=False)
    train_dataset.load_from_default_path()
    test_dataset.load_from_default_path()

    total = encode_datasets(vae, train_dataset, test_dataset, batch_size=args.batch_size)
    print(f"Latent encoding complete. Encoded {total} episode(s).")


if __name__ == "__main__":
    main()
