#!/usr/bin/env python3
"""Check processed CS:GO dataset integrity before training or play."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR / "src"))

from csgo.action_layout import ACTION_DIM
from data.csgo_validate import fix_csgo_dataset_actions, validate_csgo_dataset
from data.dataset import Dataset


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate CS:GO processed dataset.")
    parser.add_argument("--path-data", type=Path, required=True, help="Dataset root (train/ + test/).")
    parser.add_argument("--action-dim", type=int, default=ACTION_DIM)
    parser.add_argument("--require-latents", action="store_true")
    parser.add_argument("--fix", action="store_true", help="Truncate/pad episode actions to action_dim.")
    parser.add_argument("--samples", type=int, default=32, help="Episodes to spot-check (default: 32, 0=all).")
    args = parser.parse_args()

    root = args.path_data.expanduser().resolve()
    samples = None if args.samples == 0 else args.samples

    for split in ("train", "test"):
        path = root / split
        if not (path / "info.pt").is_file():
            print(f"SKIP {split}: {path / 'info.pt'} not found")
            continue

        dataset = Dataset(path, split, cache_in_ram=False)
        dataset.load_from_default_path()
        print(f"{split}: {dataset.num_episodes} episodes, {dataset.num_steps} steps")

        if args.fix:
            n = fix_csgo_dataset_actions(dataset, args.action_dim)
            print(f"  fixed {n} episode(s)")

        validate_csgo_dataset(
            dataset,
            args.action_dim,
            num_samples=samples,
            require_latents=args.require_latents,
        )
        print(f"  OK (action_dim={args.action_dim})")

    print("Dataset validation passed.")


if __name__ == "__main__":
    main()
