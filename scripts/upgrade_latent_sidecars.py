#!/usr/bin/env python3
"""Upgrade legacy tensor-only .latents.pt sidecars to full training bundles."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR / "src"))

from data.dataset import Dataset
from data.latent_encoding import upgrade_latent_sidecars


def main() -> None:
    parser = argparse.ArgumentParser(description="Upgrade latent sidecars for fast denoiser data loading.")
    parser.add_argument(
        "--path-data",
        type=Path,
        required=True,
        help="Processed dataset root (train/ + test/).",
    )
    args = parser.parse_args()

    root = args.path_data.expanduser().resolve()
    train = Dataset(root / "train", cache_in_ram=False)
    test = Dataset(root / "test", cache_in_ram=False)
    train.load_from_default_path()
    test.load_from_default_path()

    n_train = upgrade_latent_sidecars(train, desc="Upgrading train sidecars")
    n_test = upgrade_latent_sidecars(test, desc="Upgrading test sidecars")
    print(f"Done. Upgraded {n_train + n_test} sidecar(s) ({n_train} train, {n_test} test).")


if __name__ == "__main__":
    main()
