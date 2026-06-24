#!/usr/bin/env python3
"""Convert extracted CS:GO HDF5 episodes to DIAMOND format for SANA training.

Expects a directory of unpacked .hdf5 files (from TeaPearce dataset tars).
Outputs train/ and test/ episode folders at a square resolution for SANA VAE.

Usage:
  python scripts/process_csgo_sana_dataset.py /path/to/hdf5_dir /path/to/out_dir --resolution 256
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

import torch
import torchvision.transforms.functional as T
from tqdm import trange

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR / "src"))

from data.csgo_hdf5 import CSGOHdf5Dataset
from data.dataset import Dataset
from data.segment import SegmentId


def read_split_filenames(path: Path) -> set[str]:
    raw = path.read_bytes()
    if raw.startswith(b"\xff\xfe") or raw.startswith(b"\xfe\xff"):
        text = raw.decode("utf-16")
    else:
        text = raw.decode("utf-8-sig")
    return {line.strip() for line in text.splitlines() if line.strip()}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare CS:GO HDF5 data for SANA world model training.")
    parser.add_argument("hdf5_dir", type=Path, help="Directory containing extracted .hdf5 episode files.")
    parser.add_argument("out_dir", type=Path, help="Output directory (will contain train/ and test/).")
    parser.add_argument("--resolution", type=int, default=256, help="Square frame size for SANA VAE (default: 256).")
    parser.add_argument(
        "--test-split",
        type=Path,
        default=ROOT_DIR / "scripts" / "csgo_test_split.txt",
        help="Text file with HDF5 filenames in the test split.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace an existing output directory instead of failing.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    hdf5_dir = args.hdf5_dir.expanduser().resolve()
    out_dir = args.out_dir.expanduser().resolve()

    if not hdf5_dir.is_dir():
        raise FileNotFoundError(f"HDF5 directory not found: {hdf5_dir}")
    if out_dir.exists():
        if args.overwrite:
            print(f"Removing existing output directory: {out_dir}")
            shutil.rmtree(out_dir)
        else:
            raise FileExistsError(
                f"Output directory already exists: {out_dir}. "
                "Pass --overwrite to rebuild from scratch."
            )

    test_files: set[str] = set()
    if args.test_split.is_file():
        test_files = read_split_filenames(args.test_split)

    csgo = CSGOHdf5Dataset(hdf5_dir)
    train_dataset = Dataset(out_dir / "train", "train")
    test_dataset = Dataset(out_dir / "test", "test")

    for episode_id in trange(csgo.num_episodes, desc="Processing episodes"):
        episode = csgo.load_episode(episode_id)
        episode.obs = T.resize(
            episode.obs,
            [args.resolution, args.resolution],
            interpolation=T.InterpolationMode.BICUBIC,
        )
        filename = csgo._filenames[episode_id].name
        episode.info = {"source_hdf5": str(csgo._filenames[episode_id])}
        target = test_dataset if filename in test_files else train_dataset
        target.add_episode(episode)

    train_dataset.save_to_default_path()
    test_dataset.save_to_default_path()

    print(f"Done. train={train_dataset.num_episodes} episodes, test={test_dataset.num_episodes} episodes")
    print(f"Set env.path_data={out_dir} in config/env/csgo.yaml or pass env.path_data={out_dir} to train_sana_csgo.sh")


if __name__ == "__main__":
    main()
