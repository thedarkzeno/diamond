from pathlib import Path

import numpy as np
import torch
from tqdm import trange

from data.dataset import Dataset
from data.episode import Episode
from models.vae_wrapper import DCVAEWrapper


def _episode_path(split_dir: Path, episode_id: int) -> Path:
    n = 3
    powers = np.arange(n)
    subfolders = np.floor((episode_id % 10 ** (1 + powers)) / 10**powers) * 10**powers
    subfolders = [int(x) for x in subfolders[::-1]]
    subfolders = "/".join([f"{x:0{n - i}d}" for i, x in enumerate(subfolders)])
    return split_dir / subfolders / f"{episode_id}.pt"


def _latents_path(episode_path: Path) -> Path:
    return episode_path.with_suffix(".latents.pt")


def episode_has_latents(episode_path: Path) -> bool:
    if _latents_path(episode_path).is_file():
        return True
    if not episode_path.is_file():
        return False
    data = torch.load(episode_path, map_location="cpu")
    return "latents" in data


def save_episode_latents(episode_path: Path, episode: Episode, latents: torch.Tensor) -> None:
    """Persist latents and lightweight metadata, skipping pixel rewrites."""
    episode.latents = latents
    episode.obs = latents
    Episode.save_latent_training_bundle(episode_path, episode)


def encode_episode_obs(
    vae: DCVAEWrapper,
    episode: Episode,
    batch_size: int = 16,
) -> torch.Tensor:
    obs = episode.obs
    micro_batch = max(batch_size, vae.cfg.encode_micro_batch)
    prev_micro_batch = vae.cfg.encode_micro_batch
    vae.cfg.encode_micro_batch = micro_batch
    try:
        latents = vae.encode_batch_obs(obs)
    finally:
        vae.cfg.encode_micro_batch = prev_micro_batch
    return latents.cpu()


def upgrade_latent_sidecars(dataset: Dataset, desc: str = "Upgrading latent sidecars") -> int:
    """Upgrade legacy tensor-only sidecars to full training bundles (latents + act metadata)."""
    upgraded = 0
    for episode_id in trange(dataset.num_episodes, desc=desc, disable=dataset.num_episodes == 0):
        episode_path = _episode_path(dataset._directory, episode_id)
        sidecar = _latents_path(episode_path)
        if not sidecar.is_file():
            continue
        payload = torch.load(sidecar, map_location="cpu")
        if isinstance(payload, dict) and "act" in payload:
            continue
        Episode.load_latent_training(episode_path)
        upgraded += 1
    return upgraded


def count_missing_latents(dataset: Dataset) -> int:
    missing = 0
    for episode_id in range(dataset.num_episodes):
        if not episode_has_latents(_episode_path(dataset._directory, episode_id)):
            missing += 1
    return missing


def count_missing_latents_datasets(train_dataset: Dataset, test_dataset: Dataset) -> int:
    return count_missing_latents(train_dataset) + count_missing_latents(test_dataset)


def encode_dataset_split(
    vae: DCVAEWrapper,
    dataset: Dataset,
    batch_size: int = 16,
    desc: str = "Encoding latents",
) -> int:
    """Encode missing latents for all episodes in a dataset. Returns number of episodes encoded."""
    encoded = 0
    for episode_id in trange(dataset.num_episodes, desc=desc, disable=dataset.num_episodes == 0):
        episode_path = _episode_path(dataset._directory, episode_id)
        if episode_has_latents(episode_path):
            continue
        episode = Episode.load(episode_path)
        latents = encode_episode_obs(vae, episode, batch_size=batch_size)
        save_episode_latents(episode_path, episode, latents)
        if dataset._cache_in_ram and episode_id in dataset._cache:
            episode.latents = latents
            dataset._cache[episode_id] = episode
        encoded += 1
    return encoded


def encode_datasets(
    vae: DCVAEWrapper,
    train_dataset: Dataset,
    test_dataset: Dataset,
    batch_size: int = 16,
) -> int:
    total = 0
    total += encode_dataset_split(
        vae, train_dataset, batch_size, desc="Encoding train latents"
    )
    total += encode_dataset_split(
        vae, test_dataset, batch_size, desc="Encoding test latents"
    )
    return total
