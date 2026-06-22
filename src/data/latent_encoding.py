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


def encode_episode_obs(vae: DCVAEWrapper, episode: Episode, batch_size: int = 16) -> torch.Tensor:
    obs = episode.obs
    latents = []
    for start in range(0, len(episode), batch_size):
        stop = min(start + batch_size, len(episode))
        batch = obs[start:stop].to(vae.device)
        latents.append(vae.encode(batch).cpu())
    return torch.cat(latents, dim=0)


def count_missing_latents(dataset: Dataset) -> int:
    missing = 0
    for episode_id in range(dataset.num_episodes):
        episode = Episode.load(_episode_path(dataset._directory, episode_id))
        if not episode.has_latents:
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
        episode = Episode.load(episode_path)
        if episode.has_latents:
            continue
        episode.latents = encode_episode_obs(vae, episode, batch_size=batch_size)
        episode.save(episode_path)
        if dataset._cache_in_ram and episode_id in dataset._cache:
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
    total += encode_dataset_split(vae, train_dataset, batch_size, desc="Encoding train latents")
    total += encode_dataset_split(vae, test_dataset, batch_size, desc="Encoding test latents")
    return total
