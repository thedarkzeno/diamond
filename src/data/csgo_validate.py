"""Validate processed CS:GO datasets before training or play."""

from __future__ import annotations

from typing import Optional

import numpy as np
import torch

from csgo.action_layout import ACTION_DIM, normalize_csgo_action
from data.dataset import Dataset
from data.episode import Episode


def validate_csgo_episode_actions(episode: Episode, expected_dim: int = ACTION_DIM) -> None:
    act = episode.act
    if act.size(-1) != expected_dim:
        raise ValueError(
            f"Action dim {act.size(-1)} != expected {expected_dim}. "
            "Re-run process_data.sh or upgrade episodes with scripts/validate_csgo_dataset.py --fix."
        )


def validate_csgo_dataset(
    dataset: Dataset,
    expected_dim: int = ACTION_DIM,
    *,
    num_samples: Optional[int] = 32,
    require_latents: bool = False,
) -> None:
    if dataset.num_episodes == 0:
        raise ValueError(f"Dataset '{dataset.name}' has no episodes.")

    if num_samples is None or num_samples >= dataset.num_episodes:
        episode_ids = np.arange(dataset.num_episodes)
    else:
        episode_ids = np.random.default_rng(0).choice(dataset.num_episodes, size=num_samples, replace=False)

    bad_dims = []
    missing_latents = []
    for episode_id in episode_ids:
        episode = dataset.load_episode(int(episode_id))
        if episode.act.size(-1) != expected_dim:
            bad_dims.append((int(episode_id), int(episode.act.size(-1))))
        if require_latents and not episode.has_latents:
            missing_latents.append(int(episode_id))

    if bad_dims:
        examples = ", ".join(f"ep {eid}={dim}d" for eid, dim in bad_dims[:5])
        raise ValueError(
            f"Dataset '{dataset.name}': {len(bad_dims)} episode(s) with wrong action dim "
            f"(expected {expected_dim}). Examples: {examples}"
        )

    if missing_latents:
        examples = ", ".join(str(eid) for eid in missing_latents[:5])
        raise ValueError(
            f"Dataset '{dataset.name}': {len(missing_latents)} episode(s) missing latent sidecars. "
            f"Run: python scripts/encode_dataset.py --dataset-dir <path>. Examples: {examples}"
        )


def fix_csgo_dataset_actions(dataset: Dataset, expected_dim: int = ACTION_DIM) -> int:
    """Truncate/pad stored action tensors in all episodes. Returns number of episodes fixed."""
    fixed = 0
    for episode_id in range(dataset.num_episodes):
        path = dataset._get_episode_path(episode_id)
        episode = Episode.load(path)
        normalized = normalize_csgo_action(episode.act, expected_dim)
        if episode.act.size(-1) == normalized.size(-1) and torch.equal(episode.act, normalized):
            continue
        episode.act = normalized
        episode.save(path)
        if Episode._latents_sidecar_path(path).is_file() and episode.has_latents:
            Episode.save_latent_training_bundle(path, episode)
        fixed += 1
    return fixed
