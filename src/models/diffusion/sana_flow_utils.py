"""Flow-matching helpers aligned with HuggingFace SANA DreamBooth training."""

from __future__ import annotations

import copy
from typing import Tuple

import torch
from torch import Tensor

try:
    from diffusers import FlowMatchEulerDiscreteScheduler
    from diffusers.training_utils import compute_density_for_timestep_sampling
except ImportError as exc:
    raise ImportError("diffusers is required for SANA integration. Install with: pip install diffusers>=0.33") from exc


def load_flow_scheduler(pretrained_model_id: str) -> Tuple[FlowMatchEulerDiscreteScheduler, FlowMatchEulerDiscreteScheduler]:
    """Return (live scheduler, frozen copy for training noise sampling)."""
    scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(pretrained_model_id, subfolder="scheduler")
    return scheduler, copy.deepcopy(scheduler)


def sigmas_from_timesteps(
    scheduler: FlowMatchEulerDiscreteScheduler,
    timesteps: Tensor,
    n_dim: int,
    dtype: torch.dtype,
) -> Tensor:
    sigmas = scheduler.sigmas.to(device=timesteps.device, dtype=dtype)
    schedule_timesteps = scheduler.timesteps.to(timesteps.device)
    step_indices = [(schedule_timesteps == t).nonzero().item() for t in timesteps]
    sigma = sigmas[step_indices].flatten()
    while len(sigma.shape) < n_dim:
        sigma = sigma.unsqueeze(-1)
    return sigma


def sample_flow_training_timesteps(
    scheduler: FlowMatchEulerDiscreteScheduler,
    batch_size: int,
    device: torch.device,
    dtype: torch.dtype,
    weighting_scheme: str = "none",
) -> Tuple[Tensor, Tensor]:
    u = compute_density_for_timestep_sampling(
        weighting_scheme=weighting_scheme,
        batch_size=batch_size,
        device=device,
    )
    indices = (u * scheduler.config.num_train_timesteps).long().cpu()
    timesteps = scheduler.timesteps[indices].to(device=device)
    sigmas = sigmas_from_timesteps(scheduler, timesteps, n_dim=4, dtype=dtype)
    return timesteps, sigmas
