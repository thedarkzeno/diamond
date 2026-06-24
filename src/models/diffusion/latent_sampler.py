from dataclasses import dataclass
from typing import List, Tuple

import torch
from torch import Tensor

try:
    from diffusers import DPMSolverMultistepScheduler
except ImportError as exc:
    raise ImportError("diffusers is required for SANA integration. Install with: pip install diffusers>=0.33") from exc

from .flow_denoiser import FlowDenoiser


@dataclass
class LatentSamplerConfig:
    num_steps_denoising: int = 20
    flow_shift: float = 3.0
    num_train_timesteps: int = 1000
    store_trajectory: bool = False
    use_amp: bool = True


class LatentSampler:
    """Latent-space sampler using DPM-Solver++ with flow prediction."""

    def __init__(self, denoiser: FlowDenoiser, cfg: LatentSamplerConfig) -> None:
        self.denoiser = denoiser
        self.cfg = cfg
        self.scheduler = DPMSolverMultistepScheduler(
            num_train_timesteps=cfg.num_train_timesteps,
            prediction_type="flow_prediction",
            use_flow_sigmas=True,
            flow_shift=cfg.flow_shift,
            algorithm_type="dpmsolver++",
            solver_order=2,
            lower_order_final=True,
        )

    @property
    def device(self) -> torch.device:
        return self.denoiser.device

    @torch.no_grad()
    def sample(self, prev_obs: Tensor, prev_act: Tensor) -> Tuple[Tensor, List[Tensor]]:
        device = prev_obs.device
        b, t, c, h, w = prev_obs.size()
        obs_latents = prev_obs.reshape(b, t * c, h, w)
        latent_channels = self.denoiser.cfg.inner_model.latent_channels

        latents = torch.randn(b, latent_channels, h, w, device=device)
        trajectory: List[Tensor] = []
        if self.cfg.store_trajectory:
            trajectory.append(latents.clone())

        self.scheduler.set_timesteps(self.cfg.num_steps_denoising, device=device)
        use_amp = self.cfg.use_amp and device.type == "cuda"
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=use_amp):
            for timestep in self.scheduler.timesteps:
                timestep_batch = timestep.expand(b).to(device=latents.device)
                model_output = self.denoiser.denoise_velocity(latents, timestep_batch, obs_latents, prev_act)
                latents = self.scheduler.step(model_output, timestep, latents).prev_sample
                if self.cfg.store_trajectory:
                    trajectory.append(latents.clone())

        return latents, trajectory
