from dataclasses import dataclass
from typing import List, Optional, Tuple, TYPE_CHECKING

import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F

from data import Batch
from .sana_flow_utils import load_flow_scheduler, sample_flow_training_timesteps, sigmas_from_timesteps
from .sana_inner_model import SanaInnerModel, SanaInnerModelConfig
from utils import LossAndLogs

if TYPE_CHECKING:
    from models.vae_wrapper import DCVAEWrapper


def add_dims(input: Tensor, n: int) -> Tensor:
    return input.reshape(input.shape + (1,) * (n - input.ndim))


@dataclass
class FlowDenoiserConfig:
    inner_model: SanaInnerModelConfig
    flow_shift: float = 3.0
    sigma_offset_noise: float = 0.0
    use_cached_latents: bool = False
    use_amp: bool = True
    weighting_scheme: str = "none"


class FlowDenoiser(nn.Module):
    """Flow-matching denoiser operating in DC-AE latent space."""

    def __init__(self, cfg: FlowDenoiserConfig, vae: Optional["DCVAEWrapper"] = None) -> None:
        super().__init__()
        self.cfg = cfg
        object.__setattr__(self, "_vae", vae)
        self.inner_model = SanaInnerModel(cfg.inner_model)
        self._noise_scheduler, self._noise_scheduler_train = load_flow_scheduler(cfg.inner_model.pretrained_model_id)
        self._training_initialized = False

    @property
    def device(self) -> torch.device:
        return self.inner_model.device

    def setup_training(self, *_args, **_kwargs) -> None:
        """No-op for compatibility with the DIAMOND trainer interface."""
        self._training_initialized = True

    def apply_noise(self, z0: Tensor, sigmas: Tensor) -> Tuple[Tensor, Tensor]:
        eps = torch.randn_like(z0)
        if self.cfg.sigma_offset_noise > 0:
            offset = self.cfg.sigma_offset_noise * torch.randn(z0.size(0), z0.size(1), 1, 1, device=z0.device)
            eps = eps + offset
        sigmas = add_dims(sigmas, z0.ndim)
        z_t = (1.0 - sigmas) * z0 + sigmas * eps
        return z_t, eps

    def compute_velocity_target(self, z0: Tensor, eps: Tensor) -> Tensor:
        return eps - z0

    def _obs_to_latents(self, obs: Tensor) -> Tensor:
        if self.cfg.use_cached_latents:
            return obs
        if self._vae is None:
            raise RuntimeError("VAE is required when use_cached_latents=False")
        return self._vae.encode_batch_obs(obs)

    def _prepare_obs_latents(self, obs: Tensor) -> Tensor:
        b, t, c, h, w = obs.shape
        return obs.reshape(b, t * c, h, w)

    def denoise(self, z_t: Tensor, timesteps: Tensor, obs_latents: Tensor, act: Tensor) -> Tensor:
        velocity = self.inner_model(z_t, timesteps, obs_latents, act)
        sigmas = sigmas_from_timesteps(self._noise_scheduler, timesteps, z_t.ndim, z_t.dtype)
        return z_t - add_dims(sigmas, z_t.ndim) * velocity

    @torch.no_grad()
    def wrap_model_output(self, z_t: Tensor, velocity: Tensor, sigmas: Tensor) -> Tensor:
        return z_t - add_dims(sigmas, z_t.ndim) * velocity

    @torch.no_grad()
    def denoise_velocity(self, z_t: Tensor, timesteps: Tensor, obs_latents: Tensor, act: Tensor) -> Tensor:
        return self.inner_model(z_t, timesteps, obs_latents, act)

    def forward(self, batch: Batch) -> LossAndLogs:
        n = self.cfg.inner_model.num_steps_conditioning
        seq_length = batch.obs.size(1) - n
        use_amp = self.cfg.use_amp and self.device.type == "cuda"

        with torch.autocast(device_type=self.device.type, dtype=torch.bfloat16, enabled=use_amp):
            all_obs = self._obs_to_latents(batch.obs)
            loss = 0.0

            for i in range(seq_length):
                obs = all_obs[:, i : n + i]
                next_obs = all_obs[:, n + i]
                act = batch.act[:, i : n + i]
                mask = batch.mask_padding[:, n + i]

                obs_latents = self._prepare_obs_latents(obs)

                timesteps, sigmas = sample_flow_training_timesteps(
                    self._noise_scheduler_train,
                    next_obs.size(0),
                    self.device,
                    next_obs.dtype,
                    self.cfg.weighting_scheme,
                )
                z_t, eps = self.apply_noise(next_obs, sigmas)
                velocity = self.inner_model(z_t, timesteps, obs_latents, act)
                target = self.compute_velocity_target(next_obs, eps)
                loss = loss + F.mse_loss(velocity[mask], target[mask])

                with torch.no_grad():
                    denoised = self.wrap_model_output(z_t, velocity, sigmas)
                all_obs[:, n + i] = denoised

            loss = loss / seq_length
        return loss, {"loss_denoising": loss.detach()}
