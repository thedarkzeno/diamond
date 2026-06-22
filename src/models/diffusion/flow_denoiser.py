from dataclasses import dataclass
from typing import List, Optional, Tuple, TYPE_CHECKING

import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F

from data import Batch
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


class FlowDenoiser(nn.Module):
    """Flow-matching denoiser operating in DC-AE latent space."""

    def __init__(self, cfg: FlowDenoiserConfig, vae: Optional["DCVAEWrapper"] = None) -> None:
        super().__init__()
        self.cfg = cfg
        object.__setattr__(self, "_vae", vae)
        self.inner_model = SanaInnerModel(cfg.inner_model)
        self._training_initialized = False

    @property
    def device(self) -> torch.device:
        return self.inner_model.device

    def setup_training(self, *_args, **_kwargs) -> None:
        """No-op for compatibility with the DIAMOND trainer interface."""
        self._training_initialized = True

    def sample_timestep(self, batch_size: int, device: torch.device) -> Tensor:
        # Logit-normal sampling biased toward higher noise levels (flow matching).
        u = torch.randn(batch_size, device=device)
        t = torch.sigmoid(u)
        if self.cfg.flow_shift != 1.0:
            t = t ** (1.0 / self.cfg.flow_shift)
        return t.clamp(1e-4, 1.0 - 1e-4)

    def apply_noise(self, z0: Tensor, t: Tensor) -> Tuple[Tensor, Tensor]:
        eps = torch.randn_like(z0)
        if self.cfg.sigma_offset_noise > 0:
            offset = self.cfg.sigma_offset_noise * torch.randn(z0.size(0), z0.size(1), 1, 1, device=z0.device)
            eps = eps + offset
        t_spatial = add_dims(t, z0.ndim)
        z_t = (1.0 - t_spatial) * z0 + t_spatial * eps
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

    def denoise(self, z_t: Tensor, t: Tensor, obs_latents: Tensor, act: Tensor) -> Tensor:
        velocity = self.inner_model(z_t, t, obs_latents, act)
        t_spatial = add_dims(t, z_t.ndim)
        return z_t - t_spatial * velocity

    @torch.no_grad()
    def wrap_model_output(self, z_t: Tensor, velocity: Tensor, t: Tensor) -> Tensor:
        z0 = z_t - add_dims(t, z_t.ndim) * velocity
        return z0

    @torch.no_grad()
    def denoise_velocity(self, z_t: Tensor, t: Tensor, obs_latents: Tensor, act: Tensor) -> Tensor:
        return self.inner_model(z_t, t, obs_latents, act)

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

                t = self.sample_timestep(next_obs.size(0), self.device)
                z_t, eps = self.apply_noise(next_obs, t)
                velocity = self.inner_model(z_t, t, obs_latents, act)
                target = self.compute_velocity_target(next_obs, eps)
                loss = loss + F.mse_loss(velocity[mask], target[mask])

                with torch.no_grad():
                    denoised = self.wrap_model_output(z_t, velocity, t)
                all_obs[:, n + i] = denoised

            loss = loss / seq_length
        return loss, {"loss_denoising": loss.detach()}
