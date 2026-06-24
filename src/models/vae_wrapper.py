from dataclasses import dataclass
from typing import Optional

import torch
from torch import Tensor
import torch.nn as nn

try:
    from diffusers import AutoencoderDC
except ImportError as exc:
    raise ImportError("diffusers is required for SANA integration. Install with: pip install diffusers>=0.33") from exc


@dataclass
class VAEConfig:
    model_id: str = "Efficient-Large-Model/Sana_600M_512px_diffusers"
    freeze: bool = True
    enable_tiling: bool = True
    encode_micro_batch: int = 4
    device: str = "cuda"  # cuda | cpu | auto


class DCVAEWrapper(nn.Module):
    """Thin wrapper around SANA's DC-AE (AutoencoderDC).

    Encode: [B, 3, H, W] pixels in [-1, 1] -> [B, 32, H/32, W/32] latents.
    Decode: latents -> pixels in [-1, 1].
    """

    def __init__(self, cfg: VAEConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.vae = AutoencoderDC.from_pretrained(cfg.model_id, subfolder="vae")
        self.vae.to(torch.float32)
        if cfg.enable_tiling:
            self.vae.enable_tiling()
        if cfg.freeze:
            self.vae.requires_grad_(False)
            self.vae.eval()

    def apply_device_policy(self, agent_device: torch.device) -> None:
        """Place the VAE according to config after the agent is moved to the train device."""
        if self.cfg.device == "cpu":
            self.cpu()
        else:
            self.to(agent_device)

    @property
    def device(self) -> torch.device:
        return next(self.vae.parameters()).device

    @property
    def latent_channels(self) -> int:
        return self.vae.config.latent_channels

    @property
    def scaling_factor(self) -> float:
        return self.vae.config.scaling_factor

    @property
    def spatial_compression_ratio(self) -> int:
        return 2 ** (len(self.vae.config.encoder_block_out_channels) - 1)

    @torch.no_grad()
    def encode(self, x: Tensor, output_device: Optional[torch.device] = None) -> Tensor:
        """Encode pixels to latent space."""
        if self.cfg.freeze:
            self.vae.eval()
        x = x.to(self.device, dtype=torch.float32)
        encoded = self.vae.encode(x).latent
        encoded = encoded * self.scaling_factor
        if output_device is not None:
            encoded = encoded.to(output_device)
        return encoded

    @torch.no_grad()
    def decode(self, z: Tensor, output_device: Optional[torch.device] = None) -> Tensor:
        """Decode latents to pixels."""
        if self.cfg.freeze:
            self.vae.eval()
        z = z.to(self.device) / self.scaling_factor
        decoded = self.vae.decode(z).sample.clamp(-1, 1)
        if output_device is not None:
            decoded = decoded.to(output_device)
        return decoded

    def _encode_chunked(self, x: Tensor, output_device: torch.device) -> Tensor:
        micro_batch = max(1, self.cfg.encode_micro_batch)
        chunks = []
        for start in range(0, x.size(0), micro_batch):
            chunk = x[start : start + micro_batch]
            chunks.append(self.encode(chunk, output_device=output_device))
        return torch.cat(chunks, dim=0)

    def encode_batch_obs(self, obs: Tensor) -> Tensor:
        """Encode observation sequences.

        Args:
            obs: [B, T, C, H, W] or [B, C, H, W] in [-1, 1].

        Returns:
            Latents with same batch/time layout and spatial dims divided by compression ratio.
        """
        output_device = obs.device
        if obs.ndim == 5:
            b, t, c, h, w = obs.shape
            if self.device.type == "cpu":
                frames = []
                for frame_idx in range(t):
                    frames.append(self._encode_chunked(obs[:, frame_idx], output_device))
                return torch.stack(frames, dim=1)
            flat = obs.reshape(b * t, c, h, w)
            latents = self._encode_chunked(flat, output_device)
            _, lc, lh, lw = latents.shape
            return latents.reshape(b, t, lc, lh, lw)
        return self._encode_chunked(obs, output_device)

    def decode_batch_obs(self, latents: Tensor) -> Tensor:
        """Decode latents back to pixels."""
        output_device = latents.device
        if latents.ndim == 5:
            b, t, c, h, w = latents.shape
            frames = []
            for frame_idx in range(t):
                frame = latents[:, frame_idx]
                frames.append(self.decode(frame, output_device=output_device))
            return torch.stack(frames, dim=1)
        return self.decode(latents, output_device=output_device)
