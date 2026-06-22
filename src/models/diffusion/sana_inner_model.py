from dataclasses import dataclass
from typing import Optional

import torch
from torch import Tensor
import torch.nn as nn

try:
    from diffusers import SanaTransformer2DModel
except ImportError as exc:
    raise ImportError("diffusers is required for SANA integration. Install with: pip install diffusers>=0.33") from exc


@dataclass
class SanaInnerModelConfig:
    latent_channels: int
    num_steps_conditioning: int
    num_actions: Optional[int] = None
    pretrained_model_id: str = "Efficient-Large-Model/Sana_600M_512px_diffusers"
    load_pretrained: bool = True
    context_weight_scale: float = 0.1


class SanaInnerModel(nn.Module):
    """SANA transformer adapted for DIAMOND-style frame + action conditioning.

    Previous frame latents and the noisy target latent are channel-concatenated
    (DIAMOND-style). Actions are embedded as cross-attention tokens, replacing
    the original text conditioning from SANA.
    """

    def __init__(self, cfg: SanaInnerModelConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.in_channels = (cfg.num_steps_conditioning + 1) * cfg.latent_channels
        self.inner_dim = None

        if cfg.load_pretrained:
            self.transformer = self._build_from_pretrained(cfg)
        else:
            self.transformer = SanaTransformer2DModel.from_pretrained(
                cfg.pretrained_model_id, subfolder="transformer"
            )
            self._expand_patch_embed(cfg)

        self.inner_dim = self.transformer.config.num_attention_heads * self.transformer.config.attention_head_dim
        self.act_emb = nn.Embedding(cfg.num_actions, self.inner_dim)

    def _expand_patch_embed(self, cfg: SanaInnerModelConfig) -> None:
        pretrained_proj = self.transformer.patch_embed.proj
        if pretrained_proj.in_channels != self.in_channels:
            out_channels = pretrained_proj.out_channels
            kernel_size = pretrained_proj.kernel_size
            stride = pretrained_proj.stride
            padding = pretrained_proj.padding

            new_proj = nn.Conv2d(
                self.in_channels,
                out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                bias=pretrained_proj.bias is not None,
            )
            with torch.no_grad():
                pretrained_proj_w = pretrained_proj.weight
                new_proj_w = torch.zeros_like(new_proj.weight)
                new_proj_w[:, -cfg.latent_channels :, :, :] = pretrained_proj_w
                for i in range(cfg.num_steps_conditioning):
                    start = i * cfg.latent_channels
                    end = start + cfg.latent_channels
                    new_proj_w[:, start:end, :, :] = pretrained_proj_w * cfg.context_weight_scale
                new_proj.weight.copy_(new_proj_w)
                if pretrained_proj.bias is not None:
                    new_proj.bias.copy_(pretrained_proj.bias)

            self.transformer.patch_embed.proj = new_proj
            self.transformer.register_to_config(in_channels=self.in_channels)

        self._adapt_action_conditioning()
        if hasattr(self.transformer, "enable_gradient_checkpointing"):
            self.transformer.enable_gradient_checkpointing()

    def _adapt_action_conditioning(self) -> None:
        """Replace Gemma text projection (2304-dim) with identity for action tokens (inner_dim)."""
        if hasattr(self.transformer, "caption_projection"):
            self.transformer.caption_projection = nn.Identity()

    def _build_from_pretrained(self, cfg: SanaInnerModelConfig) -> SanaTransformer2DModel:
        transformer = SanaTransformer2DModel.from_pretrained(cfg.pretrained_model_id, subfolder="transformer")
        self.transformer = transformer
        self._expand_patch_embed(cfg)
        if hasattr(self.transformer, "enable_gradient_checkpointing"):
            self.transformer.enable_gradient_checkpointing()
        return transformer

    @property
    def device(self) -> torch.device:
        return next(self.transformer.parameters()).device

    def forward(
        self,
        noisy_next_latent: Tensor,
        timestep: Tensor,
        obs_latents: Tensor,
        act: Tensor,
    ) -> Tensor:
        """Predict flow velocity for the noisy next-frame latent.

        Args:
            noisy_next_latent: [B, C, H, W]
            timestep: [B] flow timestep in [0, 1]
            obs_latents: [B, T*C, H, W] channel-concatenated previous frame latents
            act: [B, T] previous actions
        """
        hidden_states = torch.cat((obs_latents, noisy_next_latent), dim=1)
        encoder_hidden_states = self.act_emb(act)

        if timestep.ndim == 0:
            timestep = timestep.unsqueeze(0).expand(hidden_states.size(0))
        timestep = timestep * 1000.0

        output = self.transformer(
            hidden_states=hidden_states,
            encoder_hidden_states=encoder_hidden_states,
            timestep=timestep,
            return_dict=True,
        )
        return output.sample
