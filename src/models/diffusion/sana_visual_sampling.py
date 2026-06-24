"""Generate visual rollout samples for SANA world-model training monitoring."""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import torch
from PIL import Image, ImageDraw

from agent import SanaAgent
from data.dataset import Dataset
from data.segment import SegmentId
from data.utils import make_segment
from models.diffusion import LatentSampler, LatentSamplerConfig


def _latent_to_uint8(vae, latent: torch.Tensor) -> np.ndarray:
    pixels = vae.decode(latent.unsqueeze(0)).squeeze(0)
    return pixels.add(1).div(2).mul(255).byte().permute(1, 2, 0).cpu().numpy()


def _make_labeled_row(images: List[np.ndarray], labels: List[str], label_height: int = 20) -> Image.Image:
    if not images:
        raise ValueError("No images for sample row.")
    h, w, _ = images[0].shape
    canvas = Image.new("RGB", (w * len(images), h + label_height), color=(0, 0, 0))
    draw = ImageDraw.Draw(canvas)
    for i, (img, label) in enumerate(zip(images, labels)):
        canvas.paste(Image.fromarray(img), (i * w, label_height))
        draw.text((i * w + 4, 2), label, fill=(255, 255, 255))
    return canvas


def _stack_rows(rows: List[Image.Image]) -> Image.Image:
    width = max(row.width for row in rows)
    height = sum(row.height for row in rows)
    canvas = Image.new("RGB", (width, height), color=(0, 0, 0))
    y = 0
    for row in rows:
        canvas.paste(row, (0, y))
        y += row.height
    return canvas


@torch.no_grad()
def generate_sana_rollout_panel(
    agent: SanaAgent,
    dataset: Dataset,
    sampler_cfg: LatentSamplerConfig,
    *,
    episode_id: int = 0,
    segment_start: int = 100,
    num_rollout_frames: int = 5,
    device: Optional[torch.device] = None,
) -> Tuple[Image.Image, dict]:
    """Roll out the world model from a fixed dataset segment and build a GT vs pred panel."""
    device = device or agent.device
    n_cond = agent.denoiser.cfg.inner_model.num_steps_conditioning
    seg_len = n_cond + num_rollout_frames + 1
    segment = make_segment(
        dataset.load_episode(episode_id),
        SegmentId(episode_id, segment_start, segment_start + seg_len),
        should_pad=False,
        use_latents=dataset._use_latents,
    )
    if not dataset._use_latents:
        raise RuntimeError("Visual sampling requires cached latents (use_latents=True).")

    obs = segment.obs.to(device)
    act = segment.act.to(device)
    latent_buffer = obs[:n_cond].unsqueeze(0)
    act_buffer = act[:n_cond].unsqueeze(0)

    sampler = LatentSampler(agent.denoiser, sampler_cfg)
    agent.vae.eval()
    agent.denoiser.eval()

    context = _latent_to_uint8(agent.vae, latent_buffer[0, -1])
    gt_frames: List[np.ndarray] = []
    pred_frames: List[np.ndarray] = []

    for step in range(num_rollout_frames):
        gt_idx = n_cond + step
        if gt_idx >= obs.size(0):
            break
        gt_frames.append(_latent_to_uint8(agent.vae, obs[gt_idx]))
        next_latent, _ = sampler.sample(latent_buffer, act_buffer)
        pred_frames.append(_latent_to_uint8(agent.vae, next_latent.squeeze(0)))

        latent_buffer = torch.cat([latent_buffer[:, 1:], next_latent.unsqueeze(1)], dim=1)
        if gt_idx + 1 < act.size(0):
            act_buffer = torch.cat([act_buffer[:, 1:], act[gt_idx].unsqueeze(0).unsqueeze(0)], dim=1)

    gt_labels = ["context"] + [f"gt {i + 1}" for i in range(len(gt_frames))]
    pred_labels = ["context"] + [f"pred {i + 1}" for i in range(len(pred_frames))]
    panel = _stack_rows(
        [
            _make_labeled_row([context] + gt_frames, gt_labels),
            _make_labeled_row([context] + pred_frames, pred_labels),
        ]
    )
    meta = {
        "episode_id": episode_id,
        "segment_start": segment_start,
        "num_rollout_frames": len(pred_frames),
    }
    return panel, meta


def save_sana_rollout_panel(panel: Image.Image, path: Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    panel.save(path)
    return path
