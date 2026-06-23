"""CS:GO multi-hot action layout (TeaPearce behavioural cloning dataset).

Single source of truth for training data, world model, and play encoding.
"""

from __future__ import annotations

import torch

N_KEYS = 11
N_CLICKS = 2
N_MOUSE_X = 22
N_MOUSE_Y = 15
ACTION_DIM = N_KEYS + N_CLICKS + N_MOUSE_X + N_MOUSE_Y

MOUSE_X_POSSIBLES = [
    -1000, -500, -300, -200, -100, -60, -30, -20, -10, -4, -2, 0,
    2, 4, 10, 20, 30, 60, 100, 200, 300, 500,
]

MOUSE_Y_POSSIBLES = [
    -200, -100, -50, -20, -10, -4, -2, 0, 2, 4, 10, 20, 50, 100, 200,
]

assert len(MOUSE_X_POSSIBLES) == N_MOUSE_X
assert len(MOUSE_Y_POSSIBLES) == N_MOUSE_Y

MOUSE_X_LIM = (MOUSE_X_POSSIBLES[0], MOUSE_X_POSSIBLES[-1])
MOUSE_Y_LIM = (MOUSE_Y_POSSIBLES[0], MOUSE_Y_POSSIBLES[-1])


def normalize_csgo_action(act: torch.Tensor, action_dim: int = ACTION_DIM) -> torch.Tensor:
    """Pad or truncate action vectors to the canonical multi-hot size."""
    d = act.size(-1)
    if d == action_dim:
        return act
    if d > action_dim:
        return act[..., :action_dim].contiguous()
    pad = act.new_zeros(*act.shape[:-1], action_dim - d)
    return torch.cat([act, pad], dim=-1)
