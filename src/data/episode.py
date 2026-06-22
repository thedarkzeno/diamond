from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

import torch


@dataclass
class Episode:
    obs: torch.FloatTensor
    act: torch.Tensor
    rew: torch.FloatTensor
    end: torch.ByteTensor
    trunc: torch.ByteTensor
    info: Dict[str, Any]
    latents: Optional[torch.FloatTensor] = None

    def __len__(self) -> int:
        return self.obs.size(0)

    def __add__(self, other: Episode) -> Episode:
        assert self.dead.sum() == 0
        d = {k: torch.cat((v, other.__dict__[k]), dim=0) for k, v in self.__dict__.items() if k not in ("info", "latents")}
        latents = None
        if self.latents is not None and other.latents is not None:
            latents = torch.cat((self.latents, other.latents), dim=0)
        return Episode(**d, info=merge_info(self.info, other.info), latents=latents)

    def to(self, device) -> Episode:
        return Episode(
            **{
                k: v.to(device) if k != "info" and v is not None else v
                for k, v in self.__dict__.items()
            }
        )

    @property
    def dead(self) -> torch.ByteTensor:
        return (self.end + self.trunc).clip(max=1)

    def compute_metrics(self) -> Dict[str, Any]:
        return {"length": len(self), "return": self.rew.sum().item()}

    @property
    def has_latents(self) -> bool:
        return self.latents is not None

    @classmethod
    def load(cls, path: Path, map_location: Optional[torch.device] = None) -> Episode:
        data = torch.load(Path(path), map_location=map_location)
        obs = data["obs"].div(255).mul(2).sub(1)
        latents = data.get("latents")
        if latents is not None:
            latents = latents.float()
        return cls(
            obs=obs,
            act=data["act"],
            rew=data["rew"],
            end=data["end"],
            trunc=data["trunc"],
            info=data["info"],
            latents=latents,
        )

    def save(self, path: Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        d = {k: v.add(1).div(2).mul(255).byte() if k == "obs" else v for k, v in self.__dict__.items() if v is not None}
        torch.save(d, path.with_suffix(".tmp"))
        path.with_suffix(".tmp").rename(path)


def merge_info(info_a, info_b):
    keys_a = set(info_a)
    keys_b = set(info_b)
    intersection = keys_a & keys_b
    info = {
        **{k: info_a[k] for k in keys_a if k not in intersection},
        **{k: info_b[k] for k in keys_b if k not in intersection},
        **{k: torch.cat((info_a[k], info_b[k]), dim=0) for k in intersection},
    }
    return info
