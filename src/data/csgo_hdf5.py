from pathlib import Path
from typing import Dict

import h5py
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset as TorchDataset

from .episode import Episode
from .segment import Segment, SegmentId
from utils import StateDictMixin

# CS:GO action vector layout (TeaPearce behavioural cloning dataset).
N_KEYS = 11
N_CLICKS = 2
N_MOUSE_X = 22
N_MOUSE_Y = 15
ACTION_DIM = N_KEYS + N_CLICKS + N_MOUSE_X + N_MOUSE_Y


def hdf5_action_to_vector(y: np.ndarray) -> torch.Tensor:
    """Convert raw HDF5 frame_y vector to rounded multi-hot float tensor."""
    y = np.asarray(y, dtype=np.float32).reshape(-1)
    if y.size < ACTION_DIM:
        padded = np.zeros(ACTION_DIM, dtype=np.float32)
        padded[: y.size] = y
        y = padded
    else:
        y = y[:ACTION_DIM]
    return torch.from_numpy(np.round(y).clip(0, 1))


class CSGOHdf5Dataset(StateDictMixin, TorchDataset):
    """Read CS:GO episodes from extracted HDF5 files (1000 frames each)."""

    def __init__(self, directory: Path) -> None:
        super().__init__()
        filenames = sorted(Path(directory).rglob("*.hdf5"), key=lambda x: int(x.stem.split("_")[-1]))
        self._filenames = [x for x in filenames]
        self._length_one_episode = 1000
        self.num_episodes = len(self._filenames)
        self.num_steps = self._length_one_episode * self.num_episodes
        self.lengths = np.array([self._length_one_episode] * self.num_episodes, dtype=np.int64)

    def __len__(self) -> int:
        return self.num_steps

    def save_to_default_path(self) -> None:
        pass

    def __getitem__(self, segment_id: SegmentId) -> Segment:
        assert segment_id.start < self._length_one_episode and segment_id.stop > 0 and segment_id.start < segment_id.stop
        pad_len_right = max(0, segment_id.stop - self._length_one_episode)
        pad_len_left = max(0, -segment_id.start)

        start = max(0, segment_id.start)
        stop = min(self._length_one_episode, segment_id.stop)
        mask_padding = torch.cat((torch.zeros(pad_len_left), torch.ones(stop - start), torch.zeros(pad_len_right))).bool()

        path = self._filenames[int(segment_id.episode_id)]
        with h5py.File(path, "r") as f:
            obs = torch.stack(
                [
                    torch.tensor(f[f"frame_{i}_x"][:]).flip(2).permute(2, 0, 1).div(255).mul(2).sub(1)
                    for i in range(start, stop)
                ]
            )
            act = torch.stack([hdf5_action_to_vector(f[f"frame_{i}_y"][:]) for i in range(start, stop)])

        def pad_obs(x: torch.Tensor) -> torch.Tensor:
            right = F.pad(x, [0, 0, 0, 0, 0, pad_len_right]) if pad_len_right > 0 else x
            return F.pad(right, [0, 0, 0, 0, pad_len_left, 0]) if pad_len_left > 0 else right

        def pad_act(x: torch.Tensor) -> torch.Tensor:
            right = F.pad(x, [0, 0, pad_len_right]) if pad_len_right > 0 else x
            return F.pad(right, [0, pad_len_left, 0]) if pad_len_left > 0 else right

        obs = pad_obs(obs)
        act = pad_act(act)
        rew = torch.zeros(obs.size(0))
        end = torch.zeros(obs.size(0), dtype=torch.uint8)
        trunc = torch.zeros(obs.size(0), dtype=torch.uint8)
        return Segment(obs, act, rew, end, trunc, mask_padding, info={}, id=SegmentId(segment_id.episode_id, start, stop))

    def load_episode(self, episode_id: int) -> Episode:
        s = self[SegmentId(episode_id, 0, self._length_one_episode)]
        return Episode(s.obs, s.act, s.rew, s.end, s.trunc, s.info)

    @property
    def filename_by_episode_id(self) -> Dict[int, Path]:
        return {i: p for i, p in enumerate(self._filenames)}
