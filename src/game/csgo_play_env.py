from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Tuple

import torch
from torch import Tensor

from agent import Agent, SanaAgent
from csgo.action_processing import CSGOAction, decode_csgo_action, encode_csgo_action, print_csgo_action
from data import Dataset, Episode
from envs import WorldModelEnv


class CsgoPlayEnv:
    """Human-in-the-loop CS:GO world model playground."""

    def __init__(
        self,
        agent: Agent | SanaAgent,
        wm_env: WorldModelEnv,
        recording_mode: bool = False,
    ) -> None:
        self.agent = agent
        self.env = wm_env
        self.recording_mode = recording_mode
        self.is_human_player = True
        self.env_name = "world model"
        self.obs = None
        self.t = 0
        self.buffer = None
        self.rec_dataset = None

    def print_controls(self) -> None:
        from csgo.keymap import CSGO_KEYMAP
        import pygame

        print("\nControls (CS:GO world model):\n")
        print("m       : switch human / replay actions from dataset")
        print("↑ / ↓   : imagination horizon")
        print("Esc     : quit")
        print("\nKeyboard:\n")
        for key, action_name in CSGO_KEYMAP.items():
            key_name = pygame.key.name(key)
            key_name = "⎵" if key_name == "space" else key_name
            print(f"{key_name} : {action_name}")
        print("\nMouse: move to look, LMB/RMB to shoot.")

    def next_mode(self) -> bool:
        self.is_human_player = not self.is_human_player
        print(f"> Control: {'human' if self.is_human_player else 'replay actions'}")
        return True

    def next_axis_1(self) -> bool:
        self.env.horizon = max(1, self.env.horizon + 1)
        print(f"> Horizon: {self.env.horizon}")
        return True

    def prev_axis_1(self) -> bool:
        self.env.horizon = max(1, self.env.horizon - 1)
        print(f"> Horizon: {self.env.horizon}")
        return True

    def next_axis_2(self) -> bool:
        return False

    def prev_axis_2(self) -> bool:
        return False

    def reset_recording(self) -> None:
        self.buffer = defaultdict(list)
        dir = Path("dataset") / "rec_csgo_wm"
        self.rec_dataset = Dataset(dir, None)
        self.rec_dataset.load_from_default_path()

    def reset(self) -> Tuple[Tensor, None]:
        self.obs, _ = self.env.reset()
        self.t = 0
        if self.recording_mode:
            self.reset_recording()
        return self.obs, None

    @torch.no_grad()
    def step(self, csgo_action: CSGOAction) -> Tuple[Tensor, Tensor, Tensor, Tensor, Dict[str, Any]]:
        if self.is_human_player:
            action = encode_csgo_action(csgo_action, device=self.agent.device)
        else:
            action = self.env.act_buffer[0, -1].clone()
            csgo_action = decode_csgo_action(action.cpu())

        next_obs, rew, end, trunc, env_info = self.env.step(action)
        keys, mouse, clicks = print_csgo_action(csgo_action)

        header = [
            [
                f"Env     : {self.env_name}",
                f"Control : {'human' if self.is_human_player else 'replay'}",
                f"Timestep: {self.t + 1}",
                f"Horizon : {self.env.horizon}",
            ],
            [
                f"Keys  : {keys}",
                f"Mouse : {mouse}",
                f"Clicks: {clicks}",
            ],
        ]
        info = {"header": header}

        if self.recording_mode:
            self.buffer["obs"].append(self.obs.cpu())
            self.buffer["act"].append(action.cpu())
            self.buffer["rew"].append(rew.cpu())
            self.buffer["end"].append(end.cpu())
            self.buffer["trunc"].append(trunc.cpu())
            if end or trunc:
                ep = Episode(
                    obs=torch.cat(self.buffer["obs"], dim=0),
                    act=torch.stack(self.buffer["act"], dim=0),
                    rew=torch.cat(self.buffer["rew"], dim=0),
                    end=torch.cat(self.buffer["end"], dim=0),
                    trunc=torch.cat(self.buffer["trunc"], dim=0),
                    info={},
                )
                self.rec_dataset.add_episode(ep)
                self.rec_dataset.save_to_default_path()
                self.reset_recording()

        self.obs = next_obs
        self.t += 1
        return next_obs, rew, end, trunc, info
