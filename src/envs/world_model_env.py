from dataclasses import dataclass
from typing import Any, Dict, Generator, List, Optional, Tuple, Union

import torch
from torch import Tensor
from torch.distributions.categorical import Categorical
from torch.utils.data import DataLoader

from coroutines import coroutine
from models.diffusion import Denoiser, DiffusionSampler, DiffusionSamplerConfig
from models.rew_end_model import RewEndModel
from models.vae_wrapper import DCVAEWrapper
from utils import resize_obs

ResetOutput = Tuple[torch.FloatTensor, Dict[str, Any]]
StepOutput = Tuple[Tensor, Tensor, Tensor, Tensor, Dict[str, Any]]
InitialCondition = Tuple[Tensor, Tensor, Tuple[Tensor, Tensor]]


@dataclass
class WorldModelEnvConfig:
    horizon: int
    num_batches_to_preload: int
    diffusion_sampler: Optional[DiffusionSamplerConfig] = None
    latent_sampler: Optional[Any] = None
    rl_img_size: Optional[int] = None


class WorldModelEnv:
    def __init__(
        self,
        denoiser: Union[Denoiser, Any],
        rew_end_model: RewEndModel,
        data_loader: DataLoader,
        cfg: WorldModelEnvConfig,
        return_denoising_trajectory: bool = False,
        vae: Optional[DCVAEWrapper] = None,
    ) -> None:
        self.denoiser = denoiser
        self.vae = vae
        self.use_latents = vae is not None

        if self.use_latents:
            from models.diffusion import FlowDenoiser, LatentSampler

            assert isinstance(denoiser, FlowDenoiser)
            assert cfg.latent_sampler is not None
            self.sampler = LatentSampler(denoiser, cfg.latent_sampler)
            self.sampler.cfg.store_trajectory = return_denoising_trajectory
        else:
            assert cfg.diffusion_sampler is not None
            self.sampler = DiffusionSampler(denoiser, cfg.diffusion_sampler)

        self.rew_end_model = rew_end_model
        self.horizon = cfg.horizon
        self.rl_img_size = cfg.rl_img_size
        self.return_denoising_trajectory = return_denoising_trajectory
        self.num_envs = data_loader.batch_sampler.batch_size
        self.generator_init = self.make_generator_init(data_loader, cfg.num_batches_to_preload)
        self.latent_buffer: Optional[Tensor] = None

    @property
    def device(self) -> torch.device:
        if self.use_latents:
            return self.denoiser.device
        return self.sampler.denoiser.device

    def _encode_obs(self, obs: Tensor) -> Tensor:
        assert self.vae is not None
        return self.vae.encode_batch_obs(obs)

    def _pixels_for_rl(self, pixels: Tensor) -> Tensor:
        if self.rl_img_size is None:
            return pixels
        return resize_obs(pixels, self.rl_img_size)

    def _decode_obs(self, latents: Tensor) -> Tensor:
        assert self.vae is not None
        if latents.ndim == 5:
            b, t, c, h, w = latents.shape
            flat = latents.reshape(b * t, c, h, w)
            pixels = self.vae.decode(flat)
            _, pc, ph, pw = pixels.shape
            pixels = pixels.reshape(b, t, pc, ph, pw)
        else:
            pixels = self.vae.decode(latents)
        return self._pixels_for_rl(pixels)

    @torch.no_grad()
    def reset(self, **kwargs) -> ResetOutput:
        obs, act, (hx, cx) = self.generator_init.send(self.num_envs)
        self.obs_buffer = self._pixels_for_rl(obs)
        self.act_buffer = act
        if self.use_latents:
            self.latent_buffer = self._encode_obs(obs)
        self.hx_rew_end = hx
        self.cx_rew_end = cx
        self.ep_len = torch.zeros(self.num_envs, dtype=torch.long, device=obs.device)
        return self.obs_buffer[:, -1], {}

    @torch.no_grad()
    def reset_dead(self, dead: torch.BoolTensor) -> None:
        obs, act, (hx, cx) = self.generator_init.send(dead.sum().item())
        self.obs_buffer[dead] = self._pixels_for_rl(obs)
        self.act_buffer[dead] = act
        if self.use_latents and self.latent_buffer is not None:
            self.latent_buffer[dead] = self._encode_obs(obs)
        self.hx_rew_end[:, dead] = hx
        self.cx_rew_end[:, dead] = cx
        self.ep_len[dead] = 0

    @torch.no_grad()
    def step(self, act: torch.LongTensor) -> StepOutput:
        self.act_buffer[:, -1] = act

        next_latent, denoising_trajectory = self.predict_next_obs()
        if self.use_latents:
            next_obs = self._decode_obs(next_latent)
        else:
            next_obs = next_latent

        rew, end = self.predict_rew_end(next_obs.unsqueeze(1))

        self.ep_len += 1
        trunc = (self.ep_len >= self.horizon).long()

        self.obs_buffer = self.obs_buffer.roll(-1, dims=1)
        self.act_buffer = self.act_buffer.roll(-1, dims=1)
        self.obs_buffer[:, -1] = next_obs
        if self.use_latents and self.latent_buffer is not None:
            self.latent_buffer = self.latent_buffer.roll(-1, dims=1)
            self.latent_buffer[:, -1] = next_latent

        dead = torch.logical_or(end, trunc)

        info = {}
        if self.return_denoising_trajectory:
            info["denoising_trajectory"] = torch.stack(denoising_trajectory, dim=1)

        if dead.any():
            self.reset_dead(dead)
            info["final_observation"] = next_obs[dead]
            info["burnin_obs"] = self.obs_buffer[dead, :-1]

        return self.obs_buffer[:, -1], rew, end, trunc, info

    @torch.no_grad()
    def predict_next_obs(self) -> Tuple[Tensor, List[Tensor]]:
        if self.use_latents:
            assert self.latent_buffer is not None
            return self.sampler.sample(self.latent_buffer, self.act_buffer)
        return self.sampler.sample(self.obs_buffer, self.act_buffer)

    @torch.no_grad()
    def predict_rew_end(self, next_obs: Tensor) -> Tuple[Tensor, Tensor]:
        logits_rew, logits_end, (self.hx_rew_end, self.cx_rew_end) = self.rew_end_model.predict_rew_end(
            self.obs_buffer[:, -1:],
            self.act_buffer[:, -1:],
            next_obs,
            (self.hx_rew_end, self.cx_rew_end),
        )
        rew = Categorical(logits=logits_rew).sample().squeeze(1) - 1.0  # in {-1, 0, 1}
        end = Categorical(logits=logits_end).sample().squeeze(1)
        return rew, end

    @coroutine
    def make_generator_init(
        self,
        data_loader: DataLoader,
        num_batches_to_preload: int,
    ) -> Generator[InitialCondition, None, None]:
        num_dead = yield
        data_iterator = iter(data_loader)

        while True:
            # Preload on device and burnin rew/end model
            obs_, act_, hx_, cx_ = [], [], [], []
            for _ in range(num_batches_to_preload):
                batch = next(data_iterator)
                obs = batch.obs.to(self.device)
                act = batch.act.to(self.device)
                with torch.no_grad():
                    *_, (hx, cx) = self.rew_end_model.predict_rew_end(obs[:, :-1], act[:, :-1], obs[:, 1:])  # Burn-in of rew/end model
                assert hx.size(0) == cx.size(0) == 1
                obs_.extend(list(obs))
                act_.extend(list(act))
                hx_.extend(list(hx[0]))
                cx_.extend(list(cx[0]))

            # Yield new initial conditions for dead envs
            c = 0
            while c + num_dead <= len(obs_):
                obs = torch.stack(obs_[c : c + num_dead])
                act = torch.stack(act_[c : c + num_dead])
                hx = torch.stack(hx_[c : c + num_dead]).unsqueeze(0)
                cx = torch.stack(cx_[c : c + num_dead]).unsqueeze(0)
                c += num_dead
                num_dead = yield obs, act, (hx, cx)
