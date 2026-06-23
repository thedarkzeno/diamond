import argparse
import os
from pathlib import Path
from typing import Tuple

from huggingface_hub import hf_hub_download
from hydra import compose, initialize
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
import torch
from torch.utils.data import DataLoader

from agent import Agent, SanaAgent
from coroutines.collector import make_collector, NumToCollect
from data import BatchSampler, collate_segments_to_batch, Dataset
from envs import make_atari_env, WorldModelEnv
from game import ActionNames, DatasetEnv, Game, get_keymap_and_action_names, Keymap, NamedEnv, PlayEnv
from game.csgo_game import CsgoGame
from game.csgo_play_env import CsgoPlayEnv
from utils import list_agent_ckpts, prompt_atari_game, resolve_agent_ckpt

OmegaConf.register_new_resolver("eval", eval)


def is_sana_agent(cfg: DictConfig) -> bool:
    return "SanaAgent" in cfg.agent.get("_target_", "")


def is_csgo_cfg(cfg: DictConfig) -> bool:
    return getattr(cfg.env, "keymap", None) == "csgo" or getattr(cfg.env.train, "id", None) == "csgo"


def resolve_static_dataset_path(cfg: DictConfig) -> Path:
    path = getattr(cfg, "static_dataset", None) and cfg.static_dataset.get("path")
    if path in (None, "null"):
        path = getattr(cfg.env, "path_data", None)
    if path in (None, "null"):
        raise FileNotFoundError(
            "No dataset path in saved config (static_dataset.path / env.path_data). "
            "Pass --path-data ~/processed_data to play.py."
        )
    path = Path(path).expanduser()
    if not path.is_absolute():
        path = Path.cwd() / path
    return path.resolve()


def apply_path_data_override(cfg: DictConfig, path_data: Path) -> None:
    path = str(path_data.expanduser().resolve())
    if getattr(cfg, "env", None) is not None:
        cfg.env.path_data = path
    if getattr(cfg, "static_dataset", None) is not None:
        cfg.static_dataset.path = path


def load_cfg() -> DictConfig:
    local_cfg = Path("config/trainer.yaml")
    if local_cfg.is_file():
        return OmegaConf.load(local_cfg)
    with initialize(version_base="1.3", config_path="../config"):
        return compose(config_name="trainer")


def run_has_playable_checkpoint(run_dir: Path) -> bool:
    ckpt = run_dir / "checkpoints"
    return bool(list_agent_ckpts(ckpt)) or (ckpt / "state.pt").is_file()


def find_latest_run_dir(root: Path) -> Path:
    outputs = root / "outputs"
    if not outputs.is_dir():
        raise FileNotFoundError(f"No outputs/ directory found under {root}")
    candidates = [p for p in outputs.glob("*/*") if run_has_playable_checkpoint(p)]
    if not candidates:
        raise FileNotFoundError(
            "No training runs with playable checkpoints found under outputs/. "
            "Need checkpoints/agent_versions/*.pt or checkpoints/state.pt."
        )
    return max(candidates, key=lambda p: p.stat().st_mtime)


def download(filename: str) -> Path:
    path = hf_hub_download(repo_id="eloialonso/diamond", filename=filename)
    return Path(path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--pretrained", action="store_true", help="Download pretrained world model and agent.")
    parser.add_argument("-d", "--dataset-mode", action="store_true", help="Dataset visualization mode.")
    parser.add_argument("-r", "--record", action="store_true", help="Record episodes in PlayEnv.")
    parser.add_argument("-n", "--num-steps-initial-collect", type=int, default=1000, help="Num steps initial collect.")
    parser.add_argument("--store-denoising-trajectory", action="store_true", help="Save denoising steps in info.")
    parser.add_argument("--store-original-obs", action="store_true", help="Save original obs (pre resizing) in info.")
    parser.add_argument("--fps", type=int, default=15, help="Frame rate.")
    parser.add_argument("--size", type=int, default=640, help="Window size.")
    parser.add_argument("--run-dir", type=Path, default=None, help="Hydra output dir (contains checkpoints/ and config/).")
    parser.add_argument("--path-data", type=Path, default=None, help="Processed dataset root (CS:GO: train/ + test/).")
    parser.add_argument("--latest-run", action="store_true", help="Use the most recent run under outputs/.")
    parser.add_argument("--epoch", type=int, default=-1, help="Checkpoint epoch (default: latest).")
    parser.add_argument("--size-multiplier", type=int, default=2, help="Window size multiplier (CS:GO).")
    parser.add_argument("--mouse-multiplier", type=int, default=10, help="Mouse sensitivity (CS:GO).")
    parser.add_argument("--windowed", action="store_true", help="Windowed mode instead of fullscreen (CS:GO).")
    parser.add_argument("--no-header", action="store_true")
    return parser.parse_args()


def check_args(args: argparse.Namespace) -> bool:
    if args.dataset_mode:
        if not Path("dataset").is_dir():
            print(f"Error: {str(Path('dataset').absolute())} not found, cannot use dataset mode.")
            return False
        if Path(".git").is_dir():
            print("Error: cannot run dataset mode the root of the repository.")
            return False
        if args.pretrained or args.record:
            print("Warning: dataset mode, ignoring --pretrained and --record")
    else:
        if not args.record and (args.store_denoising_trajectory or args.store_original_obs):
            print("Warning: not in recording mode, ignoring --store* options")
    return True


def prepare_dataset_mode(cfg: DictConfig) -> Tuple[DatasetEnv, Keymap, ActionNames]:
    datasets = []
    for p in Path("dataset").iterdir():
        if p.is_dir():
            d = Dataset(p, p.stem)
            d.load_from_default_path()
            datasets.append(d)
    _, env_action_names = get_keymap_and_action_names(cfg.env.keymap)
    dataset_env = DatasetEnv(datasets, env_action_names)
    keymap, _ = get_keymap_and_action_names("dataset_mode")
    return dataset_env, keymap


def prepare_play_mode(cfg: DictConfig, args: argparse.Namespace) -> Tuple[PlayEnv, Keymap, ActionNames]:
    sana = is_sana_agent(cfg)

    if args.pretrained:
        name = prompt_atari_game()
        path_ckpt = download(f"atari_100k/models/{name}.pt")
        cfg.agent = OmegaConf.load(download("atari_100k/config/agent/default.yaml"))
        cfg.env = OmegaConf.load(download("atari_100k/config/env/atari.yaml"))
        cfg.env.train.id = cfg.env.test.id = f"{name}NoFrameskip-v4"
        cfg.world_model_env.horizon = 50
        sana = False
    else:
        path_ckpt = resolve_agent_ckpt("checkpoints", epoch=args.epoch)
        print(f"Loading checkpoint: {path_ckpt}")

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    train_env = make_atari_env(num_envs=1, device=device, **cfg.env.train)
    test_env = make_atari_env(num_envs=1, device=device, **cfg.env.test)

    agent_cfg = instantiate(cfg.agent, num_actions=test_env.num_actions)
    if sana:
        agent = SanaAgent(agent_cfg).to(device).eval()
        agent.vae.apply_device_policy(device)
    else:
        agent = Agent(agent_cfg).to(device).eval()
    agent.load(path_ckpt)

    n = args.num_steps_initial_collect
    dataset = Dataset(Path(f"dataset/{path_ckpt.stem}_{n}"))
    dataset.load_from_default_path()
    if len(dataset) == 0:
        print(f"Collecting {n} steps in real environment for world model initialization.")
        collector = make_collector(test_env, agent.actor_critic, dataset, epsilon=0)
        collector.send(NumToCollect(steps=n))
        dataset.save_to_default_path()

    sl = cfg.agent.denoiser.inner_model.num_steps_conditioning
    bs = BatchSampler(dataset, 0, 1, 1, sl, None, False)
    dl = DataLoader(dataset, batch_sampler=bs, collate_fn=collate_segments_to_batch)
    wm_env_cfg = instantiate(cfg.world_model_env, num_batches_to_preload=1)
    wm_env = WorldModelEnv(
        agent.denoiser,
        agent.rew_end_model,
        dl,
        wm_env_cfg,
        return_denoising_trajectory=True,
        vae=agent.vae if sana else None,
    )

    envs = [
        NamedEnv("wm", wm_env),
        NamedEnv("test", test_env),
        NamedEnv("train", train_env),
    ]

    env_keymap, env_action_names = get_keymap_and_action_names(cfg.env.keymap)
    play_env = PlayEnv(
        agent,
        envs,
        env_action_names,
        env_keymap,
        args.record,
        args.store_denoising_trajectory,
        args.store_original_obs,
    )

    return play_env, env_keymap


def prepare_csgo_play_mode(cfg: DictConfig, args: argparse.Namespace) -> CsgoPlayEnv:
    if not is_sana_agent(cfg):
        raise RuntimeError("CS:GO play mode currently supports SANA agents only.")

    path_ckpt = resolve_agent_ckpt("checkpoints", epoch=args.epoch)
    print(f"Loading checkpoint: {path_ckpt}")

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    num_actions = int(cfg.env.num_actions)
    agent_cfg = instantiate(cfg.agent, num_actions=num_actions)
    agent = SanaAgent(agent_cfg).to(device).eval()
    agent.vae.apply_device_policy(device)
    agent.load(path_ckpt)

    data_root = resolve_static_dataset_path(cfg)
    use_latents = bool(getattr(cfg, "use_cached_latents", False) or agent_cfg.use_cached_latents)
    dataset = Dataset(data_root / "test", "test_dataset", cache_in_ram=True, use_latents=use_latents)
    dataset.load_from_default_path()
    if dataset.num_episodes == 0:
        dataset = Dataset(data_root / "train", "train_dataset", cache_in_ram=True, use_latents=use_latents)
        dataset.load_from_default_path()

    from csgo.action_layout import ACTION_DIM
    from data.csgo_validate import validate_csgo_dataset

    action_dim = int(getattr(cfg.env, "action_dim", ACTION_DIM))
    model_action_dim = int(agent_cfg.denoiser.inner_model.action_dim)
    if action_dim != model_action_dim:
        raise ValueError(f"Config action_dim={action_dim} != model action_dim={model_action_dim}")
    validate_csgo_dataset(
        dataset,
        action_dim,
        require_latents=use_latents,
    )

    sl = cfg.agent.denoiser.inner_model.num_steps_conditioning
    bs = BatchSampler(dataset, 0, 1, 1, sl, None, False)
    dl = DataLoader(dataset, batch_sampler=bs, collate_fn=collate_segments_to_batch, num_workers=0)
    wm_env_cfg = instantiate(cfg.world_model_env, num_batches_to_preload=1)
    wm_env = WorldModelEnv(
        agent.denoiser,
        None,
        dl,
        wm_env_cfg,
        return_denoising_trajectory=args.store_denoising_trajectory,
        vae=agent.vae,
    )

    return CsgoPlayEnv(agent, wm_env, recording_mode=args.record)


@torch.no_grad()
def main():
    args = parse_args()
    ok = check_args(args)
    if not ok:
        return

    project_root = Path(__file__).resolve().parents[1]
    if args.latest_run:
        run_dir = find_latest_run_dir(project_root)
    elif args.run_dir is not None:
        run_dir = args.run_dir.expanduser().resolve()
    else:
        run_dir = None

    if run_dir is not None:
        os.chdir(run_dir)
        print(f"Using run directory: {run_dir}")

    cfg = load_cfg()
    if args.path_data is not None:
        apply_path_data_override(cfg, args.path_data)

    if args.dataset_mode:
        env, keymap = prepare_dataset_mode(cfg)
        size = (args.size // cfg.env.train.size) * cfg.env.train.size
        game = Game(env, keymap, (size, size), fps=args.fps, verbose=not args.no_header)
    elif is_csgo_cfg(cfg):
        env = prepare_csgo_play_mode(cfg, args)
        res = cfg.env.train.size
        res = (res, res) if isinstance(res, int) else tuple(res)
        h, w = res
        game = CsgoGame(
            env,
            (h * args.size_multiplier, w * args.size_multiplier),
            mouse_multiplier=args.mouse_multiplier,
            fps=args.fps,
            verbose=not args.no_header,
            fullscreen=not args.windowed,
        )
    else:
        env, keymap = prepare_play_mode(cfg, args)
        size = (args.size // cfg.env.train.size) * cfg.env.train.size
        game = Game(env, keymap, (size, size), fps=args.fps, verbose=not args.no_header)

    game.run()


if __name__ == "__main__":
    main()
