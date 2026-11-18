"""Train a PPO agent (Stable-Baselines3) from a YAML experiment config.

Usage:
    python -m src.train --config configs/ppo_mid.yaml --seed 0
    python -m src.train --config configs/ppo_mid.yaml --seed 0 --timesteps 2000
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.monitor import Monitor

from src.env_factory import load_config, make_env

MODELS_DIR = Path("models")
RESULTS_DIR = Path("results")


def _latest_checkpoint(ckpt_dir: Path, run_id: str) -> Optional[Path]:
    """Return the checkpoint with the most steps for ``run_id`` (or None)."""
    found = []
    for p in ckpt_dir.glob(f"{run_id}_*_steps.zip"):
        steps_str = p.stem[len(run_id) + 1 :].removesuffix("_steps")
        if steps_str.isdigit():
            found.append((int(steps_str), p))
    return max(found)[1] if found else None


def train(
    config_path: str,
    seed: int,
    total_timesteps: Optional[int] = None,
    models_dir: Path = MODELS_DIR,
    results_dir: Path = RESULTS_DIR,
    checkpoint_freq: int = 10_000,
    resume: bool = False,
) -> Path:
    """Train one PPO run and save the model checkpoint.

    Args:
        config_path: Path to a ``ppo_*.yaml`` config (must contain a ``train`` section).
        seed: Training seed (controls env, network init and rollout sampling).
        total_timesteps: Override of ``train.total_timesteps`` (e.g. for smoke tests).
        models_dir: Directory where the ``.zip`` checkpoint is written.
        results_dir: Directory for Monitor CSV logs (learning curves).

    Returns:
        Path of the saved model (without the ``.zip`` suffix SB3 appends).
    """
    cfg = load_config(config_path)
    if "train" not in cfg:
        raise ValueError(f"Config {config_path} has no 'train' section")
    tcfg = cfg["train"]
    name = cfg["meta"]["name"]
    steps = int(total_timesteps if total_timesteps is not None else tcfg["total_timesteps"])

    models_dir.mkdir(parents=True, exist_ok=True)
    log_dir = results_dir / "train_logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    run_id = f"{name}_seed{seed}"

    # Monitor ghi reward/độ dài từng episode ra CSV để vẽ learning curve sau này.
    # Khi resume thì nối tiếp file log cũ thay vì ghi đè.
    env = Monitor(
        make_env(config_path), filename=str(log_dir / run_id), override_existing=not resume
    )
    env.reset(seed=seed)

    ckpt_dir = models_dir / "checkpoints"
    latest = _latest_checkpoint(ckpt_dir, run_id) if resume else None
    if latest is not None:
        # Chạy tiếp từ checkpoint (RNG không khôi phục nên không lặp lại 100% bit-for-bit).
        model = PPO.load(str(latest), env=env, device="cpu")
        print(f"Resuming from {latest} at {model.num_timesteps} steps")
    else:
        # Siêu tham số PPO lấy hoàn toàn từ YAML: ba config chỉ khác collision_reward.
        model = PPO(
            tcfg["policy"],
            env,
            n_steps=tcfg["n_steps"],
            batch_size=tcfg["batch_size"],
            n_epochs=tcfg["n_epochs"],
            learning_rate=tcfg["learning_rate"],
            gamma=tcfg["gamma"],
            gae_lambda=tcfg["gae_lambda"],
            clip_range=tcfg["clip_range"],
            ent_coef=tcfg["ent_coef"],
            verbose=tcfg.get("verbose", 1),
            seed=seed,
            device="cpu",
        )

    # Lưu checkpoint định kỳ để run dài (nhiều giờ) không mất hết khi bị ngắt.
    callback = CheckpointCallback(
        save_freq=checkpoint_freq, save_path=str(ckpt_dir), name_prefix=run_id
    )
    remaining = max(steps - model.num_timesteps, 0)
    model.learn(total_timesteps=remaining, callback=callback, reset_num_timesteps=False)

    save_path = models_dir / run_id
    model.save(str(save_path))
    env.close()
    return save_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Train PPO on highway-env.")
    parser.add_argument("--config", required=True, help="Path to ppo_*.yaml")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--timesteps", type=int, default=None, help="Override total_timesteps")
    parser.add_argument("--checkpoint-freq", type=int, default=10_000)
    parser.add_argument("--resume", action="store_true", help="Continue from latest checkpoint")
    args = parser.parse_args()
    path = train(
        args.config, args.seed, args.timesteps,
        checkpoint_freq=args.checkpoint_freq, resume=args.resume,
    )
    print(f"Saved model to {path}.zip")


if __name__ == "__main__":
    main()
