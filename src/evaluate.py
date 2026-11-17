"""Evaluate an agent (random / defensive / trained PPO) on fixed episode seeds.

Usage:
    python -m src.evaluate --agent random --config configs/baseline.yaml
    python -m src.evaluate --agent defensive --config configs/baseline.yaml
    python -m src.evaluate --agent ppo --config configs/ppo_mid.yaml --train-seed 0
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Any, Callable, Optional

import gymnasium as gym
import numpy as np

from src.baseline import DefensiveDriverAgent, RandomAgent
from src.env_factory import load_config, make_env

RESULTS_DIR = Path("results")
SUMMARY_FILE = "eval_summary.csv"
# Schema cố định của CSV tổng hợp (một dòng cho mỗi lần evaluate một agent).
SUMMARY_COLUMNS = [
    "agent",
    "config",
    "train_seed",
    "n_episodes",
    "collision_rate",
    "mean_speed",
    "mean_episode_length",
    "mean_reward",
]

# act_fn(env, obs) -> action
ActFn = Callable[[gym.Env, Any], int]


def run_episodes(
    env: gym.Env, act_fn: ActFn, n_episodes: int, seed_start: int
) -> dict[str, float]:
    """Run ``n_episodes`` episodes (episode i uses seed ``seed_start + i``).

    Returns:
        Dict with ``collision_rate``, ``mean_speed``, ``mean_episode_length``
        and ``mean_reward`` averaged over episodes.
    """
    crashed_flags, speeds, lengths, rewards = [], [], [], []
    for i in range(n_episodes):
        obs, _ = env.reset(seed=seed_start + i)
        done, crashed, ep_reward, ep_speeds = False, False, 0.0, []
        while not done:
            obs, reward, terminated, truncated, _ = env.step(act_fn(env, obs))
            done = terminated or truncated
            ego = env.unwrapped.vehicle
            crashed = crashed or bool(ego.crashed)
            ep_reward += float(reward)
            ep_speeds.append(float(ego.velocity[0]))  # vận tốc dọc theo đường
        crashed_flags.append(crashed)
        speeds.append(np.mean(ep_speeds))
        lengths.append(len(ep_speeds))
        rewards.append(ep_reward)
    return {
        "collision_rate": float(np.mean(crashed_flags)),
        "mean_speed": float(np.mean(speeds)),
        "mean_episode_length": float(np.mean(lengths)),
        "mean_reward": float(np.mean(rewards)),
    }


def build_agent(agent: str, config: dict, model_path: Optional[Path]) -> ActFn:
    """Return an ``act_fn`` for the requested agent type."""
    seed_start = config["eval"]["seed_start"]
    if agent == "random":
        rnd = RandomAgent(seed=seed_start)
        return lambda env, obs: rnd.act(env)
    if agent == "defensive":
        drv = DefensiveDriverAgent()
        return lambda env, obs: drv.act(env)
    if agent == "ppo":
        from stable_baselines3 import PPO

        model = PPO.load(str(model_path), device="cpu")
        # deterministic=True: đánh giá policy tham lam, kết quả lặp lại được.
        return lambda env, obs: int(model.predict(obs, deterministic=True)[0])
    raise ValueError(f"Unknown agent: {agent}")


def append_summary(row: dict[str, Any], results_dir: Path = RESULTS_DIR) -> Path:
    """Append one row to the summary CSV (creating it with a header if needed)."""
    results_dir.mkdir(parents=True, exist_ok=True)
    path = results_dir / SUMMARY_FILE
    new_file = not path.exists()
    with open(path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=SUMMARY_COLUMNS)
        if new_file:
            writer.writeheader()
        writer.writerow({k: row[k] for k in SUMMARY_COLUMNS})
    return path


def evaluate(
    agent: str,
    config_path: str,
    train_seed: Optional[int] = None,
    n_episodes: Optional[int] = None,
    results_dir: Path = RESULTS_DIR,
) -> dict[str, Any]:
    """Evaluate one agent and append its metrics to the summary CSV."""
    cfg = load_config(config_path)
    name = cfg["meta"]["name"]
    n = int(n_episodes if n_episodes is not None else cfg["eval"]["n_episodes"])
    model_path = None
    if agent == "ppo":
        if train_seed is None:
            raise ValueError("--train-seed is required for agent=ppo")
        model_path = Path("models") / f"{name}_seed{train_seed}.zip"

    env = make_env(config_path)
    act_fn = build_agent(agent, cfg, model_path)
    metrics = run_episodes(env, act_fn, n, cfg["eval"]["seed_start"])
    env.close()

    row = {
        "agent": agent,
        "config": name,
        "train_seed": "" if train_seed is None else train_seed,
        "n_episodes": n,
        **metrics,
    }
    append_summary(row, results_dir)
    return row


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate an agent on highway-env.")
    parser.add_argument("--agent", choices=["random", "defensive", "ppo"], required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--train-seed", type=int, default=None)
    parser.add_argument("--n-episodes", type=int, default=None, help="Override eval.n_episodes")
    args = parser.parse_args()
    print(evaluate(args.agent, args.config, args.train_seed, args.n_episodes))


if __name__ == "__main__":
    main()
