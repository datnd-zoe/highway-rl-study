"""Automated tests for the highway-rl-study pipeline.

Môi trường mô phỏng khá chậm trên CPU (~0.3s/step) nên các test chỉ chạy
vài step / 1 episode.
"""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import pytest

from src.baseline import DefensiveDriverAgent, RandomAgent
from src.env_factory import make_env
from src.evaluate import SUMMARY_COLUMNS, evaluate
from src.plot_results import aggregate

CONFIGS = Path("configs")
EXPECTED_COLLISION = {"ppo_low": -1.0, "ppo_mid": -5.0, "ppo_high": -20.0, "baseline": -5.0}


@pytest.mark.parametrize("name,expected", EXPECTED_COLLISION.items())
def test_make_env_from_config(name: str, expected: float) -> None:
    """Env builds from each YAML with the right collision_reward and no normalization."""
    env = make_env(CONFIGS / f"{name}.yaml")
    cfg = env.unwrapped.config
    assert cfg["collision_reward"] == expected
    assert cfg["normalize_reward"] is False
    # Biến độc lập duy nhất là collision_reward: các thành phần khác phải giữ nguyên.
    assert cfg["high_speed_reward"] == 0.4
    assert cfg["right_lane_reward"] == 0.1
    assert cfg["lane_change_reward"] == 0.0
    obs, _ = env.reset(seed=0)
    assert obs.shape == (5, 5)
    env.close()


def _rollout(config: str, seed: int, actions: list[int]) -> tuple[list, list]:
    env = make_env(CONFIGS / config)
    obs, _ = env.reset(seed=seed)
    observations, rewards = [obs.copy()], []
    for a in actions:
        obs, r, term, trunc, _ = env.step(a)
        observations.append(obs.copy())
        rewards.append(r)
        if term or trunc:
            break
    env.close()
    return observations, rewards


def test_fixed_seed_is_reproducible() -> None:
    """Same seed + same actions -> identical observations and rewards."""
    actions = [3, 1, 0, 3, 4]
    obs_a, rew_a = _rollout("ppo_mid.yaml", 123, actions)
    obs_b, rew_b = _rollout("ppo_mid.yaml", 123, actions)
    assert rew_a == rew_b
    for a, b in zip(obs_a, obs_b):
        np.testing.assert_array_equal(a, b)


def test_summary_csv_schema(tmp_path: Path) -> None:
    """evaluate() writes a CSV whose header matches SUMMARY_COLUMNS."""
    row = evaluate("random", str(CONFIGS / "baseline.yaml"), n_episodes=1, results_dir=tmp_path)
    with open(tmp_path / "eval_summary.csv", newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    assert list(rows[0].keys()) == SUMMARY_COLUMNS
    assert len(rows) == 1
    assert row["collision_rate"] in (0.0, 1.0)  # 1 episode
    assert 0 < row["mean_episode_length"] <= 40
    assert row["mean_speed"] > 0


def _agent_trace(config: str, agent_factory, n_steps: int = 8) -> list[tuple[int, float]]:
    """Return (action, ego speed) per step for an agent on a fixed seed."""
    env = make_env(CONFIGS / config)
    env.reset(seed=100)
    agent = agent_factory()
    trace = []
    for _ in range(n_steps):
        a = agent.act(env)
        _, _, term, trunc, _ = env.step(a)
        trace.append((a, float(env.unwrapped.vehicle.speed)))
        if term or trunc:
            break
    env.close()
    return trace


@pytest.mark.parametrize(
    "factory", [DefensiveDriverAgent, lambda: RandomAgent(seed=0)], ids=["defensive", "random"]
)
def test_baselines_independent_of_collision_reward(factory) -> None:
    """Baselines behave identically whatever collision_reward is."""
    low = _agent_trace("ppo_low.yaml", factory)
    high = _agent_trace("ppo_high.yaml", factory)
    assert low == high


def test_aggregate_mean_std(tmp_path: Path) -> None:
    """PPO rows are aggregated by config over training seeds; baselines have std 0."""
    csv_path = tmp_path / "s.csv"
    csv_path.write_text(
        "agent,config,train_seed,n_episodes,collision_rate,mean_speed,mean_episode_length,mean_reward\n"
        "random,baseline,,50,1.0,23,10,1\n"
        "ppo,ppo_mid,0,50,0.2,25,35,1\n"
        "ppo,ppo_mid,1,50,0.4,27,35,1\n",
        encoding="utf-8",
    )
    agg = aggregate(csv_path)
    assert list(agg.index) == ["random", "ppo_mid"]
    assert agg.loc["ppo_mid", "collision_rate_mean"] == pytest.approx(0.3)
    assert agg.loc["ppo_mid", "collision_rate_std"] == pytest.approx(0.1)
    assert agg.loc["random", "mean_speed_std"] == 0


def test_latest_checkpoint_picks_most_steps(tmp_path: Path) -> None:
    """_latest_checkpoint compares step counts numerically and ignores other runs."""
    from src.train import _latest_checkpoint

    for name in [
        "ppo_mid_seed0_9000_steps.zip",
        "ppo_mid_seed0_10000_steps.zip",   # 10000 > 9000 (so sánh số, không so sánh chuỗi)
        "ppo_mid_seed1_90000_steps.zip",   # run khác, phải bị bỏ qua
        "ppo_mid_seed0_notes.zip",
    ]:
        (tmp_path / name).touch()
    assert _latest_checkpoint(tmp_path, "ppo_mid_seed0").name == "ppo_mid_seed0_10000_steps.zip"
    assert _latest_checkpoint(tmp_path, "ppo_high_seed0") is None
