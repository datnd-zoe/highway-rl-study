"""Environment factory: build a highway-env environment from a YAML config."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional, Union

import gymnasium as gym
import highway_env  # noqa: F401  (đăng ký các env "highway-v0" vào gymnasium)
import yaml


def load_config(config_path: Union[str, Path]) -> dict[str, Any]:
    """Load an experiment YAML config into a dictionary.

    Args:
        config_path: Path to a YAML file under ``configs/``.

    Returns:
        The parsed config with at least an ``env`` section.
    """
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    if not isinstance(cfg, dict) or "env" not in cfg:
        raise ValueError(f"Config {config_path} must contain an 'env' section")
    return cfg


def make_env(
    config_path: Union[str, Path],
    render_mode: Optional[str] = None,
    collision_reward: Optional[float] = None,
) -> gym.Env:
    """Create a highway-env environment from a YAML config.

    Args:
        config_path: Path to the YAML config file.
        render_mode: Gymnasium render mode (``None``, ``"rgb_array"``, ``"human"``).
        collision_reward: Optional override for the independent variable.

    Returns:
        A configured ``gymnasium.Env``.
    """
    cfg = load_config(config_path)
    env_cfg = dict(cfg["env"])  # copy để không sửa dict gốc
    env_id = env_cfg.pop("id", "highway-v0")

    if collision_reward is not None:
        env_cfg["collision_reward"] = float(collision_reward)

    # Quyết định thiết kế (PROJECT_PLAN §3): luôn tắt normalize_reward để
    # collision_reward không bị nén vào dải [0.976, 1.0] làm mất tín hiệu học.
    env_cfg["normalize_reward"] = False

    env = gym.make(env_id, render_mode=render_mode, config=env_cfg)
    return env
