"""Baseline agents: a random lower bound and a rule-based defensive driver.

Both agents expose ``act(env) -> int`` so the evaluator can treat them like a
trained policy. Neither agent reads the reward, so their decisions are
independent of ``collision_reward``.
"""

from __future__ import annotations

from typing import Optional

import gymnasium as gym
import numpy as np

# Action indices của highway-env (DiscreteMetaAction, ACTIONS_ALL).
LANE_LEFT, IDLE, LANE_RIGHT, FASTER, SLOWER = 0, 1, 2, 3, 4
N_ACTIONS = 5


class RandomAgent:
    """Uniformly random policy over the 5 discrete meta-actions (lower bound)."""

    def __init__(self, seed: Optional[int] = None) -> None:
        self.rng = np.random.default_rng(seed)

    def act(self, env: gym.Env) -> int:
        """Return a uniformly random action."""
        return int(self.rng.integers(N_ACTIONS))


class DefensiveDriverAgent:
    """Rule-based driver reading true vehicle state from the simulator.

    Rules:
        * default: ``FASTER``;
        * blocked (front vehicle closer than ``front_gap`` in own lane):
          try left lane, then right lane, else ``SLOWER``;
        * not blocked and not in the rightmost lane: move right if safe.
    """

    def __init__(self, front_gap: float = 40.0, rear_gap: float = 15.0) -> None:
        self.front_gap = front_gap  # [m] khoảng cách phía trước tối thiểu
        self.rear_gap = rear_gap    # [m] khoảng cách phía sau tối thiểu khi đổi làn

    def _gaps(self, env: gym.Env, lane_index: tuple) -> tuple[float, float]:
        """Return (front gap, rear gap) in metres for ``lane_index`` (inf if empty)."""
        ego = env.unwrapped.vehicle
        front, rear = env.unwrapped.road.neighbour_vehicles(ego, lane_index)
        front_d = front.position[0] - ego.position[0] if front else np.inf
        rear_d = ego.position[0] - rear.position[0] if rear else np.inf
        return float(front_d), float(rear_d)

    def _lane_is_free(self, env: gym.Env, lane_index: tuple) -> bool:
        """A target lane is safe if it has room both ahead and behind."""
        front_d, rear_d = self._gaps(env, lane_index)
        return front_d > self.front_gap and rear_d > self.rear_gap

    def act(self, env: gym.Env) -> int:
        """Choose an action from the current simulator state."""
        ego = env.unwrapped.vehicle
        road = env.unwrapped.road
        lane_index = ego.lane_index  # (from, to, lane_id); lane_id 0 = trái nhất
        _from, _to, lane_id = lane_index
        n_lanes = len(road.network.graph[_from][_to])

        left = (_from, _to, lane_id - 1) if lane_id > 0 else None
        right = (_from, _to, lane_id + 1) if lane_id < n_lanes - 1 else None

        front_d, _ = self._gaps(env, lane_index)
        if front_d < self.front_gap:
            # Bị chặn: ưu tiên vượt trái, rồi phải, cuối cùng giảm tốc.
            if left is not None and self._lane_is_free(env, left):
                return LANE_LEFT
            if right is not None and self._lane_is_free(env, right):
                return LANE_RIGHT
            return SLOWER

        # Không bị chặn: dồn về làn phải nếu an toàn.
        if right is not None and self._lane_is_free(env, right):
            return LANE_RIGHT
        return FASTER
