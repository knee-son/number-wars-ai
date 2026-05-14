"""
numberwars_env.py
=====================
PettingZoo ParallelEnv wrapper for the "Closest to 80% of Average" game.

Install deps:
    pip install pettingzoo gymnasium numpy

Quick start:
    from .env import NumberwarsEnv

    env = NumberwarsEnv(num_players=4, history_len=5)
    observations, infos = env.reset()

    while env.agents:
        actions = {agent: env.action_space(agent).sample() for agent in env.agents}
        observations, rewards, terminations, truncations, infos = env.step(actions)

Observation layout (per agent, flat float32 array of fixed size):
    [0]   own_score             / INITIAL_SCORE        → [0, 1]
    [1]   own_consecutive_wins  / MAX_PLAYERS          → [0, 1]
    [2]   num_players_alive     / MAX_PLAYERS          → [0, 1]
    [3]   current_round         / MAX_ROUNDS           → [0, 1]
    [4:]  opponent history, shape (MAX_PLAYERS-1, history_len), flattened.
          Ordered by fixed slot index (self excluded), oldest-first per slot.
          Valid choices are normalised to [0, 1] (value / 100).
          Padding / eliminated / absent slots use -1.

Obs size = 4 + (MAX_PLAYERS - 1) * history_len  — constant for all configs.
"""

from __future__ import annotations

from typing import Optional
import functools

import numpy as np
from gymnasium import spaces
from pettingzoo import ParallelEnv
from pettingzoo.utils import parallel_to_aec, wrappers

from .game_rules import GameRules, GameState, INITIAL_SCORE, MAX_PLAYERS


# ---------------------------------------------------------------------------
# Module-level constants
# ---------------------------------------------------------------------------

MAX_ROUNDS:   int = 100
_NUMBER_LOW:  int = 0
_NUMBER_HIGH: int = 100

# Fixed observation size — never changes regardless of num_players or
# how many players remain alive.
_N_OPP_SLOTS: int = MAX_PLAYERS - 1  # 9 when MAX_PLAYERS == 10


# ---------------------------------------------------------------------------
# Env
# ---------------------------------------------------------------------------

class NumberwarsEnv(ParallelEnv):
    """
    Parallel (simultaneous-action) PettingZoo environment.

    All agents submit their integer choice [0, 100] at the same time each
    step, matching the real game's simultaneous reveal mechanic.

    Parameters
    ----------
    num_players : int
        Number of agents (2–MAX_PLAYERS).
    history_len : int
        How many past rounds of opponent choices each agent observes.
        Older rounds and absent slots are padded with -1.
    """

    metadata = {
        "render_modes": ["human"],
        "name": "numberwars_80pct_avg_v0",
        "is_parallelizable": True,
    }

    def __init__(self, num_players: int = 4, history_len: int = 10) -> None:
        super().__init__()

        if not (2 <= num_players <= MAX_PLAYERS):
            raise ValueError(f"num_players must be 2–{MAX_PLAYERS}, got {num_players}")
        if history_len < 1:
            raise ValueError(f"history_len must be >= 1, got {history_len}")

        self.num_players  = num_players
        self.history_len  = history_len

        # Fixed agent names — never mutated after __init__.
        self.possible_agents: list[str] = [
            f"player_{i}" for i in range(num_players)
        ]

        # Runtime state (initialised in reset())
        self.agents: list[str] = []
        self._state: Optional[GameState] = None

        # Per-agent circular history buffer: agent_id → list[int | -1]
        # Keyed by ALL possible_agents so eliminated agents keep their slot.
        self._history: dict[str, list[int]] = {}
        self._obs_size: int = self._compute_obs_size()

    def _compute_obs_size(self) -> int:
        n_opponents = MAX_PLAYERS - 1
        return 4 + self.history_len * n_opponents

    # ------------------------------------------------------------------
    # Spaces  (lru_cache satisfies PettingZoo's "same object" contract)
    # ------------------------------------------------------------------

    @functools.lru_cache(maxsize=None)
    def observation_space(self, agent: str) -> spaces.Box:
        """
        Fixed-size Box in [-1, 1].
        -1 is the canonical "unknown / absent" sentinel.
        Valid feature values lie in [0, 1].
        """
        low  = np.full(self._obs_size, -1.0, dtype=np.float32)
        high = np.ones(self._obs_size,        dtype=np.float32)
        return spaces.Box(low=low, high=high, dtype=np.float32)

    @functools.lru_cache(maxsize=None)
    def action_space(self, agent: str) -> spaces.Discrete:
        """Integer in [0, 100] → Discrete(101)."""
        return spaces.Discrete(_NUMBER_HIGH - _NUMBER_LOW + 1)

    # ------------------------------------------------------------------
    # PettingZoo API
    # ------------------------------------------------------------------

    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[dict] = None,
    ) -> tuple[dict[str, np.ndarray], dict[str, dict]]:

        if seed is not None:
            np.random.seed(seed)

        self._state  = GameState()
        self.agents  = list(self.possible_agents)
        self._history = {agent: [] for agent in self.possible_agents}

        for agent in self.possible_agents:
            GameRules.add_player(self._state, agent, agent)

        observations = {a: self._observe(a) for a in self.agents}
        infos        = {a: {}               for a in self.agents}
        return observations, infos

    def step(
        self,
        actions: dict[str, int],
    ) -> tuple[
        dict[str, np.ndarray],
        dict[str, float],
        dict[str, bool],
        dict[str, bool],
        dict[str, dict],
    ]:
        # Snapshot agents alive at the START of this step.
        # PettingZoo requires return dicts to contain exactly these keys,
        # including agents that get eliminated during the step.
        active_before = list(self.agents)

        # ── 1. Submit actions ──────────────────────────────────────────
        for agent, action in actions.items():
            if agent in self._state.players:
                GameRules.submit_number(self._state, agent, int(action))

        # ── 2. Record choices into history BEFORE resolve clears them ──
        for agent in self.possible_agents:
            player = self._state.players.get(agent)
            val    = player.submitted_number if (player and player.submitted_number is not None) else -1
            buf    = self._history[agent]
            buf.append(val)
            if len(buf) > self.history_len:
                # Trim in-place — avoids a fresh list allocation every step
                del buf[0]

        # ── 3. Resolve the round ───────────────────────────────────────
        result = GameRules.resolve_round(self._state)

        # ── 4. Build return dicts ──────────────────────────────────────
        truncate_all = self._state.current_round >= MAX_ROUNDS

        rewards:      dict[str, float] = {}
        terminations: dict[str, bool]  = {}
        truncations:  dict[str, bool]  = {}
        infos:        dict[str, dict]  = {}

        for agent in active_before:
            terminated = agent in result.eliminated_ids or self._state.game_over
            rewards[agent]      = float(result.rewards.get(agent, 0.0))
            terminations[agent] = terminated
            truncations[agent]  = truncate_all
            infos[agent] = {
                "round":          result.round_number,
                "average":        result.average,
                "target":         result.target,
                "winner":         result.winner_id,
                "rule_triggered": result.info.get("rule_triggered"),
                "score":          result.scores_after.get(agent, 0),
                "eliminated":     agent in result.eliminated_ids,
            }

        # ── 5. Update live agent list ──────────────────────────────────
        self.agents = [
            a for a in self.agents
            if not terminations[a] and not truncations[a]
        ]

        # ── 6. Observations for every agent that was alive this step ───
        observations: dict[str, np.ndarray] = {
            a: self._observe(a) for a in active_before
        }

        return observations, rewards, terminations, truncations, infos

    def render(self) -> None:
        if self._state is None:
            print("Environment not initialised — call reset() first.")
            return
        print(f"\n── Round {self._state.current_round} ──")
        for pid, player in self._state.players.items():
            print(f"  {pid:12s}  score={player.score:2d}  "
                  f"consec_wins={player.consecutive_wins}")
        if self._state.elimination_order:
            print("  Eliminated:", [e["player_id"] for e in self._state.elimination_order])

    def close(self) -> None:
        pass

    # ------------------------------------------------------------------
    # Observation builder
    # ------------------------------------------------------------------

    def _observe(self, agent: str) -> np.ndarray:
        """
        Returns a fixed-size float32 vector of shape (self._obs_size,).

        Slots for players that were never in this game (i.e. num_players <
        MAX_PLAYERS) are permanently -1, giving a consistent layout for any
        model trained across different player counts.
        """
        obs    = np.full(self._obs_size, -1.0, dtype=np.float32)
        player = self._state.players.get(agent)

        # ── Scalar features ────────────────────────────────────────────
        obs[0] = (player.score             if player else 0) / INITIAL_SCORE
        obs[1] = (player.consecutive_wins  if player else 0) / MAX_PLAYERS   # fixed scale
        obs[2] = len(self._state.players)                    / MAX_PLAYERS   # fixed scale
        obs[3] = self._state.current_round                   / MAX_ROUNDS

        # ── Opponent history ───────────────────────────────────────────
        # Opponent slots are indexed by their position in possible_agents
        # (self excluded), always ordered the same way.
        # Slots beyond num_players stay at -1 (already initialised above).
        opp_index = 0
        cursor    = 4
        for opp in self.possible_agents:
            if opp == agent:
                continue
            hist   = self._history[opp]                        # list of ints, len ≤ history_len
            padded = ([-1] * self.history_len + hist)[-self.history_len:]   # oldest-first, left-padded
            for val in padded:
                obs[cursor] = val / 100.0 if val != -1 else -1.0
                cursor += 1
            opp_index += 1

        # cursor is now at 4 + num_players-1 * history_len.
        # Remaining slots (for absent MAX_PLAYERS slots) stay -1.
        return obs


# ---------------------------------------------------------------------------
# AEC wrapper (for turn-based algorithms)
# ---------------------------------------------------------------------------

def env(**kwargs) -> wrappers.OrderEnforcingWrapper:
    """
    Returns a fully wrapped AEC environment.

        from .env import env
        e = env(num_players=10, history_len=10)
        e.reset()
    """
    raw = NumberwarsEnv(**kwargs)
    aec = parallel_to_aec(raw)
    return wrappers.OrderEnforcingWrapper(aec)


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    from pettingzoo.test import parallel_api_test

    print("Running PettingZoo parallel API test …")
    test_env = NumberwarsEnv(num_players=4, history_len=3)
    parallel_api_test(test_env, num_cycles=100)
    print("All checks passed.")

    print("\nManual rollout (random policy, 4 players) …")
    test_env2 = NumberwarsEnv(num_players=4, history_len=3)
    obs, _ = test_env2.reset(seed=42)
    print(f"Obs shape : {obs['player_0'].shape}")
    print(f"Obs size  : {test_env2._obs_size}  "
          f"(4 + {_N_OPP_SLOTS} * {test_env2.history_len})")

    step = 0
    while test_env2.agents:
        step += 1
        acts = {a: test_env2.action_space(a).sample() for a in test_env2.agents}
        obs, rews, terms, truncs, infos = test_env2.step(acts)
        test_env2.render()
        print(f"  actions={acts}  rewards={rews}")
        if all(terms.values()):
            break

    print(f"\nGame ended after {step} round(s).")
    print("Rankings:", GameRules.get_rankings(test_env2._state))
