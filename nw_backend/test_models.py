"""
test_models.py
==============
Loads 10 saved PolicyNet models and runs them in a single NumberwarsEnv
game, then prints a full breakdown of results.

Usage:
    python -m nw_backend.test_models                         # default prefix "model"
    python -m nw_backend.test_models --save_prefix my_model  # custom prefix
    python -m nw_backend.test_models --num_players 10 --history_len 5
"""

from __future__ import annotations

import argparse
import os
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

from .env import NumberwarsEnv
from .train import PolicyNet, HiddenState


# ============================================================================
# CLI
# ============================================================================

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Fight 10 saved PPO+LSTM models against each other for one game."
    )
    p.add_argument("--num_players",  type=int, default=10,    help="Number of players (default: 10)")
    p.add_argument("--history_len",  type=int, default=10,     help="History length passed to env (default: 5)")
    p.add_argument("--lstm_hidden",  type=int, default=128,   help="LSTM hidden size used during training (default: 128)")
    p.add_argument("--save_prefix",  type=str, default="model", help="Model filename prefix (default: 'model')")
    p.add_argument("--seed",         type=int, default=0,     help="Env seed for reproducibility (default: 0)")
    return p.parse_args()


# ============================================================================
# Model loading
# ============================================================================

def load_models(
    possible_agents: List[str],
    obs_dim:         int,
    n_actions:       int = 101,
    lstm_hidden:     int = 128,
    path_prefix:     str = "model",
) -> Dict[str, PolicyNet]:
    nets: Dict[str, PolicyNet] = {}
    missing = []

    for agent in possible_agents:
        fname = f"{path_prefix}_{agent}.pt"
        if not os.path.isfile(fname):
            missing.append(fname)
            continue
        net = PolicyNet(obs_dim, n_actions, hidden=lstm_hidden)
        net.load_state_dict(torch.load(fname, map_location="cpu"))
        net.eval()
        nets[agent] = net
        print(f"  ✔  Loaded  {fname}")

    if missing:
        print("\n  ✘  Missing model files:")
        for f in missing:
            print(f"       {f}")
        raise FileNotFoundError(
            f"{len(missing)} model file(s) not found. "
            "Make sure you trained and saved with --save --save_prefix matching your --save_prefix here."
        )

    return nets


# ============================================================================
# Single game runner
# ============================================================================

def run_game(
    env:         NumberwarsEnv,
    nets:        Dict[str, PolicyNet],
    seed:        int = 0,
) -> Tuple[List[dict], Dict[str, float], Dict[str, int]]:
    """
    Plays one full game.

    Returns:
        history       : list of round snapshots
        final_rewards : cumulative reward per agent
        finish_order  : {agent: round_eliminated} (surviving agents get the last round)
    """
    obs_dict, _ = env.reset(seed=seed)
    last_obs: Dict[str, np.ndarray] = dict(obs_dict)

    hiddens: Dict[str, HiddenState] = {
        agent: nets[agent].init_hidden()
        for agent in env.possible_agents
    }

    cumulative_rewards: Dict[str, float] = {a: 0.0 for a in env.possible_agents}
    finish_order:       Dict[str, int]   = {}
    history:            List[dict]       = []
    round_num = 0

    while env.agents:
        round_num += 1
        actions:   Dict[str, int]         = {}
        new_hx:    Dict[str, HiddenState] = {}

        for agent in env.agents:
            with torch.no_grad():
                a, _, _, hx = nets[agent].act(last_obs[agent], hiddens[agent])
            actions[agent] = a
            new_hx[agent]  = hx

        next_obs_dict, rewards, terminations, truncations, infos = env.step(actions)

        # track eliminations this round
        newly_eliminated = []
        for agent in list(actions.keys()):
            cumulative_rewards[agent] += rewards.get(agent, 0.0)
            done = terminations.get(agent, False) or truncations.get(agent, False)
            if done and agent not in finish_order:
                finish_order[agent] = round_num
                newly_eliminated.append(agent)

        # Advance hidden states for surviving agents
        for agent in env.agents:
            if agent in new_hx:
                hiddens[agent] = new_hx[agent]

        # snapshot
        any_info = next(iter(infos.values()), {})
        history.append({
            "round":     round_num,
            "average":   any_info.get("average"),
            "target":    any_info.get("target"),
            "actions":   dict(actions),
            "rewards":   dict(rewards),
            "scores":    {a: infos[a].get("score") for a in infos},
            "alive":     list(env.agents),
            "eliminated": newly_eliminated,
        })

        last_obs.update(next_obs_dict)

    # surviving agents (if any) finished at the last round
    for agent in env.possible_agents:
        if agent not in finish_order:
            finish_order[agent] = round_num

    return history, cumulative_rewards, finish_order


# ============================================================================
# Pretty printing
# ============================================================================

MEDAL = {1: "🥇", 2: "🥈", 3: "🥉"}

def print_results(
    history:       List[dict],
    cum_rewards:   Dict[str, float],
    finish_order:  Dict[str, int],
    num_players:   int,
) -> None:
    sep  = "═" * 65
    thin = "─" * 65

    print(f"\n{sep}")
    print(f"  ⚔️   NUMBERWARS — TEST BATTLE ({num_players} players)  ⚔️")
    print(f"{sep}\n")

    # ── Round-by-round breakdown ──────────────────────────────────────
    print("  ROUND-BY-ROUND BREAKDOWN")
    print(f"  {thin}")

    for snap in history:
        avg    = snap["average"]
        target = snap["target"]
        elim   = snap["eliminated"]

        avg_str    = f"{avg:.1f}"    if avg    is not None else "?"
        target_str = f"{target:.1f}" if target is not None else "?"

        elim_str = ""
        if elim:
            elim_str = "  ☠  " + ", ".join(elim)

        print(f"  Round {snap['round']:>3d}  |  avg={avg_str:>6}  target={target_str:>6}{elim_str}")

        for agent in sorted(snap["actions"]):
            choice = snap["actions"][agent]
            score  = snap["scores"].get(agent, "?")
            rew    = snap["rewards"].get(agent, 0.0)
            marker = "💀" if agent in elim else "  "
            print(f"    {marker} {agent:<14}  chose={choice:>3}  "
                  f"score={str(score):>4}  reward={rew:+.2f}")
        print()

    # ── Final standings ───────────────────────────────────────────────
    print(f"  {thin}")
    print("  FINAL STANDINGS")
    print(f"  {thin}")

    # rank by finish_order (higher round = survived longer = better)
    ranked = sorted(finish_order.items(), key=lambda kv: -kv[1])

    # break ties by cumulative reward
    ranked = sorted(
        finish_order.items(),
        key=lambda kv: (kv[1], cum_rewards.get(kv[0], 0.0)),
        reverse=True,
    )

    print(f"  {'Rank':<5}  {'Agent':<14}  {'Survived Until':>14}  {'Cumulative Reward':>18}")
    print(f"  {'─'*4}  {'─'*14}  {'─'*14}  {'─'*18}")

    for rank, (agent, last_round) in enumerate(ranked, start=1):
        medal = MEDAL.get(rank, f"#{rank:>2}")
        cum_r = cum_rewards.get(agent, 0.0)
        print(f"  {medal:<5}  {agent:<14}  {'round '+str(last_round):>14}  {cum_r:>+18.4f}")

    winner, w_round = ranked[0]
    print(f"\n  🏆  Winner: {winner}  (survived {w_round} rounds, "
          f"reward={cum_rewards.get(winner, 0.0):+.4f})")
    print(f"{sep}\n")


# ============================================================================
# Main
# ============================================================================

def main() -> None:
    cfg = parse_args()

    env = NumberwarsEnv(num_players=cfg.num_players, history_len=cfg.history_len)

    obs_dim   = env._compute_obs_size()
    n_actions = 101

    print(f"\nLoading {cfg.num_players} models  (prefix='{cfg.save_prefix}') …\n")
    nets = load_models(
        possible_agents = env.possible_agents,
        obs_dim         = obs_dim,
        n_actions       = n_actions,
        lstm_hidden     = cfg.lstm_hidden,
        path_prefix     = cfg.save_prefix,
    )

    print(f"\nRunning one game  (seed={cfg.seed}) …")
    history, cum_rewards, finish_order = run_game(env, nets, seed=cfg.seed)

    print_results(history, cum_rewards, finish_order, cfg.num_players)


if __name__ == "__main__":
    main()
