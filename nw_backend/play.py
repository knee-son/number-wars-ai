"""
play.py
=======
Play "Closest to 80% of Average" against 5 trained PPO agents in your terminal.

Usage:
    python play.py                                      # looks for model_player_N.pt
    python play.py --model_prefix my_run/model          # custom save path prefix
    python play.py --model_prefix model --you player_0  # pick which slot you occupy
"""

from __future__ import annotations

import argparse
import sys
from typing import Dict

import numpy as np
import torch

from .env import NumberwarsEnv
from .train import PolicyNet, load_models   # reuse what we already wrote

# ── ANSI colours (degrade gracefully on Windows) ────────────────────────────
try:
    import os
    if os.name == "nt":
        raise ImportError
    BOLD   = "\033[1m"
    GREEN  = "\033[92m"
    YELLOW = "\033[93m"
    RED    = "\033[91m"
    CYAN   = "\033[96m"
    DIM    = "\033[2m"
    RESET  = "\033[0m"
except ImportError:
    BOLD = GREEN = YELLOW = RED = CYAN = DIM = RESET = ""

NUM_BOTS   = 9
NUM_PLAYERS = NUM_BOTS + 1   # you + 5 bots
N_ACTIONS   = 101


# ============================================================================
# Helpers
# ============================================================================

def colour_number(n: int) -> str:
    """Green if low strategic, yellow if mid, red if high."""
    if n <= 33:
        return f"{GREEN}{n}{RESET}"
    if n <= 66:
        return f"{YELLOW}{n}{RESET}"
    return f"{RED}{n}{RESET}"


def ask_number(agent_name: str, score: int, alive: list[str]) -> int:
    """Prompt the human until they enter a valid integer 0–100."""
    while True:
        try:
            raw = input(f"\n{BOLD}Your turn{RESET} "
                        f"(score={CYAN}{score}{RESET}, "
                        f"players alive={len(alive)})  "
                        f"Enter 0–100: ").strip()
            val = int(raw)
            if 0 <= val <= 100:
                return val
            print(f"  {RED}Must be between 0 and 100.{RESET}")
        except (ValueError, EOFError):
            print(f"  {RED}Please type a whole number.{RESET}")
        except KeyboardInterrupt:
            print("\nBye!")
            sys.exit(0)


def print_round_header(round_num: int) -> None:
    bar = "─" * 52
    print(f"\n{bar}")
    print(f"  {BOLD}Round {round_num}{RESET}")
    print(bar)


def print_round_result(
    choices:   Dict[str, int],
    average:   float,
    target:    float,
    winner_id: str | None,
    scores:    Dict[str, int],
    human_id:  str,
    alive:     list[str],
) -> None:
    print(f"\n  All choices this round:")
    for agent, num in sorted(choices.items()):
        tag = f" {BOLD}← YOU{RESET}" if agent == human_id else ""
        win_tag = f"  {GREEN}✓ wins round{RESET}" if agent == winner_id else ""
        eliminated_tag = "" if agent in alive else f"  {RED}eliminated{RESET}"
        print(f"    {agent:12s}  {colour_number(num)}{tag}{win_tag}{eliminated_tag}")

    print(f"\n  Average : {average:.2f}   Target (80%) : {BOLD}{target:.2f}{RESET}")

    print(f"\n  Scores after round:")
    for agent, sc in sorted(scores.items(), key=lambda x: -x[1]):
        marker = f" {BOLD}← YOU{RESET}" if agent == human_id else ""
        print(f"    {agent:12s}  {CYAN}{sc}{RESET}{marker}")


def print_final_rankings(
    rankings: list[dict],
    human_id: str,
) -> None:
    bar = "═" * 52
    print(f"\n{bar}")
    print(f"  {BOLD}GAME OVER — Final Rankings{RESET}")
    print(bar)

    medals = ["🥇", "🥈", "🥉"]
    for i, entry in enumerate(rankings):
        pid   = entry["player_id"]
        medal = medals[i] if i < 3 else f"  {i+1}."
        you   = f"  {BOLD}← YOU{RESET}" if pid == human_id else ""
        print(f"  {medal}  {pid:12s}"
              f"  (eliminated round {entry.get('round', 'N/A')}){you}")
    print(bar + "\n")


# ============================================================================
# Main game loop
# ============================================================================

def play(cfg: argparse.Namespace) -> None:
    env = NumberwarsEnv(num_players=cfg.num_players)
    obs_dim = env._compute_obs_size()

    # ── Load bot models ─────────────────────────────────────────────────────
    # All possible agents; human occupies cfg.human_slot, bots take the rest.
    human_id = cfg.human_slot
    bot_ids  = [a for a in env.possible_agents if a != human_id]

    print(f"\n{BOLD}Loading {NUM_BOTS} bot models …{RESET}")
    bot_nets: Dict[str, PolicyNet] = {}
    for bot in bot_ids:
        fname = f"{cfg.model_prefix}_{bot}.pt"
        try:
            net = PolicyNet(obs_dim, N_ACTIONS)
            net.load_state_dict(torch.load(fname, map_location="cpu"))
            net.eval()
            bot_nets[bot] = net
            print(f"  ✓  {fname}")
        except FileNotFoundError:
            print(f"\n{RED}Error: could not find '{fname}'.{RESET}")
            print("  Train first with:  python train.py --num_players 6 --save")
            print("  Or pass a custom prefix:  python play.py --model_prefix <prefix>\n")
            sys.exit(1)

    print(f"\n{GREEN}All bots loaded!{RESET}  You are {BOLD}{human_id}{RESET}.\n")
    input("Press Enter to start …")

    # ── Reset ────────────────────────────────────────────────────────────────
    obs_dict, _ = env.reset(seed=cfg.seed)
    last_obs: Dict[str, np.ndarray] = dict(obs_dict)
    # ── Game loop ────────────────────────────────────────────────────────────
    while env.agents:
        round_num = env._state.current_round + 1
        print_round_header(round_num)

        # Collect actions
        actions: Dict[str, int] = {}
        hiddens = {bot: None for bot in bot_ids}  # RNN hidden states per bot

        # Bots choose silently
        for bot in bot_ids:
            if bot not in env.agents:
                continue
            obs   = last_obs[bot]
            x     = torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0)
            with torch.no_grad():
                logits, _, hidden = bot_nets[bot](x, hiddens[bot])
            action = torch.argmax(logits).item()   # greedy at play time
            actions[bot] = int(action)
            hiddens[bot] = hidden

        # Human chooses
        if human_id in env.agents:
            human_player = env._state.players.get(human_id)
            human_score  = human_player.score if human_player else 0
            actions[human_id] = ask_number(human_id, human_score, env.agents)
        else:
            # Human was already eliminated — just watch bots finish
            pass

        # Step env
        next_obs_dict, rewards, terminations, truncations, infos = env.step(actions)

        # Pull result info from any agent's info dict
        sample_info = next(iter(infos.values())) if infos else {}
        average     = sample_info.get("average", 0.0)
        target      = sample_info.get("target",  0.0)
        winner_id   = sample_info.get("winner")
        scores_after: Dict[str, int] = {a: infos[a]["score"] for a in infos}

        # Show what everyone picked
        print_round_result(
            choices   = actions,
            average   = average,
            target    = target,
            winner_id = winner_id,
            scores    = scores_after,
            human_id  = human_id,
            alive     = env.agents,
        )

        last_obs.update(next_obs_dict)

        # Notify if human was just eliminated
        if terminations.get(human_id) and human_id in actions:
            print(f"\n  {RED}{BOLD}You were eliminated this round!{RESET}"
                  f"  Watching the remaining bots finish …")

    # ── Rankings ─────────────────────────────────────────────────────────────
    from .game_rules import GameRules
    rankings = GameRules.get_rankings(env._state)
    print_final_rankings(rankings, human_id)

    # Personal outcome
    human_rank = next(
        (i + 1 for i, r in enumerate(rankings) if r["player_id"] == human_id),
        NUM_PLAYERS,
    )
    if human_rank == 1:
        print(f"{GREEN}{BOLD}You won! 🎉{RESET}\n")
    elif human_rank <= 3:
        print(f"{YELLOW}Top 3 — well played!{RESET}\n")
    else:
        print(f"{DIM}Better luck next time.{RESET}\n")


# ============================================================================
# CLI
# ============================================================================

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Play NumberWars against 5 trained PPO bots."
    )
    p.add_argument(
        "--model_prefix", type=str, default="model",
        help="Prefix used when saving models (default: 'model' → model_player_N.pt)",
    )
    p.add_argument(
        "--human_slot", type=str, default="player_0",
        choices=[f"player_{i}" for i in range(NUM_PLAYERS)],
        help="Which agent slot you occupy (default: player_0)",
    )
    p.add_argument(
        "--num_players", type=int, default=NUM_PLAYERS,
        help=f"Total number of players in the game (default: {NUM_PLAYERS})",
    )
    p.add_argument(
        "--seed", type=int, default=None,
        help="Optional RNG seed for reproducible games",
    )
    return p.parse_args()


if __name__ == "__main__":
    play(parse_args())
