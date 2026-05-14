"""
train.py
========
Trains independent PPO agents on NumberwarsEnv using an LSTM-based
actor-critic policy.

Each agent ("player_0", "player_1", …) has its own LSTM policy network.
Hidden state is carried across timesteps within an episode and reset at
the start of each new episode.

Install deps:
    pip install pettingzoo gymnasium numpy torch

Usage:
    python -m nw_backend.train                       # defaults: 4 players, 2 000 episodes
    python -m nw_backend.train --num_players 10 --episodes 2000 --lr 3e-4 --save

    # Seed all agents from player_3's checkpoint before training
    python -m nw_backend.train --seed_from 3 --load_prefix model --save --save_prefix model_v2

    # Seed agents from a pool — each agent randomly gets one of these checkpoints
    python -m nw_backend.train --seed_from 0 3 5 --load_prefix model --save --save_prefix model_v2
"""

from __future__ import annotations

import argparse
import json
import random
import time
from collections import defaultdict, deque
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical

from .env import NumberwarsEnv


# ============================================================================
# Hyper-parameters
# ============================================================================

DEFAULTS = dict(
    num_players   = 4,
    episodes      = 2_000,
    lr            = 3e-4,
    gamma         = 0.99,
    gae_lambda    = 0.95,
    clip_eps      = 0.2,
    entropy_coef  = 0.05,
    value_coef    = 1.0,
    ppo_epochs    = 4,
    batch_size    = 64,   # sequence chunk length for LSTM replay
    lstm_hidden   = 128,
    log_interval  = 50,
    seed          = 42,
)

# Convenience type for an LSTM hidden state
HiddenState = Tuple[torch.Tensor, torch.Tensor]   # (h, c) each (1, 1, hidden)


# ============================================================================
# Neural network — LSTM actor-critic
# ============================================================================

class PolicyNet(nn.Module):
    """
    LSTM Actor-Critic.

    Architecture:
        obs  →  Linear(obs_dim, hidden)  →  Tanh
             →  LSTM(hidden, hidden)
             →  actor head  (logits over 101 actions)
             →  critic head (scalar value)

    Hidden state (h, c) is maintained externally so the caller can
    carry it across timesteps and reset it between episodes.
    """

    def __init__(self, obs_dim: int, n_actions: int, hidden: int = 128):
        super().__init__()
        self.hidden_size = hidden

        self.encoder = nn.Sequential(
            nn.Linear(obs_dim, hidden),
            nn.Tanh(),
        )
        self.lstm   = nn.LSTM(hidden, hidden, batch_first=True)
        self.actor  = nn.Linear(hidden, n_actions)
        self.critic = nn.Linear(hidden, 1)

    # ------------------------------------------------------------------
    # Core forward: processes a (batch, seq_len, obs_dim) tensor
    # ------------------------------------------------------------------
    def forward(
        self,
        x:      torch.Tensor,           # (B, T, obs_dim)
        hidden: Optional[HiddenState],  # None → zero-init
    ) -> Tuple[torch.Tensor, torch.Tensor, HiddenState]:
        """
        Returns:
            logits : (B, T, n_actions)
            values : (B, T)
            hidden : updated (h, c)
        """
        enc           = self.encoder(x)               # (B, T, hidden)
        lstm_out, hx  = self.lstm(enc, hidden)        # (B, T, hidden)
        logits        = self.actor(lstm_out)           # (B, T, n_actions)
        values        = self.critic(lstm_out).squeeze(-1)  # (B, T)
        return logits, values, hx

    # ------------------------------------------------------------------
    # Single-step convenience used during rollout collection
    # ------------------------------------------------------------------
    def act(
        self,
        obs:    np.ndarray,
        hidden: Optional[HiddenState],
    ) -> Tuple[int, float, float, HiddenState]:
        """
        obs    : 1-D numpy array (obs_dim,)
        hidden : previous (h, c) or None

        Returns (action, log_prob, value, new_hidden)
        """
        x = torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        # x shape: (1, 1, obs_dim)
        with torch.no_grad():
            logits, values, new_hx = self.forward(x, hidden)
        dist   = Categorical(logits=logits[0, 0])
        action = dist.sample()
        return (
            action.item(),
            dist.log_prob(action).item(),
            values[0, 0].item(),
            new_hx,
        )

    # ------------------------------------------------------------------
    def init_hidden(self) -> HiddenState:
        """Returns a zero hidden state (1, 1, hidden_size)."""
        z = torch.zeros(1, 1, self.hidden_size)
        return (z, z)


# ============================================================================
# Rollout buffer — stores one full episode per agent
# ============================================================================

class RolloutBuffer:
    """
    Stores one episode of transitions for a single agent.

    Compared with the MLP version we additionally store the LSTM hidden
    state at each step so we can re-initialise the LSTM correctly when
    replaying the sequence during the PPO update.
    """

    def __init__(self):
        self.obs:       List[np.ndarray]  = []
        self.actions:   List[int]         = []
        self.log_probs: List[float]       = []
        self.rewards:   List[float]       = []
        self.values:    List[float]       = []
        self.dones:     List[bool]        = []
        # (h, c) tensors stored as cpu tensors, shape (1,1,hidden)
        self.hiddens:   List[HiddenState] = []

    def store(
        self,
        obs:      np.ndarray,
        action:   int,
        log_prob: float,
        reward:   float,
        value:    float,
        done:     bool,
        hidden:   HiddenState,
    ) -> None:
        self.obs.append(obs)
        self.actions.append(action)
        self.log_probs.append(log_prob)
        self.rewards.append(reward)
        self.values.append(value)
        self.dones.append(done)
        # detach and move to cpu so we don't hold computation graphs
        self.hiddens.append(
            (hidden[0].detach().cpu(), hidden[1].detach().cpu())
        )

    def clear(self):
        self.__init__()

    def __len__(self):
        return len(self.rewards)


# ============================================================================
# GAE
# ============================================================================

def compute_gae(
    rewards: List[float],
    values:  List[float],
    dones:   List[bool],
    gamma:   float,
    lam:     float,
) -> Tuple[torch.Tensor, torch.Tensor]:
    advantages = []
    gae        = 0.0
    next_value = 0.0
    for r, v, d in zip(reversed(rewards), reversed(values), reversed(dones)):
        delta = r + gamma * next_value * (1 - d) - v
        gae   = delta + gamma * lam * (1 - d) * gae
        advantages.insert(0, gae)
        next_value = v
    advantages = torch.tensor(advantages, dtype=torch.float32)
    returns    = advantages + torch.tensor(values, dtype=torch.float32)
    return advantages, returns


# ============================================================================
# PPO update — replays sequences through the LSTM
# ============================================================================

def ppo_update(
    net:       PolicyNet,
    optimizer: optim.Optimizer,
    buffer:    RolloutBuffer,
    cfg:       argparse.Namespace,
) -> Tuple[float, float, float]:
    """
    Replays the stored episode in contiguous chunks of length
    cfg.batch_size, re-running the LSTM from the stored hidden state
    at the start of each chunk.  This is the "truncated BPTT" approach
    commonly used with recurrent PPO.
    """
    if len(buffer) == 0:
        return 0.0, 0.0, 0.0

    T = len(buffer)
    obs_arr    = np.array(buffer.obs)                               # (T, obs_dim)
    acts_t     = torch.tensor(buffer.actions,   dtype=torch.long)  # (T,)
    old_lp_t   = torch.tensor(buffer.log_probs, dtype=torch.float32)

    advs, rets = compute_gae(
        buffer.rewards, buffer.values, buffer.dones,
        cfg.gamma, cfg.gae_lambda,
    )
    advs = (advs - advs.mean()) / (advs.std() + 1e-8)

    total_pl = total_vl = total_ent = 0.0
    update_count = 0

    chunk = cfg.batch_size   # sequence length per chunk

    for _ in range(cfg.ppo_epochs):
        # iterate over contiguous chunks (no shuffling — order matters for LSTM)
        for start in range(0, T, chunk):
            end = min(start + chunk, T)
            idx = slice(start, end)

            # Re-initialise LSTM from the stored hidden state at chunk start
            h0, c0 = buffer.hiddens[start]
            hidden  = (h0.clone(), c0.clone())

            # obs tensor: (1, chunk_len, obs_dim)
            x = torch.tensor(obs_arr[idx], dtype=torch.float32).unsqueeze(0)

            logits, values, _ = net(x, hidden)
            # logits: (1, chunk_len, n_actions)  values: (1, chunk_len)
            logits = logits.squeeze(0)   # (chunk_len, n_actions)
            values = values.squeeze(0)   # (chunk_len,)

            dist    = Categorical(logits=logits)
            new_lp  = dist.log_prob(acts_t[idx])
            entropy = dist.entropy().mean()

            ratio  = (new_lp - old_lp_t[idx]).exp()
            surr1  = ratio * advs[idx]
            surr2  = ratio.clamp(1 - cfg.clip_eps, 1 + cfg.clip_eps) * advs[idx]
            pl     = -torch.min(surr1, surr2).mean()
            vl     = nn.functional.mse_loss(values, rets[idx])
            loss   = pl + cfg.value_coef * vl - cfg.entropy_coef * entropy

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(net.parameters(), 0.5)
            optimizer.step()

            total_pl  += pl.item()
            total_vl  += vl.item()
            total_ent += entropy.item()
            update_count += 1

    denom = max(update_count, 1)
    return total_pl / denom, total_vl / denom, total_ent / denom


# ============================================================================
# Training loop
# ============================================================================

def train(cfg: argparse.Namespace) -> Dict[str, PolicyNet]:
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)

    env = NumberwarsEnv(num_players=cfg.num_players)

    obs_dim   = env._compute_obs_size()
    n_actions = 101

    nets: Dict[str, PolicyNet] = {
        agent: PolicyNet(obs_dim, n_actions, hidden=cfg.lstm_hidden)
        for agent in env.possible_agents
    }
    log: str = ""

    # ------------------------------------------------------------------
    # Optionally seed agents from a pool of checkpoints.
    # Each agent is assigned one checkpoint from the pool at random
    # (with replacement), so the starting population is diverse.
    #
    # Examples:
    #   --seed_from 5          → all agents get player_5's weights (original behaviour)
    #   --seed_from 0 3 5      → each agent randomly draws from {0, 3, 5}
    # ------------------------------------------------------------------
    if cfg.seed_from is not None:
        pool = cfg.seed_from  # list of ints

        # Pre-load each unique checkpoint once to avoid redundant disk reads
        pool_states = {}
        for idx in set(pool):
            fname = f"{cfg.load_prefix}_player_{idx}.pt"
            pool_states[idx] = torch.load(fname, map_location="cpu")
            print(f"  Loaded checkpoint: '{fname}'")

        print(f"\nSeeding {len(env.possible_agents)} agents from pool {pool} ...")
        for agent in env.possible_agents:
            chosen = random.choice(pool)
            nets[agent].load_state_dict(pool_states[chosen])
            nets[agent].train()
            print(f"  {agent}  ←  player_{chosen}")
        print()

    optimizers: Dict[str, optim.Optimizer] = {
        agent: optim.Adam(nets[agent].parameters(), lr=cfg.lr)
        for agent in env.possible_agents
    }
    buffers: Dict[str, RolloutBuffer] = {
        agent: RolloutBuffer() for agent in env.possible_agents
    }

    reward_window: Dict[str, deque] = {
        a: deque(maxlen=cfg.log_interval) for a in env.possible_agents
    }
    ep_len_window: deque = deque(maxlen=cfg.log_interval)
    policy_loss_window  = {agent: deque(maxlen=cfg.log_interval) for agent in env.possible_agents}
    value_loss_window   = {agent: deque(maxlen=cfg.log_interval) for agent in env.possible_agents}
    entropy_window      = {agent: deque(maxlen=cfg.log_interval) for agent in env.possible_agents}

    print(f"\n{'='*60}")
    print(f"  NumberWars — Independent PPO + LSTM")
    print(f"  Players    : {cfg.num_players}   |  obs_dim    : {obs_dim}")
    print(f"  LSTM hidden: {cfg.lstm_hidden}   |  lr         : {cfg.lr}")
    print(f"  Episodes   : {cfg.episodes}")
    if cfg.seed_from is not None:
        print(f"  Seed pool    : {cfg.seed_from}  (prefix: {cfg.load_prefix})")
    print(f"{'='*60}\n")

    t_start = time.time()

    for episode in range(1, cfg.episodes + 1):

        obs_dict, _ = env.reset(seed=None)
        last_obs: Dict[str, np.ndarray] = dict(obs_dict)

        # Each agent starts the episode with a zeroed LSTM hidden state
        hiddens: Dict[str, HiddenState] = {
            agent: nets[agent].init_hidden()
            for agent in env.possible_agents
        }

        ep_rewards: Dict[str, float] = defaultdict(float)
        round_count = 0

        # ----------------------------------------------------------------
        # Rollout
        # ----------------------------------------------------------------
        while env.agents:
            actions:   Dict[str, int]         = {}
            log_probs: Dict[str, float]       = {}
            values_:   Dict[str, float]       = {}
            new_hx:    Dict[str, HiddenState] = {}

            for agent in env.agents:
                a, lp, v, hx = nets[agent].act(last_obs[agent], hiddens[agent])
                actions[agent]   = a
                log_probs[agent] = lp
                values_[agent]   = v
                new_hx[agent]    = hx

            next_obs_dict, rewards, terminations, truncations, _ = env.step(actions)

            for agent in list(actions.keys()):
                done = terminations.get(agent, False) or truncations.get(agent, False)
                buffers[agent].store(
                    obs      = last_obs[agent],
                    action   = actions[agent],
                    log_prob = log_probs[agent],
                    reward   = rewards.get(agent, 0.0),
                    value    = values_[agent],
                    done     = done,
                    hidden   = hiddens[agent],   # hidden *before* this step
                )
                ep_rewards[agent] += rewards.get(agent, 0.0)

            # Advance hidden states for surviving agents
            for agent in env.agents:
                if agent in new_hx:
                    hiddens[agent] = new_hx[agent]

            last_obs.update(next_obs_dict)
            round_count += 1

        ep_len_window.append(round_count)

        # ----------------------------------------------------------------
        # PPO update
        # ----------------------------------------------------------------
        for agent in env.possible_agents:
            if len(buffers[agent]) == 0:
                continue

            pl, vl, ent = ppo_update(nets[agent], optimizers[agent], buffers[agent], cfg)
            policy_loss_window[agent].append(pl)
            value_loss_window[agent].append(vl)
            entropy_window[agent].append(ent)
            buffers[agent].clear()

        for agent in env.possible_agents:
            reward_window[agent].append(ep_rewards.get(agent, 0.0))

        # ----------------------------------------------------------------
        # Logging
        # ----------------------------------------------------------------
        if episode % cfg.log_interval == 0:
            elapsed = time.time() - t_start
            avg_len = np.mean(ep_len_window)
            log = (f"Episode {episode:>6d}/{cfg.episodes}  |  "
                   f"avg_rounds={avg_len:.1f}  |  "
                   f"elapsed={elapsed:.1f}s\n")
            for agent in env.possible_agents:
                avg_r   = np.mean(reward_window[agent])      if reward_window[agent]      else 0.0
                avg_pl  = np.mean(policy_loss_window[agent]) if policy_loss_window[agent] else 0.0
                avg_vl  = np.mean(value_loss_window[agent])  if value_loss_window[agent]  else 0.0
                avg_ent = np.mean(entropy_window[agent])     if entropy_window[agent]     else 0.0
                log += (f"    {agent}  avg_reward={avg_r:+.4f}  "
                        f"pl={avg_pl:.4f}  vl={avg_vl:.4f}  ent={avg_ent:.4f}\n")
            print(log)

    print("Training complete.")
    return nets, log


# ============================================================================
# Save / load helpers
# ============================================================================

def save_models(nets: Dict[str, PolicyNet], path_prefix: str = "model") -> None:
    for agent, net in nets.items():
        fname = f"{path_prefix}_{agent}.pt"
        torch.save(net.state_dict(), fname)
        print(f"  Saved {fname}")


def load_models(
    possible_agents: List[str],
    obs_dim:         int,
    n_actions:       int = 101,
    lstm_hidden:     int = 128,
    path_prefix:     str = "model",
) -> Dict[str, PolicyNet]:
    nets = {}
    for agent in possible_agents:
        fname = f"{path_prefix}_{agent}.pt"
        net   = PolicyNet(obs_dim, n_actions, hidden=lstm_hidden)
        net.load_state_dict(torch.load(fname, map_location="cpu"))
        net.eval()
        nets[agent] = net
        print(f"  Loaded {fname}")
    return nets


# ============================================================================
# CLI
# ============================================================================

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Train independent PPO+LSTM agents on NumberwarsEnv"
    )
    for key, val in DEFAULTS.items():
        p.add_argument(f"--{key}", type=type(val), default=val)

    # Checkpoint seeding
    p.add_argument(
        "--seed_from",
        type=int,
        nargs="+",
        default=None,
        metavar="N",
        help="Seed agents from a pool of player checkpoints. "
             "A single value seeds all agents identically (original behaviour). "
             "Multiple values randomly assign one checkpoint per agent. "
             "e.g. --seed_from 0 3 5  draws from {player_0, player_3, player_5}.",
    )
    p.add_argument(
        "--load_prefix",
        type=str,
        default="model",
        help="Prefix for checkpoint files used with --seed_from (default: model)",
    )

    p.add_argument("--save",        action="store_true")
    p.add_argument("--save_prefix", type=str, default="model")
    return p.parse_args()


if __name__ == "__main__":
    cfg: argparse.Namespace = parse_args()
    nets, log = train(cfg)

    if cfg.save:
        print("\nSaving training log …")
        with open(f"{cfg.save_prefix}_training_log.txt", "w") as f:
            json.dump(cfg.__dict__, f, indent=2)
            f.write("\n\n")
            f.write(log)

        print("\nSaving models …")
        save_models(nets, path_prefix=cfg.save_prefix)
