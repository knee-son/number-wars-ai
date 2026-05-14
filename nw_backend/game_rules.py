"""
game_rules.py
=============
Pure-logic port of the "Closest to 80% of Average" multiplayer game.
Designed to be MARL-friendly:
  - No I/O, no sockets, no side-effects.
  - All state lives in GameState (a plain dataclass).
  - A single step() call advances the game by one round.
  - Observations, rewards, and done-flags are returned as dicts keyed by player_id.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional
import math

# ---------------------------------------------------------------------------
# Constants (mirrors the JS server)
# ---------------------------------------------------------------------------

MAX_PLAYERS: int = 10
INITIAL_SCORE: int = 10
GAME_OVER_POINTS: int = 0          # eliminated when score reaches this


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class Player:
    player_id: str
    name: str
    score: int = INITIAL_SCORE
    submitted_number: Optional[int] = None
    consecutive_wins: int = 0


@dataclass
class RoundResult:
    """Returned by GameRules.resolve_round() — everything a MARL env needs."""
    round_number: int
    average: float
    target: float                           # average * 0.8
    winner_id: Optional[str]                # None if no valid winner
    rewards: dict[str, float]               # player_id -> scalar reward
    eliminated_ids: list[str]               # players removed this round
    scores_before: dict[str, int]           # snapshot before score updates
    scores_after: dict[str, int]
    info: dict                              # diagnostics / rule flags


@dataclass
class GameState:
    players: dict[str, Player] = field(default_factory=dict)
    current_round: int = 0
    elimination_order: list[dict] = field(default_factory=list)
    game_over: bool = False
    last_eliminated_count: int = 0


# ---------------------------------------------------------------------------
# Pure rule logic
# ---------------------------------------------------------------------------

class GameRules:
    """
    Stateless helper that operates on a GameState.
    Call in order:
        1. add_player(state, ...)          - lobby phase
        2. submit_number(state, id, num)   - collection phase
        3. result = resolve_round(state)   - end-of-round phase
        4. check game_over via result or state.game_over
    """

    # ------------------------------------------------------------------
    # Lobby
    # ------------------------------------------------------------------

    @staticmethod
    def add_player(state: GameState, player_id: str, name: str) -> dict:
        """
        Returns {"success": bool, "message": str}.
        Mirrors game.addPlayer() — same validation guards.
        """
        if state.game_over:
            return {"success": False, "message": "Game is already over."}
        if len(state.players) >= MAX_PLAYERS:
            return {"success": False, "message": "Game is full."}
        if any(p.name.lower() == name.lower() for p in state.players.values()):
            return {"success": False, "message": "Username already taken."}
        if player_id in state.players:
            return {"success": False, "message": "Player ID already registered."}

        state.players[player_id] = Player(player_id=player_id, name=name)
        return {"success": True, "message": ""}

    @staticmethod
    def remove_player(state: GameState, player_id: str) -> None:
        state.players.pop(player_id, None)

    # ------------------------------------------------------------------
    # Submission
    # ------------------------------------------------------------------

    @staticmethod
    def submit_number(state: GameState, player_id: str, number: int) -> bool:
        """Validates and records a player's number. Returns True on success."""
        if player_id not in state.players:
            return False
        if not isinstance(number, int) or isinstance(number, bool):
            return False
        if not (0 <= number <= 100):
            return False
        state.players[player_id].submitted_number = number
        return True

    @staticmethod
    def all_submitted(state: GameState) -> bool:
        return all(p.submitted_number is not None for p in state.players.values())

    # ------------------------------------------------------------------
    # Round resolution  (mirrors calculateWinner + checkGameOver)
    # ------------------------------------------------------------------

    @staticmethod
    def resolve_round(state: GameState) -> RoundResult:
        """
        Applies all game rules for the current round in order:
          1. Penalise non-submitters  (-2 pts)
          2. Calculate average & target (avg * 0.8)
          3. Apply player-count-based special rules:
               2 players : instant-win rule (0 vs 100)
               3 players : exact-match double-penalty rule
               4 players : duplicate-number rule
          4. Normal closest-wins logic with duplicate penalties
          5. Eliminate players at 0 score
          6. Detect game-over

        Returns a RoundResult with per-agent rewards suitable for MARL training.
        """
        state.current_round += 1
        n_players = len(state.players)

        # Snapshot scores before any mutation
        scores_before = {pid: p.score for pid, p in state.players.items()}

        info: dict = {
            "rule_triggered": None,
            "duplicate_ids": [],
            "non_submitters": [],
        }

        # ---- 1. Penalise non-submitters --------------------------------
        for pid, player in state.players.items():
            if player.submitted_number is None:
                player.score = max(0, player.score - 2)
                player.consecutive_wins = 0
                info["non_submitters"].append(pid)

        # ---- 2. Average & target ----------------------------------------
        submitted = {
            pid: p.submitted_number
            for pid, p in state.players.items()
            if p.submitted_number is not None
        }

        if submitted:
            average = sum(submitted.values()) / len(submitted)
        else:
            average = 0.0
        target = average * 0.8

        winner_id: Optional[str] = None

        # ---- 3a. 2-player instant-win rule ------------------------------
        if n_players == 2:
            pids = list(state.players.keys())
            p1, p2 = state.players[pids[0]], state.players[pids[1]]
            n1, n2 = p1.submitted_number, p2.submitted_number
            if n1 is not None and n2 is not None:
                if (n1 == 0 and n2 == 100) or (n1 == 100 and n2 == 0):
                    # Player who chose 100 wins; loser loses all points
                    if n1 == 100:
                        winner_id = pids[0]
                        p2.score = 0
                    else:
                        winner_id = pids[1]
                        p1.score = 0
                    state.players[winner_id].consecutive_wins += 1
                    info["rule_triggered"] = "2player_instant_win"

        # ---- 3b. 3-player exact-match rule ------------------------------
        if n_players == 3 and winner_id is None:
            for pid, player in state.players.items():
                if player.submitted_number is not None and math.isclose(
                    player.submitted_number, target, abs_tol=1e-9
                ):
                    winner_id = pid
                    # Double penalty for all losers
                    for opid, op in state.players.items():
                        if opid != pid:
                            op.score = max(0, op.score - 2)
                    player.consecutive_wins += 1
                    info["rule_triggered"] = "3player_exact_match"
                    break

        # ---- 3c. 4-player duplicate-number rule -------------------------
        if n_players == 4 and winner_id is None:
            numbers_list = [
                p.submitted_number
                for p in state.players.values()
                if p.submitted_number is not None
            ]
            dup_values = {
                num for num in numbers_list if numbers_list.count(num) > 1
            }

            if dup_values:
                dup_ids = [
                    pid
                    for pid, p in state.players.items()
                    if p.submitted_number in dup_values
                ]
                for pid in dup_ids:
                    state.players[pid].score = max(0, state.players[pid].score - 1)
                info["duplicate_ids"] = dup_ids
                info["rule_triggered"] = "4player_duplicate_penalty"

                # Winner is closest non-duplicate
                best_diff = math.inf
                for pid, player in state.players.items():
                    if pid not in dup_ids and player.submitted_number is not None:
                        diff = abs(player.submitted_number - target)
                        if diff < best_diff:
                            best_diff = diff
                            winner_id = pid
                if winner_id:
                    state.players[winner_id].consecutive_wins += 1

        # ---- 4. Normal closest-wins logic (with global duplicate check) -
        if winner_id is None:
            # Identify ALL duplicate submitters across all players
            all_numbers = [
                p.submitted_number
                for p in state.players.values()
                if p.submitted_number is not None
            ]
            dup_values_normal = {
                num for num in all_numbers if all_numbers.count(num) > 1
            }
            dup_ids_normal = [
                pid
                for pid, p in state.players.items()
                if p.submitted_number in dup_values_normal
            ]
            info["duplicate_ids"] = list(set(info["duplicate_ids"] + dup_ids_normal))

            best_diff = math.inf
            for pid, player in state.players.items():
                if pid not in dup_ids_normal and player.submitted_number is not None:
                    diff = abs(player.submitted_number - target)
                    if diff < best_diff:
                        best_diff = diff
                        winner_id = pid

            if winner_id:
                # Update scores for all players
                for pid, player in state.players.items():
                    if pid == winner_id:
                        player.consecutive_wins += 1
                    else:
                        if pid in dup_ids_normal:
                            player.score = max(0, player.score - 2)
                        else:
                            player.score = max(0, player.score - 1)
                        player.consecutive_wins = 0
            else:
                # No valid winner (everyone chose duplicates) — all lose 2
                for player in state.players.values():
                    player.score = max(0, player.score - 2)
                    player.consecutive_wins = 0

        # ---- 5. Reset submissions for next round -------------------------
        for player in state.players.values():
            player.submitted_number = None

        # ---- 6. Eliminate players at/below GAME_OVER_POINTS --------------
        eliminated_ids: list[str] = []
        for pid, player in list(state.players.items()):
            if player.score <= GAME_OVER_POINTS:
                state.elimination_order.append({
                    "player_id": pid,
                    "name": player.name,
                    "round": state.current_round,
                })
                eliminated_ids.append(pid)
                del state.players[pid]

        state.last_eliminated_count += len(eliminated_ids)

        # ---- 7. Game-over check ------------------------------------------
        if len(state.players) <= 1:
            state.game_over = True

        # ---- 8. Scores after mutations -----------------------------------
        scores_after = {pid: p.score for pid, p in state.players.items()}
        # Include eliminated players at 0
        for pid in eliminated_ids:
            scores_after[pid] = 0

        # ---- 9. Build per-agent rewards (sparse, +1/-1/0 shaped) ---------
        round_num = state.current_round
        rewards: dict[str, float] = {}
        for pid in scores_before:
            if pid == winner_id:
                rewards[pid] = n_players ** 2    # Big reward for winning, scaled by player count
            elif pid in eliminated_ids:
                rewards[pid] = -(n_players ** 2)  # Big penalty for elimination, scaled by player count
            elif pid in info["non_submitters"]:
                rewards[pid] = -0.5
            else:
                rewards[pid] = -1.0 * round_num if scores_after.get(pid, 0) < scores_before[pid] else 0.0
            # rewards[pid] += 1 * scores_after.get(pid, 0)

        return RoundResult(
            round_number=state.current_round,
            average=average,
            target=target,
            winner_id=winner_id,
            rewards=rewards,
            eliminated_ids=eliminated_ids,
            scores_before=scores_before,
            scores_after=scores_after,
            info=info,
        )

    # ------------------------------------------------------------------
    # Observations (for MARL agents)
    # ------------------------------------------------------------------

    @staticmethod
    def get_observation(state: GameState, player_id: str) -> dict:
        """
        Returns a flat dict observation for a single agent.
        Agents see: their own score, num_players alive, current round.
        They do NOT see other agents' numbers (partial observability).
        """
        player = state.players.get(player_id)
        if player is None:
            return {}
        return {
            "player_id": player_id,
            "own_score": player.score,
            "consecutive_wins": player.consecutive_wins,
            "num_players_alive": len(state.players),
            "current_round": state.current_round,
            "game_over": state.game_over,
        }

    @staticmethod
    def get_all_observations(state: GameState) -> dict[str, dict]:
        return {
            pid: GameRules.get_observation(state, pid)
            for pid in state.players
        }

    # ------------------------------------------------------------------
    # Final rankings (mirrors JS gameOver payload)
    # ------------------------------------------------------------------

    @staticmethod
    def get_rankings(state: GameState) -> list[dict]:
        """
        Returns players ranked from 1st (winner) to last (first eliminated).
        """
        rankings: list[dict] = []
        remaining = list(state.players.values())
        if remaining:
            winner = remaining[0]
            rankings.append({
                "place": 1,
                "player_id": winner.player_id,
                "name": winner.name,
                "round": state.current_round,
            })

        for i, entry in enumerate(reversed(state.elimination_order)):
            rankings.append({
                "place": len(rankings) + 1,
                "player_id": entry["player_id"],
                "name": entry["name"],
                "round": entry["round"],
            })

        return rankings
