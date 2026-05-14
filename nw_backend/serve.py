from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, field_validator
from typing import Optional
import torch
import numpy as np
import argparse

from .env import NumberwarsEnv
from .train import PolicyNet, load_models

# ── Constants ────────────────────────────────────────────────────────────────
N_ACTIONS   = 101
MODEL_PREFIX = "model"
HISTORY_LEN = 10

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Game session ──────────────────────────────────────────────────────────────
class GameSession:
    def __init__(self, num_players: int):
        self.num_players = num_players
        self.env         = NumberwarsEnv(num_players=num_players)
        self.bot_id      = f"player_{BOT_NUM}" if BOT_NUM >= 0 else f"player_{num_players - 1}"
        self.bot_net     = self._load_bot(self.bot_id, self.env._compute_obs_size())
        self.hidden = self.bot_net.init_hidden()
        obs_dict, _      = self.env.reset()
        self.last_obs: dict[str, np.ndarray] = dict(obs_dict)
        self.game_over   = False

        # Bot decides its move immediately after reset, before humans submit
        self.pending_bot_move: Optional[int] = self._bot_decide()

        print(f"GameSession initialized with {num_players} players. Bot ID: {self.bot_id}")

    def _load_bot(self, bot_id: str, obs_dim: int) -> "PolicyNet":  # fix: added self
        nets = load_models([bot_id], obs_dim)
        return nets[bot_id]

    def _bot_decide(self) -> Optional[int]:
        if self.bot_id not in self.env.agents:
            return None
        obs = self.last_obs[self.bot_id]
        with torch.no_grad():
            a, _, _, self.hidden = self.bot_net.act(obs, self.hidden)
        return a

    def step(self, human_moves: list[int]) -> dict:
        """
        human_moves: one move per human player (indices 0..num_players-2).
        Uses the pre-computed pending_bot_move, then immediately decides
        the bot's move for the *next* round.
        """
        if self.game_over:
            raise ValueError("Game is already over.")

        actions: dict[str, int] = {}

        for i, move in enumerate(human_moves):
            agent = f"player_{i}"
            if move is not None and agent in self.env.agents:
                actions[agent] = int(move)

        # Use the pre-committed bot move
        if self.bot_id in self.env.agents and self.pending_bot_move is not None:
            actions[self.bot_id] = self.pending_bot_move

        committed_bot_move = self.pending_bot_move

        # Step environment
        next_obs, rewards, terminations, truncations, infos = self.env.step(actions)
        self.last_obs.update(next_obs)
        self.game_over = len(self.env.agents) == 0

        # Bot immediately decides for the next round (if still alive)
        self.pending_bot_move = self._bot_decide() if not self.game_over else None

        sample_info = next(iter(infos.values())) if infos else {}

        return {
            "bot_move":       committed_bot_move,
            "next_bot_move":  self.pending_bot_move,   # bot's move for the upcoming round
            "average":        sample_info.get("average"),
            "target":         sample_info.get("target"),
            "winner_id":      sample_info.get("winner"),
            "round":          sample_info.get("round"),
            "scores":         {a: infos[a]["score"] for a in infos},
            "eliminated":     [a for a in infos if infos[a]["eliminated"]],
            "alive":          list(self.env.agents),
            "game_over":      self.game_over,
        }


# ── Global session ────────────────────────────────────────────────────────────
_session: Optional[GameSession] = None


# ── Schemas ───────────────────────────────────────────────────────────────────
class NewGameRequest(BaseModel):
    num_players: int

class NewGameResponse(BaseModel):
    message:      str
    agents:       list[str]
    human_slots:  list[str]
    first_bot_move: Optional[int]   # bot's move for round 1, ready immediately

class TurnRequest(BaseModel):
    round: int
    moves: list[Optional[int]]

    @field_validator('moves', mode='before')
    @classmethod
    def empty_str_to_none(cls, v):
        return [None if item == "" else item for item in v]

class TurnResponse(BaseModel):
    bot_move:      Optional[int]   # the move the bot played this round
    next_bot_move: Optional[int]   # bot's move already chosen for next round
    average:       Optional[float]
    target:        Optional[float]
    winner_id:     Optional[str]
    round:         Optional[int]
    scores:        dict[str, int]
    eliminated:    list[str]
    alive:         list[str]
    game_over:     bool


# ── Endpoints ─────────────────────────────────────────────────────────────────
@app.post("/new-game", response_model=NewGameResponse)
async def new_game(body: NewGameRequest):
    global _session

    if body.num_players < 2:
        raise HTTPException(status_code=400, detail="num_players must be >= 2")

    _session = GameSession(num_players=body.num_players)

    if _session.bot_net is None:
        raise HTTPException(status_code=500, detail="Bot model not loaded.")

    # fix: use session's bot_id, not hardcoded BOT_ID
    human_slots = [a for a in _session.env.possible_agents if a != _session.bot_id]

    return NewGameResponse(
        message       = f"New game started with {body.num_players} players.",
        agents        = list(_session.env.possible_agents),
        human_slots   = human_slots,
        first_bot_move = _session.pending_bot_move,   # populated, not missing anymore
    )


@app.post("/submit-turn", response_model=TurnResponse)
async def submit_turn(body: TurnRequest):
    global _session

    if _session is None:
        raise HTTPException(status_code=400, detail="No active game. POST /new-game first.")
    if _session.game_over:
        raise HTTPException(status_code=400, detail="Game is over. POST /new-game to start again.")
    if _session.env._state.current_round != body.round:
        raise HTTPException(
            status_code=400,
            detail=f"Round mismatch. Expected round {_session.env._state.current_round}."
        )

    result = _session.step(body.moves)
    return TurnResponse(**result)


@app.get("/state")
async def get_state():
    if _session is None:
        return {"active": False}
    return {
        "active":          not _session.game_over,
        "alive":           list(_session.env.agents),
        "round":           _session.env._state.current_round if _session.env._state else None,
        "game_over":       _session.game_over,
        "pending_bot_move": _session.pending_bot_move,
    }


# ── Entrypoint ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn

    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--bot_num", type=int, default=-1, help="Which bot model to load (e.g. 0 for model_0.pt)")
    
    global BOT_NUM
    BOT_NUM = parser.parse_args().bot_num

    args = parser.parse_args()

    uvicorn.run(app, host=args.host, port=args.port)
