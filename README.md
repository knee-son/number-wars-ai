# Number Wars AI

A multi-agent reinforcement learning project for a turn-based imperfect information game.

## Project overview

- Game type: turn-based, imperfect information
- Players: 2 or more
- Observation parameters: number of players, previous rounds' choices as a list of integers
- Goal: build a custom environment and train multi-agent policies using RLlib

## Files

- `requirements.txt` - Python dependencies
- `README.md` - project description and setup notes
- `src/__init__.py` - package marker
- `src/game_rules.py` - core game logic and rules
- `src/env.py` - Gymnasium-style multi-agent environment wrapper
- `src/train.py` - training entrypoint using Ray RLlib
- `tests/test_env.py` - baseline environment unit tests
- `.gitignore` - ignores Python build artifacts

## Setup

1. Create a virtual environment using your preferred Python version.
2. Install dependencies:

```bash
python -m pip install -r requirements.txt
```

## Next step

Implement the game-specific rule parser and RLlib training configuration.
