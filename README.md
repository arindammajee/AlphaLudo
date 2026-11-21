# AlphaLudo

AlphaLudo is a Ludo game engine and a set of baselines for reinforcement-learning research. The name points at an AlphaZero-style agent: search with a learned policy and value, trained by self-play. That agent is not in the repository yet. What is here is a tested engine with two rulesets, a tabular Q-learning baseline, a hand-written heuristic, and a vanilla Monte Carlo tree search.

The engine is a fork of [LUDOpy](https://github.com/SimonLBSoerensen/LUDOpy) 1.3.1 by Simon L. B. Sørensen.

## Licence

Two licences apply, and this repository does not relicense either body of code.

- The engine under `src/ludopy/` is derived from LUDOpy and stays under the MIT licence. The upstream notice is `src/ludopy/LICENSE.MIT` (Copyright (c) 2020 SimonLBS).
- Everything else in the repository, including the experiment scripts, tests, and this README, stays under the GNU GPL-3 in the root `LICENSE` file.

## Layout

The implementation lives in `src/ludopy`. `LUDOpy/ludopy/__init__.py` is a redirect onto that tree, so notebooks that put `LUDOpy` on `sys.path` and scripts that import `ludopy` from `src` load the same code.

| Path | Role |
|---|---|
| `src/ludopy/game.py` | One-die game. `ludopy.Game`. |
| `src/ludopy/game_3dice.py` | Shared three-die queue. `ludopy.Game3Dice`. |
| `src/ludopy/player.py` | Movement rules. |
| `src/ludopy/env.py` | `LudoEnv`: `reset(seed)`, `step(piece)`. |
| `src/Qlearn.py` | Tabular Q-learning baseline. 16 states, 10 actions. |
| `src/VanilaMCTS.py`, `src/randomMCTS.py` | Vanilla MCTS against a random opponent. |
| `3_Dice_Q_Learning_&_Aggressive_Walk_Two_Players.ipynb` | Aggressive heuristic vs Q-learning and vs random, on the three-die game. |
| `tests/` | Rule tests, environment tests, import checks. |
| `docs/paper_plan.md` | Research plan for a benchmark paper. |

## Install and test

```sh
pip install -r requirements.txt
pytest
```

`randomMCTS.py` also imports `wandb` inside its `__main__` block. Install `wandb` only if you run that sweep. Importing the module does not require it.

Python 3.9 or newer. Continuous integration is `.github/workflows/tests.yml`.

## Rulesets

`Game(..., ruleset="classic")` is the default. It follows upstream LUDOpy's Danish rules:

- A piece leaves home only on a 6.
- After round 1, a 6 grants another roll.
- In round 1, while every piece is still at home, upstream's attempt counter keeps the turn for three failed rolls and passes on the next roll. A player who never rolls a 6 therefore rolls four times. That is the upstream condition (`start_attempts < 3`), copied here rather than silently "fixed".
- Landing on a star jumps to the next star. Landing exactly on the star before the goal lane (square 51) enters the goal, and a single enemy on that star is sent home.
- Overshooting the goal bounces back by the extra pips.
- A single enemy on a free square is sent home. Two enemies stacked on a square send the mover home.
- An occupied globe sends the mover home.
- Entering the goal requires an exact count.

`ruleset="legacy"` is the behaviour this fork had before the ruleset switch, which is what the saved notebook numbers were measured on: any face leaves home, a 6 does not grant another roll, there is no opening-attempt rule, stars do not jump (except square 51, which still enters the goal), overshooting the goal leaves the piece where it is, and landing on a star does not capture.

`strategic_player` is accepted and stored. The engine does not move that seat by itself. Older scripts passed the argument and crashed on the one-die `Game`.

`Game3Dice` deals one roll of three faces, always playing the highest face still in the queue, across successive turns. The queue is shared. The agent does not choose a die. `ruleset` still selects movement.

`save_hist` on `Game` writes a `.npz` archive and rejects other extensions. `Game3Dice.save_hist` writes a `.npy` object array. `get_piece_hist` reads the `"pieces"` history key.

## Environment

`LudoEnv` follows the Gymnasium calling convention and does not import Gymnasium or PettingZoo.

```python
from ludopy import LudoEnv

env = LudoEnv(n_agents=2, ruleset="classic", seed=0)
obs, info = env.reset(seed=0)
# obs["die"] is the public roll.
# obs["action_mask"] has shape (4,), one bit per piece.
obs, reward, terminated, truncated, info = env.step(int(obs["action_mask"].argmax()))
```

`n_agents=4` uses every seat. `n_agents=2` uses seats 0 and 2, the pairing the experiment scripts use when they ghost players 1 and 3. `reward` is 1 when the player who just moved wins, otherwise 0. `info["rewards"]` maps each active seat to its reward.

## What the baselines do

The Q-learner (`src/Qlearn.py`) is a 16-by-10 table. States are home, goal-lane, safe, and threatened, repeated for four pieces. The committed tables are `src/Qs.txt`, `src/Qs_BestOne.txt`, and `src/Qs80_50.txt`. `averageqs` copies one block of four rows across the table, so each file is four repeated rows. The state geometry still uses goal square 59, which was this fork's board before it was aligned with upstream's 58-tile track (goal 57). Training under `ruleset="classic"` is not a meaningful run until that geometry is fixed. The script constructs `Game(..., ruleset="legacy")` so a re-run stays on the rules the table was built for. Fixing the learner is later work.

The aggressive heuristic lives in the notebook, not in the engine. It tries to land on square 57, then to capture using `enemy_pieces[1] - 27`, then to step onto `{1, 5, 9, 12, 18, 25}`, otherwise it moves the furthest piece. `Player.aggresive` is stored and never read.

`VanilaMCTS.py` searches the MCTS player's piece choice. The opponent's reply is one random move written into the child. The backed-up value is the difference of piece-index sums, `points[2] - points[0]`. `randomMCTS.py` plays that search as player 2 against a random player 0, with players 1 and 3 ghosted, on `ruleset="legacy"`.

There is no policy network, no value network, and no self-play loop.

## Measured results

These numbers are measurements, not a leaderboard on `ruleset="classic"`. Nothing below was re-run on classic rules.

Notebook outputs, three-die game, `ruleset` equivalent to legacy, games stopped at a round cap. If nobody has finished, the recorded "winner" is whoever has the larger sum of piece indices.

- `3_Dice_Q_Learning_&_Aggressive_Walk_Two_Players.ipynb`, aggressive vs random, 2,000 games per cap. Round 1: aggressive 1,609/2,000 (80.4%). Round 5: 1,000/2,000. Round 30: 1,422/2,000 (71.1%). The Q-learning cell in that file only finished round 1 (446/1,000 for the Q-learner).
- The deleted checkpoint notebook had a longer Q-vs-aggressive sweep on an older checkout (goal square 59): round 1 was 271/1,000 for Q, round 20 was 965/1,000. That run was interrupted, and it is not a result on the current engine.

Full games played to a real winner on 21 Nov 2026, on the engine behaviour that is now `ruleset="legacy"`, three-die queue, seats 0 and 2:

- Aggressive vs random: 136/200 (68%).
- `src/Qs.txt` vs random: 176/200 (88%).
- `src/Qs.txt` vs aggressive: 72/150 (48%).

The same shape held on the one-die game: 137/200, 172/200, and 60/100 with the Q-table seated as player 2. `Qs_BestOne` vs aggressive was 50/100. Sample sizes are the denominators. These are not Wilson intervals and they are not an Elo rating.

`src/Data/TestingData.csv` ends, after 1,000 games, at cumulative player-0 rates of 82.3% (1 opponent), 70.6% (2), and 58.4% (3). The function that would regenerate it calls a name that does not exist (`Qlearn`) and ignores its ghost-player argument, so the file cannot be reproduced from the current script.

`images/VanillaMCTSvsRandomPlayerWinnPercentage.png` is a Weights & Biases chart. `randomMCTS.py` logs 19 per-move time budgets from 0.01 s to 1.0 s, 1,000 games each. Read off the image, the win rate starts just above 50% and ends in the low 80s. There is no raw table in the repo. The chart's horizontal axis is the logging step, not a tree depth.

## Roadmap

1. Freeze `ruleset="classic"` as the benchmark, and keep `legacy` for the numbers above.
2. Repair the Q-table geometry and the MCTS chance handling. The search should treat future dice as chance nodes, and it should copy a state before rolling it out. That work is not in this change.
3. An evaluation protocol: full games, every seat, 1,000 or more games, Wilson intervals, then Elo or TrueSkill once there are more than two agents. Baselines: random, the aggressive heuristic, a one-ply capture/goal bot, flat Monte Carlo, then a learned agent.
4. A policy-value network and self-play, using this simulator. A learned dynamics model (MuZero, Stochastic MuZero) is a comparison, not the first agent, because the rules are known.
5. Indian household variants: choice of die, blockades, three sixes in a row. They are specified in `docs/paper_plan.md` and are not implemented.

## Contact

Arindam Majee. Issues and discussion belong on this repository.
