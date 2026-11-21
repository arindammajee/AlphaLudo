# Paper plan: a multi-player Ludo benchmark with chance-node search

Working title: **AlphaLudo: a tested multi-player Ludo benchmark, and how much of the game is skill.**

This is a plan, not a paper. Citations below are a reading list to verify before they are used. Rule variants described as Indian household rules are not implemented yet. The engine today has `ruleset="classic"` (Danish, upstream LUDOpy) and `ruleset="legacy"` (this fork's previous behaviour).

## Claim the paper has to earn

Ludo is a four-player, perfect-information, stochastic race. The die is public before the piece is chosen, so the decision itself is deterministic. Future dice, and the other players, are the stochastic and adversarial parts. That split is a clean test of search methods that add chance nodes, and of whether a learned stochastic model is worth it when a correct simulator is free.

The paper should contribute three things:

1. A public environment with frozen, tested rules, including more than one variant.
2. A baseline ladder and an evaluation protocol that reports skill separately from dice luck.
3. A chance-node AlphaZero agent, compared with Stochastic MuZero on the same games.

A 16-state Q-table is a baseline in that ladder. It is not the contribution.

## Rules to freeze

### Danish, implemented as `ruleset="classic"`

Upstream LUDOpy. One die. Exit home only on a 6. A 6 after round 1 grants another roll. Round-1 opening attempts follow upstream's counter (`start_attempts < 3`), which keeps the turn after three failed rolls and passes on the fourth if no 6 has been rolled. Stars jump to the next star. Square 51 enters the goal. Overshoot bounces. One enemy on a square is captured. Two enemies send the mover home. An occupied globe protects the occupant. The goal requires an exact count. Four players, four pieces, goal square 57.

`ruleset="legacy"` stays in the code so older notebook numbers can be regenerated. It is not a benchmark condition.

### Indian household variants, not implemented

These need a written rules card and golden tests before any training run. Household Ludo is not one game. The paper should pick one card per variant and say whose rules they are.

- **Dice choice.** Each turn the player rolls one die, or rolls two and must play one of the faces (not the sum, unless the card says so). The unused face is not carried. This adds a real decision on top of the piece choice.
- **Blockade.** Two of your own pieces on one square cannot be captured and block passage. A piece may not land on the blockade or pass it, depending on the card. The card has to pick one.
- **Three sixes.** Three sixes in a row forfeit the turn, or forfeit the turn and send the piece that moved on the first six back home. The card has to pick one. Consecutive sixes already interact with the extra-turn rule, so the test should include the sequence 6, 6, 6 explicitly.

A variant is in the benchmark only after the same style of test that `tests/test_rules.py` uses for classic and legacy.

### Three-die queue

`Game3Dice` deals the maximum of a shared three-face queue. It is an existing fork behaviour, not a standard ruleset. The paper should either redefine it as a per-player choice of one of three faces, or leave it out of the benchmark. Do not report it as "Indian Ludo" without that change.

## Skill versus luck

The protocol has to separate the die from the player. Two measurements, both run on every variant:

1. **Shared dice, swapped seats.** Fix the dice stream. Play the match, then play it again with the agents rotated. A result that flips when the seats flip is luck. A result that holds is skill. Report the agreement rate and a confidence interval. For four players, use a Latin square of seat assignments rather than one swap.
2. **Strength above random, under the same dice budget.** The gap between an agent and a random mover, in win rate, is a lower bound on skill only if both face the same opponent distribution. Report it, and also the gap to the best search agent. The residual down to 100% is not "luck" by itself, because the search agent is not optimal.

Do not use piece-index sums as wins. The saved notebook "win rates" at short round caps are that score. The paper reports finished games only.

## Evaluation protocol

- Finish the game. No round cap, no piece-sum tie-break, except a declared ply limit that is long enough that truncation is rare, and truncation is reported separately.
- Every seat. Two-player results use both directions. Four-player results rotate the agent through all four seats, or use the Latin square from the skill protocol.
- At least 1,000 finished games per cell of the matrix below. Report a Wilson interval on the win rate, not only the point estimate.
- Once the ladder has more than two agents, fit Elo (two-player) or TrueSkill (multi-player, Herbrich, Minka, and Graepel, 2006; verify the citation). Publish the pairwise win matrix next to the rating so a rating collapse is visible.
- Opponents in the ladder, weakest first: uniform random piece, the aggressive heuristic (after its capture offset is checked against the engine), a one-ply bot that prefers goal, then capture, then a safe square, flat Monte Carlo, chance-node MCTS with a uniform policy, then the learned agents.
- The same dice seeds are released with the paper so the tables can be regenerated.

## Agents

### Chance-node AlphaZero, known model

The simulator is the environment. At a decision node the die has already been rolled, and the policy picks a piece under the action mask. The next node's chance event is the upcoming roll, which has six faces of probability 1/6, or fewer if the ruleset's extra-turn and three-sixes rules change the length of the turn. Search either expands all six faces or samples them (progressive widening at chance nodes). Opponent seats use the current network, or a frozen baseline when the experiment says so.

This is expectimax-style MCTS with a learned prior and value. It is the primary learned agent, because the rules are known and a game is a few hundred plies.

Value target: the acting player wins (+1) or not (0) in the two-player game. In the four-player game, +1 for the winner and 0 for the others, or +1 and −1/3. Pick one and stick to it. State encoding: four pieces per player on the 58-square track, the public die, whose turn, and the ruleset id if one network sees several variants. An MLP or a small convolution over piece-planes is enough. The action head has four logits and the mask.

Train by self-play with seat rotation, and keep the frozen baselines in the mixture so the net does not only learn to beat itself.

### Stochastic MuZero, learned model, as a comparison

Antonoglou, Schrittwieser, Ozair, Hubert, and Silver, "Planning in Stochastic Environments with a Learned Model", ICLR 2022. Afterstates separate the piece move from the next die. Chance nodes are sampled from a learned categorical distribution over a codebook. The comparison is: with a free, correct simulator, does the learned stochastic model match chance-node AlphaZero, and where does it lose (three sixes, blockades, four players)?

MuZero without chance nodes is a negative control, not a contender. It was built for deterministic transitions.

### What not to lead with

Repairing `src/Qlearn.py` and `src/VanilaMCTS.py` produces stronger baselines. It does not answer the research question. The current MCTS bakes one random opponent reply into each child and backs up a piece-sum. The current Q-table addresses goal square 59 on a board whose goal is 57.

## Experiment matrix

Rows are rules. Columns are agents. Each cell is at least 1,000 finished games, all seats, Wilson intervals.

| Rules | Random | Heuristic | 1-ply | Flat MC | Chance MCTS | AlphaZero (known model) | Stochastic MuZero |
|---|---|---|---|---|---|---|---|
| Danish classic, 2 players | | | | | | | |
| Danish classic, 4 players | | | | | | | |
| Dice choice, 2 and 4 | | | | | | | |
| Blockade, 2 and 4 | | | | | | | |
| Three sixes, 2 and 4 | | | | | | | |
| One transfer setting: train on Danish, test on dice-choice | | | | | | | |

Skill-versus-luck (shared dice, swapped seats) is a second table on the same rows, for random, the heuristic, and the two learned agents.

Ablations, only after the main table exists: enumerate six faces versus sample them; self-play only versus self-play mixed with baselines; two-player training tested at four players.

## Research questions

1. Under Danish rules, what fraction of pairwise outcomes flip when the same dice are replayed with seats swapped? Does that fraction drop from random to chance-node AlphaZero?
2. Does dice choice, a blockade, or three sixes change that fraction, and does it change the ranking of the heuristic against search?
3. With the simulator available at every node, does Stochastic MuZero match chance-node AlphaZero in Elo, and on which variant does the learned model fall behind?
4. Does a network trained on Danish classic transfer to dice-choice, or does the extra decision require its own training?
5. In four-player games, how much of the win rate is the seat rather than the agent? The Latin square is the measurement.

## Venue

- **NeurIPS Datasets and Benchmarks** fits if the delivered artifact is the environment, the rules cards, the baseline ladder, and the protocol, with the learned agents as reference numbers. The bar is a reusable benchmark, not a new algorithm.
- **IEEE Conference on Games (CoG)** fits the same material for the games audience, and it is where the 2012 Ludo Q-learning paper was published. A CoG paper can be shorter and can lead with the skill-versus-luck measurement.
- **NeurIPS or AAAI main track** fits only if the comparison itself is the result: known-model chance-node search against Stochastic MuZero, with a clear win, loss, or scaling finding. A benchmark plus a reimplementation of AlphaZero will not carry a main-track paper.
- A reasonable path is CoG or NeurIPS Datasets and Benchmarks first, and a main-track paper only if question 3 produces a result that stands without the benchmark framing.

## Related work to verify

Verify author lists, years, and venues against the publisher page before citing.

- Simon L. B. Sørensen, LUDOpy, the Danish engine this repository forks. MIT, 2020 copyright notice in `src/ludopy/LICENSE.MIT`.
- Majed Alhajry, Faisal Alvi, and Moataz Ahmed. "TD(λ) and Q-learning based Ludo players." IEEE Conference on Computational Intelligence and Games, 2012. DOI 10.1109/CIG.2012.6374142. They report a TD(λ) player and a Q-learning player slightly ahead of a hand-built expert. Their Ludo is not this engine.
- "Incorporating Feature Penalty in Reinforcement Learning for Ludo Game." IEEE Congress on Information Science and Technology (CiSt), 2023. DOI 10.1109/CIST56084.2023.10409945. Confirm the author list on IEEE Xplore. Q-learning with an explicit penalty feature.
- "Autonomous Agent Using AI Q-Learning in Augmented Reality Ludo Board Game." ICAETA 2023, Springer. Confirm the authors. An AR interface with Q-learning and minimax, not a search benchmark.
- David Silver, Thomas Hubert, Julian Schrittwieser, et al. "A general reinforcement learning algorithm that masters chess, shogi, and Go through self-play." Science, 2018. AlphaZero. Also the 2017 arXiv preprint. Confirm the author list against Science.
- Julian Schrittwieser, Ioannis Antonoglou, Thomas Hubert, et al. "Mastering Atari, Go, chess and shogi by planning with a learned model." Nature, 2020. MuZero.
- Ioannis Antonoglou, Julian Schrittwieser, Sherjil Ozair, Thomas Hubert, and David Silver. "Planning in Stochastic Environments with a Learned Model." ICLR 2022. Stochastic MuZero. Afterstates, chance nodes sampled from a learned codebook. The comparison agent.
- Cameron Browne, Edward Powley, Daniel Whitehouse, Simon Lucas, Peter Cowling, Philipp Rohlfshagen, Stephen Tavener, Diego Perez, Spyridon Samothrakis, and Simon Colton. "A Survey of Monte Carlo Tree Search Methods." IEEE Transactions on Computational Intelligence and AI in Games, 2012. Confirm the author list. Use for the chance-node and expectimax variants, together with the papers it cites on stochastic MCTS.
- Maciej Świechowski, Konrad Godlewski, Bartosz Sawicki, and Jacek Mańdziuk. "Monte Carlo tree search: A review of recent modifications and applications." Artificial Intelligence Review, 2023. Already cited in the old README.
- Adrien Couëtoux, Jean-Baptiste Hoock, Nataliya Sokolovska, Olivier Teytaud, and Nicolas Bonnard. Work on continuous and stochastic MCTS (Double Progressive Widening). Confirm the exact 2011 paper title and author list. This is the sampling scheme for chance nodes with a known dice distribution.
- Ralf Herbrich, Tom Minka, and Thore Graepel. "TrueSkill: A Bayesian Skill Rating System." NeurIPS, 2006. Confirm the venue name used that year. The multi-player rating.
- Marc Lanctot, Edward Lockhart, Jean-Baptiste Lespiau, et al. OpenSpiel. Confirm the 2019 arXiv author list. The comparison point for "why a new environment instead of an OpenSpiel game."
- J. K. Terry, Benjamin Black, Nathaniel Grammel, et al. PettingZoo. Confirm the 2021 paper. The multi-agent API this environment is shaped after, without taking it as a dependency yet.
- Hui Wang, Michael Emmerich, and Aske Plaat. "Monte Carlo Q-learning for General Game Playing." arXiv:1802.05944, 2018. Relevant as a caution that tabular Q-learning plus shallow search does not scale, not as a Ludo result.

## Milestones

Calendar dates are for the owner to assign. The order is the dependency order. A later milestone does not start by training on untested rules.

| Order | Milestone | Done when |
|---|---|---|
| 1 | Engine and tests for Danish classic, legacy kept for old numbers | This change. `pytest` green, default ruleset classic, licence split written down. |
| 2 | Evaluation harness | A script runs 1,000 games, all seats, writes a win matrix and Wilson intervals. Shared-dice seat swap works for two players. No learned agent required. |
| 3 | Baseline ladder on Danish classic | Random, corrected heuristic, one-ply, flat Monte Carlo, and a repaired chance-node MCTS (known model, uniform prior) have numbers from milestone 2. The old Q-table is reported only as a legacy-rules curiosity, or it is retrained on classic after its state geometry matches goal square 57. |
| 4 | Indian rule cards | Dice choice, blockade, and three sixes each have a written card and golden tests. The matrix rows exist even if the cells are empty. |
| 5 | Chance-node AlphaZero | Self-play on Danish classic, two players, beats the milestone-3 search baseline on the harness. Four-player training is a separate run, not a claim copied from two players. |
| 6 | Stochastic MuZero comparison | Same harness, same seeds, Danish classic first, then the variant where milestone 5 was strongest. Report Elo or TrueSkill and the shared-dice agreement rate. |
| 7 | Paper | Venue chosen from the section above using the actual tables. Released code, seeds, and the rules cards match the PDF. |

## Out of scope for the first paper

- Human experiments.
- An AR or graphical client. The existing Matplotlib and OpenCV renderers are debugging tools.
- Claiming a solved game. Four-player Ludo with a 6-faced die is not going to be solved by this project.
- Treating the 21 Nov 2026 legacy-rules smoke numbers (136/200, 176/200, 72/150) as benchmark results. They are a record of the old engine, quoted in the README so they are not lost.
