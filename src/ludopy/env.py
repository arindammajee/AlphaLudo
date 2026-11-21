"""Gymnasium-style environment for one Ludo game.

The class does not import Gymnasium or PettingZoo. ``reset(seed)`` and
``step(piece)`` follow that calling convention: an observation dict, a
reward for the player who just acted, and terminated/truncated flags.

The die is public in the observation. ``action_mask`` has length 4, one
entry per piece of the player about to move. A 1 means that piece is legal.

``n_agents=4`` uses every seat. ``n_agents=2`` uses seats 0 and 2, the same
pairing the experiment scripts use when they ghost players 1 and 3.
"""

import numpy as np

from .game import Game
from .player import RULESET_CLASSIC, validate_ruleset


class LudoEnv:
    def __init__(self, n_agents=4, ruleset=RULESET_CLASSIC, max_steps=5000, seed=None):
        if n_agents not in (2, 4):
            raise ValueError("n_agents must be 2 or 4")
        self.n_agents = n_agents
        self.ruleset = validate_ruleset(ruleset)
        self.max_steps = int(max_steps)
        self._seed = seed
        self.agent_seats = [0, 1, 2, 3] if n_agents == 4 else [0, 2]
        self.game = None
        self.steps = 0
        self._pending = None

    def reset(self, seed=None):
        """Start a game. Returns ``(observation, info)``."""
        if seed is not None:
            self._seed = seed
        ghosts = [seat for seat in range(4) if seat not in self.agent_seats]
        self.game = Game(
            ghost_players=ghosts,
            ruleset=self.ruleset,
            seed=self._seed,
        )
        self.steps = 0
        obs_tuple, player = self.game.get_observation()
        observation = self._observation(obs_tuple, player)
        self._pending = observation
        return observation, {"player": player, "action_mask": observation["action_mask"]}

    def step(self, piece):
        """Apply a piece choice for the current player.

        Returns ``(observation, reward, terminated, truncated, info)``.
        ``reward`` is 1 if the acting player just won, otherwise 0.
        ``info["rewards"]`` maps every active seat to its reward.
        """
        if self._pending is None:
            raise RuntimeError("reset() must be called before step()")
        acting = int(self._pending["player"])
        legal = [i for i, bit in enumerate(self._pending["action_mask"]) if bit]
        if legal:
            if int(piece) not in legal:
                raise ValueError(
                    f"Piece {piece} is not legal for player {acting}. Legal pieces: {legal}"
                )
            chosen = int(piece)
        else:
            chosen = -1

        self.game.answer_observation(chosen)
        self.steps += 1
        winner = self.game.get_winner_of_game()
        terminated = winner != -1
        truncated = (not terminated) and self.steps >= self.max_steps
        rewards = {seat: 0.0 for seat in self.agent_seats}
        if terminated:
            rewards[winner] = 1.0
            observation = self._terminal_observation(acting, winner)
        else:
            obs_tuple, player = self.game.get_observation()
            observation = self._observation(obs_tuple, player)
        self._pending = None if (terminated or truncated) else observation
        reward = float(rewards.get(acting, 0.0))
        info = {
            "player": acting,
            "next_player": None if observation is None else int(observation["player"]),
            "winner": int(winner),
            "rewards": rewards,
            "action_mask": observation["action_mask"],
        }
        return observation, reward, terminated, truncated, info

    def _observation(self, obs_tuple, player):
        dice, move_pieces, player_pieces, enemy_pieces, _player_won, _game_won = obs_tuple
        mask = np.zeros(4, dtype=np.int8)
        for piece in move_pieces:
            mask[int(piece)] = 1
        return {
            "die": int(dice),
            "action_mask": mask,
            "player": int(player),
            "agent_index": self.agent_seats.index(int(player)),
            "pieces": np.asarray(player_pieces, dtype=int),
            "enemy_pieces": np.asarray(enemy_pieces, dtype=int),
        }

    def _terminal_observation(self, acting, winner):
        mask = np.zeros(4, dtype=np.int8)
        return {
            "die": int(self.game.current_dice),
            "action_mask": mask,
            "player": int(acting),
            "agent_index": self.agent_seats.index(int(acting)),
            "pieces": np.asarray(self.game.players[acting].get_pieces(), dtype=int),
            "enemy_pieces": np.zeros((3, 4), dtype=int),
            "winner": int(winner),
        }
