"""Tests for the Gymnasium-style Ludo environment."""

import numpy as np
import pytest

from ludopy import LudoEnv


def _drain(n_agents, ruleset, seed, cap=12000):
    env = LudoEnv(n_agents=n_agents, ruleset=ruleset, seed=seed, max_steps=cap)
    obs, _info = env.reset(seed=seed)
    terminated = False
    truncated = False
    info = {"winner": -1}
    steps = 0
    while not terminated and not truncated and steps < cap:
        legal = np.flatnonzero(obs["action_mask"])
        piece = int(legal[0]) if len(legal) else -1
        obs, _reward, terminated, truncated, info = env.step(piece)
        steps += 1
    return terminated, int(info["winner"]), steps


def test_reset_exposes_die_and_length_4_mask():
    env = LudoEnv(n_agents=2, ruleset="classic", seed=0)
    first, info = env.reset(seed=0)
    second, _info = LudoEnv(n_agents=2, ruleset="classic").reset(seed=0)
    assert first["action_mask"].shape == (4,)
    assert first["action_mask"].dtype == np.int8
    assert 1 <= first["die"] <= 6
    assert first["player"] == 0
    assert first["agent_index"] == 0
    assert info["action_mask"].shape == (4,)
    assert first["die"] == second["die"]
    assert np.array_equal(first["action_mask"], second["action_mask"])
    assert np.array_equal(first["pieces"], second["pieces"])


def test_illegal_piece_raises_when_a_move_exists():
    env = LudoEnv(n_agents=4, ruleset="legacy", seed=1)
    obs, _info = env.reset(seed=1)
    if not obs["action_mask"].any():
        obs, _reward, _term, _trunc, _info = env.step(-1)
    assert obs["action_mask"].any()
    with pytest.raises(ValueError):
        env.step(5)


@pytest.mark.parametrize("n_agents,ruleset", [(2, "classic"), (4, "classic"), (2, "legacy"), (4, "legacy")])
def test_full_games_for_two_and_four_agents(n_agents, ruleset):
    terminated, winner, steps = _drain(n_agents, ruleset, seed=4)
    again, winner_again, steps_again = _drain(n_agents, ruleset, seed=4)
    assert terminated
    assert winner == winner_again
    assert steps == steps_again
    assert steps < 12000
    if n_agents == 2:
        assert winner in (0, 2)
    else:
        assert winner in (0, 1, 2, 3)
