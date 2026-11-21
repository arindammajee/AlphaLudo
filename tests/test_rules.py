"""Golden tests for classic (upstream Danish) and legacy movement."""

import numpy as np
import pytest

from ludopy import Game, Game3Dice
from ludopy.player import (
    GOAL_INDEX,
    RULESET_CLASSIC,
    RULESET_LEGACY,
    Player,
    enemy_pos_at_pos,
)


class ScriptedRng:
    def __init__(self, faces):
        self.faces = [int(face) for face in faces]
        self.index = 0

    def integers(self, low, high=None, size=None):
        if self.index >= len(self.faces):
            raise RuntimeError("scripted dice exhausted")
        value = self.faces[self.index]
        self.index += 1
        return value


def _blank_enemies():
    return [np.zeros(4, dtype=int) for _ in range(3)]


def _enemies_on(square, count, enemy=0):
    boards = _blank_enemies()
    spot = int(enemy_pos_at_pos(square)[enemy][0])
    for piece in range(count):
        boards[enemy][piece] = spot
    return boards, spot


def _move(ruleset, start, dice, enemies=None):
    player = Player(ruleset=ruleset)
    player.set_pieces(np.array([start, 0, 0, 0]))
    if enemies is None:
        enemies = _blank_enemies()
    updated = player.move_piece(0, dice, enemies)
    return player, updated


@pytest.mark.parametrize("ruleset", [RULESET_CLASSIC, RULESET_LEGACY])
def test_exact_goal(ruleset):
    player = Player(ruleset=ruleset)
    player.set_pieces(np.array([GOAL_INDEX, GOAL_INDEX, GOAL_INDEX, 54]))
    player.move_piece(3, 3, _blank_enemies())
    assert int(player.get_pieces()[3]) == GOAL_INDEX
    assert player.player_winner()


@pytest.mark.parametrize("ruleset", [RULESET_CLASSIC, RULESET_LEGACY])
def test_single_capture_on_a_free_square(ruleset):
    enemies, _spot = _enemies_on(4, 1)
    player, updated = _move(ruleset, start=2, dice=2, enemies=enemies)
    assert int(player.get_pieces()[0]) == 4
    assert int(updated[0][0]) == 0


@pytest.mark.parametrize("ruleset", [RULESET_CLASSIC, RULESET_LEGACY])
def test_stack_of_two_sends_the_mover_home(ruleset):
    enemies, spot = _enemies_on(4, 2)
    player, updated = _move(ruleset, start=2, dice=2, enemies=enemies)
    assert int(player.get_pieces()[0]) == 0
    assert int(updated[0][0]) == spot
    assert int(updated[0][1]) == spot


@pytest.mark.parametrize("ruleset", [RULESET_CLASSIC, RULESET_LEGACY])
def test_occupied_globe_sends_the_mover_home(ruleset):
    enemies, spot = _enemies_on(9, 1)
    player, updated = _move(ruleset, start=7, dice=2, enemies=enemies)
    assert int(player.get_pieces()[0]) == 0
    assert int(updated[0][0]) == spot


def test_home_exit_depends_on_ruleset():
    classic = Player(ruleset=RULESET_CLASSIC)
    legacy = Player(ruleset=RULESET_LEGACY)
    assert classic.get_pieces_that_can_move(3) == []
    assert classic.get_pieces_that_can_move(6) == [0, 1, 2, 3]
    assert legacy.get_pieces_that_can_move(3) == [0, 1, 2, 3]

    classic_after, _ = _move(RULESET_CLASSIC, start=0, dice=6)
    legacy_after, _ = _move(RULESET_LEGACY, start=0, dice=3)
    assert int(classic_after.get_pieces()[0]) == 1
    assert int(legacy_after.get_pieces()[0]) == 1
    stayed, _ = _move(RULESET_CLASSIC, start=0, dice=3)
    assert int(stayed.get_pieces()[0]) == 0


def test_star_jump_and_star_capture_depend_on_ruleset():
    classic, _ = _move(RULESET_CLASSIC, start=6, dice=6)
    legacy, _ = _move(RULESET_LEGACY, start=6, dice=6)
    assert int(classic.get_pieces()[0]) == 18
    assert int(legacy.get_pieces()[0]) == 12

    # Landing on star 44 jumps to 51 and does not continue into the goal.
    chained, _ = _move(RULESET_CLASSIC, start=38, dice=6)
    assert int(chained.get_pieces()[0]) == 51

    enemies, spot = _enemies_on(12, 1)
    classic_hit, classic_enemies = _move(RULESET_CLASSIC, 6, 6, enemies)
    enemies, spot = _enemies_on(12, 1)
    legacy_hit, legacy_enemies = _move(RULESET_LEGACY, 6, 6, enemies)
    assert int(classic_hit.get_pieces()[0]) == 18
    assert int(classic_enemies[0][0]) == 0
    assert int(legacy_hit.get_pieces()[0]) == 12
    assert int(legacy_enemies[0][0]) == spot


def test_final_star_enters_goal_and_classic_captures_there():
    classic, _ = _move(RULESET_CLASSIC, start=45, dice=6)
    legacy, _ = _move(RULESET_LEGACY, start=45, dice=6)
    assert int(classic.get_pieces()[0]) == GOAL_INDEX
    assert int(legacy.get_pieces()[0]) == GOAL_INDEX

    enemies, spot = _enemies_on(51, 1)
    classic_hit, classic_enemies = _move(RULESET_CLASSIC, 45, 6, enemies)
    enemies, _ = _enemies_on(51, 1)
    legacy_hit, legacy_enemies = _move(RULESET_LEGACY, 45, 6, enemies)
    assert int(classic_hit.get_pieces()[0]) == GOAL_INDEX
    assert int(classic_enemies[0][0]) == 0
    assert int(legacy_hit.get_pieces()[0]) == GOAL_INDEX
    assert int(legacy_enemies[0][0]) == spot


def test_overshoot_bounces_only_in_classic():
    classic, _ = _move(RULESET_CLASSIC, start=56, dice=3)
    legacy, _ = _move(RULESET_LEGACY, start=56, dice=3)
    assert int(classic.get_pieces()[0]) == 54
    assert int(legacy.get_pieces()[0]) == 56


def _scripted_game(ruleset, faces):
    game = Game(ghost_players=[1, 3], ruleset=ruleset, seed=0)
    game.rng = ScriptedRng(faces)
    seen = []
    for _ in faces:
        obs, player = game.get_observation()
        piece = int(obs[1][0]) if len(obs[1]) else -1
        game.answer_observation(piece)
        seen.append(player)
    return game, seen


def test_opening_failures_keep_the_turn_only_in_classic():
    # Upstream keeps the turn while the attempt counter is 0, 1, or 2.
    # A player who never rolls a 6 therefore rolls four times, then the
    # turn passes. legacy passes on the first failed roll.
    _game, classic_players = _scripted_game(RULESET_CLASSIC, [1, 1, 1, 1, 5])
    _game, legacy_players = _scripted_game(RULESET_LEGACY, [1, 1, 1, 1, 5])
    assert classic_players == [0, 0, 0, 0, 2]
    assert legacy_players == [0, 2, 0, 2, 0]


def test_six_on_round_one_leaves_home_and_passes():
    game, players = _scripted_game(RULESET_CLASSIC, [6, 1])
    assert players == [0, 2]
    assert int(game.players[0].get_pieces()[0]) == 1


def test_extra_turn_on_six_after_round_one():
    for ruleset, expected_next in ((RULESET_CLASSIC, 0), (RULESET_LEGACY, 2)):
        game = Game(ghost_players=[1, 3], ruleset=ruleset, seed=0)
        game.round = 2
        game.players[0].set_pieces(np.array([10, 0, 0, 0]))
        game.rng = ScriptedRng([6, 3])
        obs, player = game.get_observation()
        assert player == 0 and obs[0] == 6
        game.answer_observation(int(obs[1][0]))
        _obs, nxt = game.get_observation()
        assert nxt == expected_next


def test_unknown_ruleset_is_rejected():
    with pytest.raises(ValueError):
        Game(ruleset="indian")


@pytest.mark.parametrize("cls", [Game, Game3Dice])
def test_strategic_player_keyword_is_accepted(cls):
    game = cls(ghost_players=[1, 3], strategic_player=0, ruleset="legacy", seed=1)
    assert game.strategic_player == 0
    assert game.ruleset == "legacy"


def test_get_piece_hist_and_save_hist_extension(tmp_path):
    game = Game(ghost_players=[1, 3], ruleset="legacy", seed=2)
    obs, _player = game.get_observation()
    piece = int(obs[1][0]) if len(obs[1]) else -1
    game.answer_observation(piece)
    history = game.get_piece_hist()
    assert len(history) >= 1
    assert len(history[0]) == 4

    archive = tmp_path / "history.npz"
    game.save_hist(str(archive))
    assert archive.exists()
    assert not (tmp_path / "history.npz.npz").exists()
    with pytest.raises(ValueError):
        game.save_hist(str(tmp_path / "history.npy"))

    three = Game3Dice(ghost_players=[1, 3], ruleset="legacy", seed=2)
    obs, _player = three.get_observation()
    piece = int(obs[1][0]) if len(obs[1]) else -1
    three.answer_observation(piece)
    assert len(three.get_piece_hist()) >= 1
    array_path = tmp_path / "three.npy"
    three.save_hist(str(array_path))
    assert array_path.exists()
    assert not (tmp_path / "three.npy.npy").exists()


def _play_full(ruleset, ghosts, seed, cap=12000):
    game = Game(ghost_players=ghosts, ruleset=ruleset, seed=seed)
    steps = 0
    while game.get_winner_of_game() == -1 and steps < cap:
        obs, _player = game.get_observation()
        if len(obs[1]):
            piece = int(obs[1][steps % len(obs[1])])
        else:
            piece = -1
        game.answer_observation(piece)
        steps += 1
    return game.get_winner_of_game(), steps


@pytest.mark.parametrize("ruleset", [RULESET_CLASSIC, RULESET_LEGACY])
@pytest.mark.parametrize("ghosts", ([1, 3], []))
def test_seeded_random_games_finish_and_replay(ruleset, ghosts):
    first = _play_full(ruleset, ghosts, seed=0)
    second = _play_full(ruleset, ghosts, seed=0)
    assert first == second
    winner, steps = first
    assert winner in range(4)
    assert winner not in ghosts
    assert steps < 12000
