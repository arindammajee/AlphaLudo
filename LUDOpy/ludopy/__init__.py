"""Compatibility loader for notebooks that put ``LUDOpy`` on ``sys.path``.

The engine implementation lives in ``src/ludopy``. This package redirects
its search path there so ``import ludopy`` from either entry point loads
one copy of the code. The implementation is MIT; see
``src/ludopy/LICENSE.MIT``. This loader does not relicense it.
"""

from pathlib import Path

_CANON = Path(__file__).resolve().parents[2] / "src" / "ludopy"
__path__ = [str(_CANON)]

from .game import Game
from .game_3dice import Game as Game3Dice
from .visualizer import make_img_of_board, save_hist_video
from .visualizer_3dice import make_img_of_board as make_img_of_board_3dice
from .visualizer_3dice import save_hist_video as save_hist_video_3dice
from .player import (
    RULESET_CLASSIC,
    RULESET_LEGACY,
    enemy_pos_at_pos,
    get_enemy_at_pos,
)
from .env import LudoEnv

__all__ = [
    "Game",
    "Game3Dice",
    "LudoEnv",
    "RULESET_CLASSIC",
    "RULESET_LEGACY",
    "enemy_pos_at_pos",
    "get_enemy_at_pos",
    "make_img_of_board",
    "make_img_of_board_3dice",
    "save_hist_video",
    "save_hist_video_3dice",
]
