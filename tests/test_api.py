"""Import and packaging checks that do not depend on a particular ruleset."""

import importlib
import subprocess
import sys
from pathlib import Path

import ludopy
from ludopy.player import RULESET_CLASSIC


ROOT = Path(__file__).resolve().parents[1]


def test_package_exports_the_one_die_artist_with_the_main_game():
    assert ludopy.make_img_of_board.__module__ == "ludopy.visualizer"
    assert ludopy.make_img_of_board_3dice.__module__ == "ludopy.visualizer_3dice"
    assert ludopy.Game(seed=0).ruleset == RULESET_CLASSIC


def test_mcts_script_imports_without_wandb():
    sys.modules.pop("wandb", None)
    module = importlib.import_module("randomMCTS")
    importlib.reload(module)
    assert "wandb" not in sys.modules


def test_ludopy_directory_shim_loads_the_canonical_engine():
    script = (
        "import sys\n"
        "sys.path.insert(0, 'LUDOpy')\n"
        "import ludopy\n"
        "game = ludopy.Game(ruleset='classic', strategic_player=1, seed=3)\n"
        "assert game.ruleset == 'classic'\n"
        "assert game.strategic_player == 1\n"
        "assert ludopy.LudoEnv is not None\n"
        "assert 'src/ludopy' in ludopy.__path__[0].replace('\\\\', '/')\n"
    )
    completed = subprocess.run(
        [sys.executable, "-c", script],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr
