import sys
import numpy as np
import sys
sys.path.append("LUDOpy")
from ludopy import Game
from VanilaMCTS import MCTS  # Assuming you have a MCTS implementation

def simulate_game():
    game = Game(ghost_players=[1, 3])
    mcts = MCTS(game)

    there_is_a_winner = False
    count = 0
    while not there_is_a_winner:
        (dice, move_pieces, player_pieces, enemy_pieces, player_is_a_winner,
         there_is_a_winner), player_i = game.get_observation()

        if player_i == 0:  # Random player
            if len(move_pieces):
                piece_to_move = move_pieces[np.random.randint(0, len(move_pieces))]
            else:
                piece_to_move = -1
        else:  # MCTS player
            piece_to_move = mcts.search()
            count += 1
            print("MCTS player moves", piece_to_move, "with dice", dice, "Search count ", count)

        _, _, _, _, _, there_is_a_winner = game.answer_observation(piece_to_move)

    print("Saving history to numpy file")
    game.save_hist("game_history.npy")
    print("Saving game video")
    game.save_hist_video("game_video.mp4")

    return True

if __name__ == '__main__':
    simulate_game()