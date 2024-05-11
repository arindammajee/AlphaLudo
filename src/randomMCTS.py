import sys
import numpy as np
import sys
sys.path.append("LUDOpy")
from ludopy import Game
from VanilaMCTS import MCTS  # Assuming you have a MCTS implementation

def simulate_game(mcts_iterations=10):
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
            piece_to_move = mcts.search(iterations=mcts_iterations)
            count += 1
            # print("MCTS player moves", piece_to_move, "with dice", dice, "Search count ", count)

        _, _, _, _, _, there_is_a_winner = game.answer_observation(piece_to_move)

    # print("Saving history to numpy file")
    # game.save_hist("game_history.npy")
    # print("Saving game video")
    # game.save_hist_video("game_video.mp4")
    print("Winner", game.first_winner_was)

    return game.first_winner_was

if __name__ == '__main__':
    print("Simulating game")
    TOTAL_GAMES = 100
    MCTS_ITERATIONS = 30
    MCTS_WIN = 0
    for i in range(TOTAL_GAMES):
        if simulate_game(mcts_iterations=MCTS_ITERATIONS) == 2:
            MCTS_WIN += 1

    print("MCTS win rate", MCTS_WIN / TOTAL_GAMES)