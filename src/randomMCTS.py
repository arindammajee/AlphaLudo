import sys
from datetime import time
import time
import numpy as np
import sys
sys.path.append("/Users/arimajee/Desktop/Arindam/A-Reinforcement-Learning-Approach-to-Ludo/LUDOpy")
from ludopy import Game
from VanilaMCTS import MCTS  # Assuming you have a MCTS implementation

def simulate_game(mcts_iterations=10, node_depth=3):
    game = Game(ghost_players=[1, 3])

    there_is_a_winner = False
    count = 0
    while not there_is_a_winner:
        (dice, move_pieces, player_pieces, enemy_pieces, player_is_a_winner,
         there_is_a_winner), player_i = game.get_observation()
        if len(move_pieces):
            if player_i == 0:  # Random player
                #print("Random Player's Turn")
                piece_to_move = move_pieces[np.random.randint(0, len(move_pieces))]
            else:  # MCTS player
                #print("MCTS Turn")
                #print(game.points)
                if len(move_pieces) > 1:
                    mcts = MCTS(game)
                    piece_to_move = mcts.search(iterations=mcts_iterations, node_depth=node_depth)
                    break
                else:
                    piece_to_move = move_pieces[0]
                count += 1
                #print("MCTS player moves", piece_to_move, "with dice", dice, "Search count ", count)
        else:
            piece_to_move = -1

        _, _, _, _, _, there_is_a_winner = game.answer_observation(piece_to_move)

    # print("Saving history to numpy file")
    # game.save_hist("game_history.npy")
    # print("Saving game video")
    # game.save_hist_video("game_video.mp4")
    #print("Winner", game.first_winner_was)

    return game.first_winner_was

if __name__ == '__main__':
    print("Simulating game")
    TOTAL_GAMES = 1
    MCTS_ITERATIONS = 50
    MCTS_DEPTH = 3
    MCTS_WIN = 0
    start_time = time.time()
    for i in range(TOTAL_GAMES):
        if simulate_game(mcts_iterations=MCTS_ITERATIONS, node_depth=MCTS_DEPTH) == 2:
            MCTS_WIN += 1
        print(f"Game Number: {i+1}, MCTS win rate: {MCTS_WIN / (i+1)}, Took {time.time() - start_time} seconds")

    end_time = time.time()
    print("Simulation took", end_time - start_time, "seconds")

    