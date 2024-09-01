from datetime import time
import time
import numpy as np
import sys
from VanilaMCTS import MCTS
sys.path.append("/Users/arimajee/Desktop/Arindam/A-Reinforcement-Learning-Approach-to-Ludo/LUDOpy")
from ludopy import Game


def simulate_game(MCTS_RUN_TIME, MCTS_ITERATIONS, isTimeBounded):
    game = Game(ghost_players=[1, 3])

    there_is_a_winner = False
    count = 0
    while not there_is_a_winner:
        (dice, move_pieces, player_pieces, enemy_pieces, player_is_a_winner,
         there_is_a_winner), player_i = game.get_observation()
        if len(move_pieces):
            if player_i == 0:
                piece_to_move = move_pieces[np.random.randint(0, len(move_pieces))]
            else:  # MCTS player
                if len(move_pieces) > 1:
                    mcts = MCTS(game, MCTS_RUN_TIME, MCTS_ITERATIONS, isTimeBounded)
                    piece_to_move = mcts.search()
                else:
                    piece_to_move = move_pieces[0]
                count += 1
        else:
            piece_to_move = -1

        _, _, _, _, _, there_is_a_winner = game.answer_observation(piece_to_move)

    # print("Saving history to numpy file")
    # game.save_hist("game_history.npy")
    # print("Saving game video")
    # game.save_hist_video("game_video.mp4")
    # print("Winner", game.first_winner_was)

    return game.first_winner_was


def simulate_mcts_experiment(TOTAL_GAMES, MCTS_RUN_TIME, MCTS_ITERATIONS, isTimeBounded):
    print("Simulating MCTS Experiment")
    mcts_win = 0
    start_time = time.time()
    for i in range(TOTAL_GAMES):
        if simulate_game(MCTS_RUN_TIME, MCTS_ITERATIONS, isTimeBounded) == 2:
            mcts_win += 1
        print(f"Game Number: {i + 1}, MCTS win rate: {mcts_win / (i + 1)}, Took {time.time() - start_time} seconds")

    end_time = time.time()
    print("Simulation took", end_time - start_time, "seconds")


if __name__ == '__main__':
    total_games = 1
    mcts_iteration = 100
    mcts_run_time = 1
    simulate_mcts_experiment(TOTAL_GAMES=total_games,
                             MCTS_RUN_TIME=mcts_run_time,
                             MCTS_ITERATIONS=mcts_iteration,
                             isTimeBounded=False)

    