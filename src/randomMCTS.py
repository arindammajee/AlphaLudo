from datetime import time
import time
import numpy as np
import wandb
from VanilaMCTS import MCTS
from ludopy import Game
import datetime
from multiprocessing import Pool


def simulate_game(game_config):
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
                    mcts = MCTS(game,
                                game_config["mcts_game_run_time"],
                                game_config["mcts_game_iterations"],
                                game_config["is_time_bounded"])
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


def simulate_mcts_experiment(TOTAL_GAMES, MCTS_RUN_TIME=0.01, MCTS_ITERATIONS=10, isTimeBounded=True,
                             multiprocessing_num=8):
    print("Simulating MCTS Experiment")
    game_configuration = {
        "mcts_game_run_time": MCTS_RUN_TIME,
        "mcts_game_iterations": MCTS_ITERATIONS,
        "is_time_bounded": isTimeBounded,
    }

    multiprocessing_epoch = TOTAL_GAMES // multiprocessing_num

    mcts_win = 0
    start_time = time.time()
    for i in range(multiprocessing_epoch):
        processes_pool = Pool(multiprocessing_num)
        multiprocessing_output = processes_pool.map(simulate_game,
                                                    [game_configuration for _ in range(multiprocessing_num)])
        for j in range(1, multiprocessing_num + 1):
            if multiprocessing_output[j-1] == 2:
                mcts_win += 1
            total_game_played = multiprocessing_num*i + j
            print(f"Multiprocessing Epoch {i+1}, Game Number: {total_game_played}, MCTS win rate: "
                  f"{mcts_win / total_game_played}")
        print(f"{multiprocessing_num} games in Multiprocessing Epoch {i+1} took {time.time() - start_time} seconds")

    simulation_time = time.time() - start_time

    print("Simulation took", simulation_time, "seconds")
    return mcts_win/TOTAL_GAMES, simulation_time


if __name__ == '__main__':
    mcts_run_time_list = [0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1,
                          0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    total_games = 1000
    wandb.init(project="MCTS-LUDO",
               name=f"MultiProcessing Random Player vs MCTS Player_2024-09-01 21:17:03.629611",
               config={"Total Games": total_games,
                       "MCTS Run Time": mcts_run_time_list})

    for mcts_run_time in mcts_run_time_list:
        print(f'Running MCTS simulation for {total_games} with allowed MCTS run time of {mcts_run_time}'
              f' ................')
        mcts_win_rate, simulation_total_time = simulate_mcts_experiment(TOTAL_GAMES=total_games,
                                                                        MCTS_RUN_TIME=mcts_run_time,
                                                                        isTimeBounded=True,
                                                                        multiprocessing_num=20)

        wandb.log({"MCTS win percentage": mcts_win_rate*100, "MCTS simulation time": simulation_total_time})
        print(f'MCTS simulation for {total_games} with allowed MCTS run time of {mcts_run_time} Ended ................')

    wandb.finish()
