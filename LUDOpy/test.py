import sys
import os
import ludopy
import numpy as np
import cv2
import time
import matplotlib.pyplot as plt
from IPython.display import clear_output
dir=os.getcwd()
qlearn_dir = '/'.join(dir.split('/')[:-1])
sys.path.append(os.path.abspath(os.path.join(qlearn_dir, 'src')))
from Qlearn import Qplayer

NUM_PLAYERS = 2 # Or 4

player_color_dict = {
    0: "GREEN (Bottom Left)",
    1: "YELLOW (Top Left)",
    2: "BLUE (Top Right)",
    3: "RED (Bottom Right)"
}

def gui_on(game, FIGURE_SIZE, SLEEP_TIME):
  enviroment_image_rgb = game.render_environment()  # RGB image of the enviroment
  #enviroment_image_bgr = cv2.cvtColor(enviroment_image_rgb, cv2.COLOR_RGB2BGR)
  clear_output(wait=True)
  figure = plt.figure(figsize=FIGURE_SIZE)
  plt.imshow(enviroment_image_rgb)
  plt.show()
  time.sleep(SLEEP_TIME)

def strategy(pieces_that_can_move, player_pieces, enemy_pieces, dice):
  safe_positions = [1, 5, 12, 18, 25]
  move_piece = -1
  if len(pieces_that_can_move)==0:
    return move_piece
  
  # Relative positions of enemy pieces
  relative_enemy_pieces = enemy_pieces[1] - 27
  # Mapping distance
  for piece in pieces_that_can_move:
    for enemy_pos in relative_enemy_pieces:
      dist = enemy_pos - player_pieces[piece]
      # Check if you can kill enemy
      if dist == dice:
        move_piece = piece
  
  if move_piece==-1:
    # If cann't kill then be safe
    min_dist = 0
    for piece in pieces_that_can_move:
      if player_pieces[piece] not in safe_positions:
        for enemy_pos in relative_enemy_pieces:
          dist = enemy_pos - player_pieces[piece]
          # Make yourself safe
          if dist < min_dist:
            min_dist = dist
            move_piece = piece 
  
  if move_piece==-1:
    move_piece = pieces_that_can_move[np.random.randint(0, len(pieces_that_can_move))]

  return move_piece

#####################################################################################################
## SETUP GLOBAL VARIABLES FOR GAMES SIMULATION
GUI = False
FIGURE_SIZE = (9.6, 7.2)  #In Inches. Try to keep the aspect ratio in 4:3
SLEEP_TIME = 2 # In Second. Wait time between two moves proper visualization
GAME_LENGTH = 10 #"FULL" # 24 or 36 or "FULL"
#####################################################################################################


## Simulation Testing
def GameSimulation(run_num=1, num_round=GAME_LENGTH, print_each_result=False, game_type=0, gui=False, figure_size=FIGURE_SIZE, wait_time=SLEEP_TIME, save_hist=False):
  player_2_type = {
    0: "Random Player",
    1: "Q-Learner Player"
  }
  print(f"We will simulate {run_num} game each for {num_round} rounds. These games will be Strategic player vs {player_2_type[game_type]}")
  print(f"Player 0 (GREEN Pieces) is Strategic player. \nPlayer 2 (BLUE Pieces) is {player_2_type[game_type]}")
  
    
  ## Winners List
  winners =[]
  start_time = time.time()
  
  for run in range(run_num):
    ## GAME Setup   
    if NUM_PLAYERS==2:
      g = ludopy.Game(ghost_players=[1, 3], strategic_player=0)  # This will prevent players 1 and 3 from moving out of the start and thereby they are not in the game
    else:
      g = ludopy.Game()
    if game_type==1:
      player2 = Qplayer(tableName=os.path.join(qlearn_dir, 'src/Qs.txt'))

    there_is_a_winner = False
    while not there_is_a_winner:
      (dice, move_pieces, player_pieces, enemy_pieces, player_is_a_winner, there_is_a_winner), player_i = g.get_observation()
        #print(f"{player_i}: {dice}, {move_pieces}  {player_pieces}")
      if gui:
        gui_on(game=g, FIGURE_SIZE=figure_size, SLEEP_TIME=wait_time)
      
      if player_i==0:
        piece_to_move = strategy(move_pieces, player_pieces, enemy_pieces, dice)
      elif player_i==2:
        if game_type==0:
          if len(move_pieces):
            piece_to_move = move_pieces[np.random.randint(0, len(move_pieces))]
          else:
            piece_to_move = -1
        elif game_type==1:
          player2.nextmove(player_i, player_pieces, enemy_pieces, dice, move_pieces)
          piece_to_move = player2.piece
        
      _, _, _, _, _, there_is_a_winner = g.answer_observation(piece_to_move)
      
      if there_is_a_winner and GUI:
        gui_on(game=g, FIGURE_SIZE=FIGURE_SIZE, SLEEP_TIME=SLEEP_TIME)
      
      if num_round!="FULL":
        if g.hist[-1][-1]==num_round+1:
          break

    winner = g.get_winner_of_game()
    if winner==-1:
      points_list = g.points_list
      winner = points_list.index(max(points_list))

    if print_each_result:
      print(f"Winner is: Player-{winner} {player_color_dict[winner]}")
    if save_hist:
      print("Saving history to numpy file")
      g.save_hist(f"game_history_random.npy")
      print("Saving game video")
      g.save_hist_video(f"game_video_random.mp4")
    
    winners.append(winner)

  end_time = time.time()
  total_time = end_time - start_time
  print(f"No of Rounds: {num_round}, No of runs: {run_num}, Total time: {total_time:.3f} sec, Average time: {total_time/run_num:.3f} sec")
  strategic_win, player2_winn = winners.count(0), winners.count(2)
  print(f"Strategic won {strategic_win} games out of {run_num} games. Winning percentage is: {strategic_win/run_num:.3f}")
  print(f"{player_2_type[game_type]} won {player2_winn} games out of {run_num} games. Winning percentage is: {player2_winn/run_num:.3f}")
  return winners




##########################################################################
## Play The GAME
##########################################################################
GUI = False
FIGURE_SIZE = (9.6, 7.2)  #In Inches. Try to keep the aspect ratio in 4:3
SLEEP_TIME = 2 # In Second. Wait time between two moves proper visualization
GAME_LENGTH = 12 #"FULL" # 24 or 36 or "FULL"

for i in range(1, 13):
  print(f"######################## NUM_ROUNDS={i} ###########################")
  winners = GameSimulation(run_num=1000, num_round=i, print_each_result=False, game_type=1)
  print("\n\n")