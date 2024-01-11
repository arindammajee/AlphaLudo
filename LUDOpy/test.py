import sys
import numpy as np
sys.path.append("/home/admin1/A-Reinforcement-Learning-Approach-to-Ludo/LUDOpy/ludopy/")
import ludopy
import time
from IPython.display import clear_output
import matplotlib.pyplot as plt


NUM_PLAYERS = 2 # Or 4
FIGURE_SIZE = (9.6, 7.2)  #In Inches. Try to keep the aspect ratio in 4:3
SLEEP_TIME = 0 # In Second. Wait time between two moves proper visualization

player_color_dict = {
    0: "GREEN (Bottom Left)",
    1: "YELLOW (Top Left)",
    2: "BLUE (Top Right)",
    3: "RED (Bottom Right)"
}


if NUM_PLAYERS==2:
  g = ludopy.Game(ghost_players=[1, 3])  # This will prevent players 1 and 3 from moving out of the start and thereby they are not in the game
else:
  g = ludopy.Game()

there_is_a_winner = False
run = 1

while not there_is_a_winner:
    (dice, move_pieces, player_pieces, enemy_pieces, player_is_a_winner, there_is_a_winner), player_i = g.get_observation()
    if run>132:
        enviroment_image_rgb = g.render_environment()  # RGB image of the enviroment
        #enviroment_image_bgr = cv2.cvtColor(enviroment_image_rgb, cv2.COLOR_RGB2BGR)
        clear_output(wait=True)
        figure = plt.figure(figsize=FIGURE_SIZE)
        plt.imshow(enviroment_image_rgb)
        plt.show()
    run += 1

    if len(move_pieces):
        piece_to_move = move_pieces[np.random.randint(0, len(move_pieces))]
    else:
        piece_to_move = -1

    _, _, _, _, _, there_is_a_winner = g.answer_observation(piece_to_move)

    time.sleep(SLEEP_TIME)

winner = g.get_winner_of_game()
print(f"Winner is: Player-{winner} {player_color_dict[winner]}")
print("Saving history to numpy file")
g.save_hist(f"game_history_random.npy")
print("Saving game video")
g.save_hist_video(f"game_video_random.mp4")
del g, player_i

"""
g = ludopy.Game()
there_is_a_winner = False

while not there_is_a_winner:
    (dice, move_pieces, player_pieces, enemy_pieces, player_is_a_winner, there_is_a_winner), player_i = g.get_observation()

    if len(move_pieces):
        piece_to_move = move_pieces[np.random.randint(0, len(move_pieces))]
    else:
        piece_to_move = -1

    _, _, _, _, _, there_is_a_winner = g.answer_observation(piece_to_move)

print("Saving history to numpy file")
g.save_hist(f"game_history.npy")
print("Saving game video")
g.save_hist_video(f"game_video.mp4")
data = np.load('game_history.npy', allow_pickle=True)
print(data.shape)
np.savetxt('game_history.txt',np.array(g.hist))
"""
