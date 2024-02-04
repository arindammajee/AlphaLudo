# Train a Q-Learner Player against Random, Aggresive and Responsible Player
import copy
import sys
import time
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

import ludopy

sys.path.append("../")
from Qlearn import *



def QLearnVsRandomTrain():
    player0 = Qplayer(newtbl=True, isTrain=True)
    ghosts = [1, 3]
    games = 60
    start_time = time.time()
    # dLUT_arr = []
    # nummoves = []
    # aiovers = []
    for i in range(0, games):
        # Before = player0.LUT.sum()
        winner, usefulmoves, overs = QLearnVsRandom(player0, ghosts)
        player0.averageqs()
        if (i % int(games / 10)) == 0:
            print(i)
        # entries = np.count_nonzero(player0.LUT)
        # dLUT_arr.append((player0.LUT.sum()-Before)/entries)
        # nummoves.append(usefulmoves)
        # aiovers.append((overs/3))
    end_time = time.time()
    print("\n", int(end_time - start_time), "Seconds")
    # df = plotLearning(np.arange(i + 1), nummoves, aiovers)
    player0.write2text()
    # print("Average overshoot by Random: " + str(np.mean(aiovers)))
    # print("Average overshoot by Qplayer: " + str(np.mean(nummoves)))
    print("Training Complete!")
    # df.to_csv('TrainingData.csv')


