import numpy as np
from copy import deepcopy
import importlib
import sys
sys.path.append("LUDOpy")
from ludopy import Game


class Node:
    def __init__(self, game_state, parent=None, move=None):
        self.game_state = game_state
        self.parent = parent
        self.move = move
        self.children = []
        self.wins = 0
        self.visits = 0
    

    def uct_value(self, exploration_param=1):
        if self.visits == 0:
            return np.inf
        else:
            return self.wins / self.visits + exploration_param * np.sqrt(
                2 * np.log(self.parent.visits) / self.visits)

    def best_child(self):
        return max(self.children, key=lambda node: node.uct_value())

    def expand(self):
        (dice, move_pieces, player_pieces, enemy_pieces, player_is_a_winner,
         there_is_a_winner), player_i = self.game_state.dummy_obs

        # print(move_pieces)
        for move in move_pieces:
            # print("Move", move)
            new_game_state = deepcopy(self.game_state)
            new_game_state.observation_pending = False
            _, _ = new_game_state.get_observation()
            new_game_state.answer_observation(move)
            self.children.append(Node(new_game_state, parent=self, move=move))

    def update(self, result):
        self.visits += 1
        self.wins += result

class MCTS:
    def __init__(self, game):
        self.game = game

    def search(self, iterations=20):
        root = Node(deepcopy(self.game))
        root.expand()

        for _ in range(iterations):
            node = root
            game = deepcopy(self.game)

            # Selection
            while node.children:
                # print(node.best_child())
                node = node.best_child()
                game.answer_observation(node.move)

            # Expansion
            if not game.get_winner_of_game():
                node.expand()
                node = node.children[-1]
                game.answer_observation(node.move)

            # Simulation
            while not game.get_winner_of_game():
                (dice, move_pieces, player_pieces, enemy_pieces, player_is_a_winner,
                 there_is_a_winner), player_i = game.get_observation()

                if len(move_pieces):
                    piece_to_move = move_pieces[np.random.randint(0, len(move_pieces))]
                else:
                    piece_to_move = -1

                game.answer_observation(piece_to_move)

            # Backpropagation
            result = game.points[2]
            while node is not None:
                node.update(result)
                node = node.parent
                #print("Result", result)
            

        return root.best_child().move