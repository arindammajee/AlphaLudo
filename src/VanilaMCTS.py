import numpy as np
from copy import deepcopy
import random


class Node:
    def __init__(self, game_state, parent=None, move=None, player=2):
        self.game_state = game_state
        self.parent = parent
        self.move = move
        self.available_actions = game_state.current_move_pieces
        self.depth = 0 if parent is None else parent.depth + 1
        self.player = player
        self.children = {}
        self.total_rewards = 0
        self.visits = 0
        self.is_terminal = False if game_state.get_winner_of_game() == -1 else True
        self.fully_expanded = False

    def uct_value(self, exploration_param=2):
        if self.visits == 0:
            return float("inf")
        else:
            return self.total_rewards / self.visits + exploration_param * np.sqrt(
                2 * np.log(self.parent.visits) / self.visits)


class MCTS:
    def __init__(self, game):
        self.game = game
        self.root = Node(deepcopy(self.game))

    def selection(self, node, node_depth=5):
        while not node.is_terminal and node.depth <= node_depth:
            if node.fully_expanded:
                return self.get_best_child(node)
            else:
                self.expand(node)

        return node

    def expand(self, node):
        (dice, move_pieces, player_pieces, enemy_pieces, player_is_a_winner,
         there_is_a_winner), player_i = node.game_state.dummy_obs
        for move in move_pieces:
            if move not in node.children:
                child_node = deepcopy(node.game_state)
                child_node.observation_pending = True
                #_, _ = child_node.get_observation()
                child_node.answer_observation(move)
                _, _ = child_node.get_observation()
                child_node = Node(child_node, parent=node, move=move)
                node.children[move] = child_node
                if len(move_pieces) == len(node.children):
                    node.fully_expanded = True

    def rollout(self, node):
        game = node.game_state

        while game.get_winner_of_game()==-1:
            try:
                (dice, move_pieces, player_pieces, enemy_pieces, player_is_a_winner,
                 there_is_a_winner), player_i = game.get_observation()
            except:
                raise ValueError

            if len(move_pieces):
                piece_to_move = move_pieces[np.random.randint(0, len(move_pieces))]
            else:
                piece_to_move = -1

            game.answer_observation(piece_to_move)

        return game.points[2] - game.points[0]

    def backpropogate(self, node, reward):
        while node is not None:
            node.visits += 1
            node.total_rewards += reward
            node = node.parent

    def get_best_child(self, node, exploration_param=2):
        best_value = float("-inf")
        best_nodes = []
        for child in node.children.values():
            value = child.uct_value(exploration_param)
            if value > best_value:
                best_value = value
                best_nodes = [child]
            elif value == best_value:
                best_nodes.append(child)
        return random.choice(best_nodes)

    def print_nodes(self, node):
        while node is not None:
            space = " "
            for child in node.children.values():
                node = child
                print(f"{space} Depth {node.depth} Node {node}, Node Points {node.total_rewards}, children {node.children}")
                space = "    " * len(space)

            if not node.children:
                break

    def executeRound(self, node_depth=3):
        # execute a selection-expansion-simulation-backpropagation round
        node = self.selection(self.root, node_depth=node_depth)
        print(f"Depth {node.depth} Parent {node.parent} Node {node}, Node Points {node.total_rewards}, children {node.children}")
        for child in self.root.children.values():
            print(f"         Child Node {child}, Parent {node.parent} Node Points {child.total_rewards}, children {child.children}")
        print(self.print_nodes(self.root))
        node.game_state.observation_pending = False
        #print("Starting Rollout Round")
        reward = self.rollout(node)
        #print("Starting Backpropagation")
        self.backpropogate(node, reward)
        #print("Finished Rollout Round", self.root.visits)



    def search(self, iterations=100, node_depth=5, exploration_param=2):
        for i in range(iterations):
            self.executeRound(node_depth=node_depth)

        best_child = self.get_best_child(self.root)

        return best_child.move