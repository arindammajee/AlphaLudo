import time
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
            return (self.total_rewards / self.visits) + exploration_param * np.sqrt(
                2 * (np.log(self.parent.visits) / self.visits))


class MCTS:
    def __init__(self, game, reserved_time, num_iterations, isTimeBounded):
        self.game = game
        self.root = Node(deepcopy(self.game))
        self.num_iterations = num_iterations
        self.reserved_time = reserved_time
        self.isTimeBounded = isTimeBounded
        self.run_time = 0
        self.total_rollouts = 0

    def selection(self, node):
        while len(node.children) != 0:
            # print(f"Current node depth: {node.depth} and isFullyExpanded:
            # {node.fully_expanded} with visits: {node.visits}")
            best_child = self.get_best_child(node)
            node = best_child
            if best_child.visits == 0:
                break

        if not node.is_terminal:
            self.expand(node)

        return node

    @staticmethod
    def expand(node):
        # print("Expanding node.........")
        (dice, move_pieces, player_pieces, enemy_pieces, player_is_a_winner,
         there_is_a_winner), player_i = node.game_state.dummy_obs
        for move in move_pieces:
            if move not in node.children:
                child_node = deepcopy(node.game_state)
                child_node.observation_pending = True
                # _, _ = child_node.get_observation()
                child_node.answer_observation(move)

                # Play opponent's turn
                ((opponent_dice, opponent_move_pieces, opponent_player_pieces, opponent_enemy_pieces,
                 opponent_player_is_a_winner, opponent_there_is_a_winner),
                 opponent_player_i) = child_node.get_observation()

                if len(opponent_move_pieces):
                    opponent_piece_to_move = opponent_move_pieces[np.random.randint(0, len(opponent_move_pieces))]
                else:
                    opponent_piece_to_move = -1

                child_node.answer_observation(opponent_piece_to_move)
                _, _ = child_node.get_observation()

                # Create a new Node
                child_node = Node(child_node, parent=node, move=move)
                node.children[move] = child_node
                if len(move_pieces) == len(node.children):
                    node.fully_expanded = True

    @staticmethod
    def rollout(node):
        game = node.game_state

        while game.get_winner_of_game() == -1:
            try:
                (dice, move_pieces, player_pieces, enemy_pieces, player_is_a_winner,
                 there_is_a_winner), player_i = game.get_observation()
                if len(move_pieces):
                    piece_to_move = move_pieces[np.random.randint(0, len(move_pieces))]
                else:
                    piece_to_move = -1

                game.answer_observation(piece_to_move)

            except Exception as ex:
                print(f"This error should not occur. Please deep dive it {ex}")

        return game.points[2] - game.points[0]

    @staticmethod
    def backpropagation(node, reward):
        while node is not None:
            node.visits += 1
            node.total_rewards += reward
            node = node.parent

    @staticmethod
    def get_best_child(node, exploration_param=2):
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

    def execute_round(self):
        # execute a selection-expansion-simulation-backpropagation round
        node = self.selection(self.root)
        # print(f"Received node of depth: {node.depth} and isFullyExpanded:
        # {node.is_terminal} with visit count: {node.visits}")
        node.game_state.observation_pending = False
        reward = self.rollout(node)
        self.total_rollouts += 1
        self.backpropagation(node, reward)

    def search(self):
        start = time.time()
        if self.isTimeBounded:
            while time.time() - start < self.reserved_time:
                self.execute_round()
        else:
            for i in range(self.num_iterations):
                self.execute_round()

        best_child = self.get_best_child(self.root)
        self.run_time = time.time() - start
        # print(f"MCTS search took {self.run_time} seconds for total {self.total_rollouts} rollouts")
        return best_child.move
