# Derived from LUDOpy, Copyright (c) 2020 SimonLBS, MIT License.
# The MIT notice is preserved in LICENSE.MIT next to this file.
# AlphaLudo modifications are described in the repository README.
# This notice does not relicense the file.

import numpy as np

RULESET_CLASSIC = "classic"
RULESET_LEGACY = "legacy"
RULESETS = (RULESET_CLASSIC, RULESET_LEGACY)

TOTAL_NUMBER_OF_TAILES = 58
DICE_MOVE_OUT_OF_HOME = 6
NO_ENEMY = -1

# This roule is that if, there are two pieces on the field, the last one has to return is to start
PLAY_WITH_RULE_A = True

TAILE_FREE = 0
TAILE_HOME = 1
TAILE_START = 2
TAILE_GLOB = 3
TAILE_GOAL_AREAL = 4
TAILE_STAR = 5
TAILE_GOAL = 6
TAILE_ENEMY_1_GLOB = 7
TAILE_ENEMY_2_GLOB = 8
TAILE_ENEMY_3_GLOB = 9
LIST_TAILE_ENEMY_GLOBS = [TAILE_ENEMY_1_GLOB, TAILE_ENEMY_2_GLOB, TAILE_ENEMY_3_GLOB]

NULL_POS = -1
HOME_INDEX = 0
START_INDEX = 1
STAR_INDEXS = [5, 12, 18, 25, 31, 38, 44, 51]

HOME_AREAL_INDEXS = [52, 53, 54, 55, 56]
GOAL_INDEX = 57
GLOB_INDEXS = [9, 22, 35, 48]
ENEMY_1_GLOB_INDX = 14
ENEMY_2_GLOB_INDX = 27
ENEMY_3_GLOB_INDX = 40
STAR_AT_GOAL_AREAL_INDX = STAR_INDEXS[-1]

BORD_TILES = np.full(TOTAL_NUMBER_OF_TAILES, TAILE_FREE)
BORD_TILES[HOME_INDEX] = TAILE_HOME
BORD_TILES[START_INDEX] = TAILE_START
BORD_TILES[STAR_INDEXS] = TAILE_STAR
BORD_TILES[GLOB_INDEXS] = TAILE_GLOB
BORD_TILES[HOME_AREAL_INDEXS] = TAILE_GOAL_AREAL
BORD_TILES[GOAL_INDEX] = TAILE_GOAL
BORD_TILES[ENEMY_1_GLOB_INDX] = TAILE_ENEMY_1_GLOB
BORD_TILES[ENEMY_2_GLOB_INDX] = TAILE_ENEMY_2_GLOB
BORD_TILES[ENEMY_3_GLOB_INDX] = TAILE_ENEMY_3_GLOB

ENEMY_1_INDX_AT_HOME = 40  # HOME_AREAL_INDEXS[0] - 6 - i * 13 # i = 1
ENEMY_2_INDX_AT_HOME = 27  # HOME_AREAL_INDEXS[0] - 6 - i * 13 # i = 2
ENEMY_3_INDX_AT_HOME = 14  # HOME_AREAL_INDEXS[0] - 6 - i * 13 # i = 3


def enemy_pos_at_pos(pos):
    """
    Returns the index's the other players has to be in to be in the same location as the one given in pos

    :param pos: The location to check for
    :type pos: int
    :return enemy_pos: The locations the enemy's pieces has to be at
    :rtype enemy_pos: list of list
    """
    enemy_pos = []

    for enemy_start_pos, enemy_pos_at_start in [[ENEMY_1_GLOB_INDX, ENEMY_1_INDX_AT_HOME],
                                                [ENEMY_2_GLOB_INDX, ENEMY_2_INDX_AT_HOME],
                                                [ENEMY_3_GLOB_INDX, ENEMY_3_INDX_AT_HOME]]:
        post_offset = enemy_start_pos - 1
        pre_offset = enemy_pos_at_start - 1

        if pos == enemy_start_pos:
            pos_enemy = [START_INDEX, HOME_AREAL_INDEXS[0]]
        elif pos < 0:
            pos_enemy = [max(enemy_pos_at_start + pos, -1)]
        elif START_INDEX <= pos < enemy_start_pos:
            pos_enemy = [pos + pre_offset]
        elif pos > HOME_AREAL_INDEXS[0] or pos == HOME_INDEX:
            pos_enemy = [-1]
        else:
            pos_enemy = [pos - post_offset]
        enemy_pos.append(pos_enemy)

    return enemy_pos


def validate_ruleset(ruleset):
    """Return ruleset if it is one of the two supported bundles."""
    if ruleset not in RULESETS:
        raise ValueError(
            f"Unknown ruleset {ruleset!r}. Expected {RULESET_CLASSIC!r} or {RULESET_LEGACY!r}."
        )
    return ruleset


def pass_turn(ruleset, round_number, dice, pieces, start_attempts):
    """Decide whether the turn passes, and the updated opening-attempt count.

    classic follows upstream LUDOpy. While every piece is still at home in
    round 1, the turn is kept when the attempt counter is below 3. A player
    who never rolls a 6 therefore rolls four times, and the fourth roll
    passes the turn. After round 1 a 6 grants another roll. legacy always
    passes the turn. That is the behaviour this fork had before the ruleset
    switch, which the old notebooks were measured on.
    """
    validate_ruleset(ruleset)
    if ruleset == RULESET_LEGACY:
        return True, start_attempts
    still_home = all(int(piece) == 0 for piece in pieces)
    if round_number == 1 and still_home and start_attempts < 3:
        return False, start_attempts + 1
    start_attempts = 0
    if round_number != 1 and dice == 6:
        return False, start_attempts
    return True, start_attempts


def get_enemy_at_pos(pos, enemys):
    """
    Returns the enemy's and the pieces they have at the given location

    :param pos: The location to check for
    :type pos: int
    :param enemys: The locations for the enemy's pieces in a list of 4 lists

    :returns:
    - enemy_at_pos: The enemy's there are at the location
    - enemy_pieces_at_pos: The pieces the enemy's has at the location
    :rtype enemy_at_pos: list
    :rtype enemy_pieces_at_pos: list of list

    """
    # Get the pos the enemy's has to be at to be at the same pos
    other_enemy_pos_at_pos = enemy_pos_at_pos(pos)
    # Check if there is a enemy and how many pieces the enemy has there
    enemy_at_pos = NO_ENEMY
    enemy_pieces_at_pos = []

    for enemy_i, other_enemy_pos in enumerate(other_enemy_pos_at_pos):
        # Check if there already is found a enemy at pos.
        if enemy_at_pos != NO_ENEMY:
            # If there is then stop checking for more (there can only be one)
            break

        for o_pos in other_enemy_pos:
            if o_pos == NULL_POS:
                continue

            for enemy_pice, enemy_pos in enumerate(enemys[enemy_i]):
                if enemy_pos == o_pos:
                    enemy_pieces_at_pos.append(enemy_pice)
                    enemy_at_pos = enemy_i

    return enemy_at_pos, enemy_pieces_at_pos


class Player:
    """
    A class used by the Game class. This class is not needed for normal use
    """

    def __init__(self, aggresive=False, responsible=True, out_of_home=None, ruleset=RULESET_CLASSIC):
        """
        Makes a player with 4 pieces all at the home locations.

        ruleset selects a whole bundle of movement rules. out_of_home is
        accepted so older callers do not raise, and is stored, but movement
        follows ruleset. classic leaves home only on a 6. legacy leaves home
        on any face.
        """
        self.ruleset = validate_ruleset(ruleset)
        self.pieces = []
        self.number_of_pieces = 4
        self.set_all_pieces_to_home()
        self.out_of_home = self.ruleset == RULESET_LEGACY if out_of_home is None else bool(out_of_home)
        self.aggresive = aggresive
        self.responsible = responsible

    def get_pieces_that_can_move(self, dice):
        """
        Return the pieces that can move with the given dice

        :param dice: The dice the move will be done with
        :type dice: int
        :return: movable_pieces: A list with the pieces that can be moved
        :rtype movable_pieces: list

        """
        movable_pieces = []
        # Go though all the pieces
        for piece_i, piece_place in enumerate(self.pieces):
            # If the piece is a goal then the piece can't move
            if BORD_TILES[piece_place] == TAILE_GOAL:
                continue

            elif BORD_TILES[piece_place] == TAILE_HOME:
                # classic: a 6 is required. legacy: any face leaves home.
                if self.ruleset == RULESET_LEGACY or dice == DICE_MOVE_OUT_OF_HOME:
                    movable_pieces.append(piece_i)
            # If the piece is not at home or at the goal it can move
            elif BORD_TILES[piece_place] != TAILE_HOME:
                movable_pieces.append(piece_i)
    
        return movable_pieces

    def player_winner(self):
        """
        Returns rather the player is a winner or not

        :return: winner: A bool that indicate rather the player is a winner or not
        :rtype winner: bool
        """
        # Go though all the pieces
        for piece_place in self.pieces:
            # If a piece is not at the goal is not the winner
            if BORD_TILES[piece_place] != TAILE_GOAL:
                return False
        # If no piece was not at the goal the player is the winner
        return True

    def set_pieces(self, pieces):
        """
        Sets the players pieces

        :param pieces: The pieces to set the players pieces to
        """
        self.pieces = np.copy(pieces)

    def get_pieces(self):
        """
        Returns the players pieces

        :return pieces: The players pieces
        :rtype pieces: list
        """
        return np.copy(self.pieces)

    def move_piece(self, piece, dice, enemys):
        """
        Move the players piece the given dice following the game rules. Returns the new locations of the enemy's pieces

        :param piece: The piece to move
        :type piece: int
        :param dice: The dice to make the move with
        :type dice: int
        :param enemys: The enemy's pieces
        :type enemys: list with 4 lists each with 4 int's
        :return enemys: The new locations of the enemy's pieces
        :rtype enemys: list with 4 lists each with 4 int's

        """
        enemys_new = enemys.copy()
        old_piece_pos = self.pieces[piece]
        new_piece_pos = old_piece_pos + dice

        move_enemy_home_from_poss = []
        do_not_check_rule_a = False
        enemy_at_pos, enemy_pieces_at_pos = get_enemy_at_pos(new_piece_pos, enemys)

        # If the dice is 0 then no movement can be done
        if dice == 0:
            pass

        # At goal
        elif BORD_TILES[old_piece_pos] == TAILE_GOAL:
            # The piece can not move
            pass

        # Goal areal
        elif BORD_TILES[old_piece_pos] == TAILE_GOAL_AREAL:
            if new_piece_pos <= GOAL_INDEX:
                self.pieces[piece] = new_piece_pos
            elif self.ruleset == RULESET_CLASSIC:
                overshoot = new_piece_pos - GOAL_INDEX
                self.pieces[piece] = old_piece_pos - overshoot
            # legacy: an overshoot does not move the piece

        # The Home areal
        elif BORD_TILES[old_piece_pos] == TAILE_HOME:
            if self.ruleset == RULESET_CLASSIC and dice != DICE_MOVE_OUT_OF_HOME:
                pass
            else:
                self.pieces[piece] = START_INDEX
                # An enemy standing on the start square is sent home.
                do_not_check_rule_a = True
                move_enemy_home_from_poss.append(START_INDEX)

        # Star before the home areal. Landing exactly on it enters the goal.
        elif new_piece_pos == STAR_AT_GOAL_AREAL_INDX:
            self.pieces[piece] = GOAL_INDEX
            if self.ruleset == RULESET_CLASSIC:
                move_enemy_home_from_poss.append(new_piece_pos)

        # The other stars
        elif BORD_TILES[new_piece_pos] == TAILE_STAR and self.ruleset == RULESET_CLASSIC:
            present_star_staridx = STAR_INDEXS.index(new_piece_pos)
            next_star_staridx = present_star_staridx + 1
            if next_star_staridx >= len(STAR_INDEXS):
                next_star_staridx = 0
            next_star_pos = STAR_INDEXS[next_star_staridx]
            self.pieces[piece] = next_star_pos
            if enemy_at_pos != NO_ENEMY:
                move_enemy_home_from_poss.append(new_piece_pos)
            next_star_enemy_at_pos, _next_pieces = get_enemy_at_pos(next_star_pos, enemys)
            if next_star_enemy_at_pos != NO_ENEMY:
                move_enemy_home_from_poss.append(next_star_pos)

        elif BORD_TILES[new_piece_pos] == TAILE_STAR:
            # legacy: stay on the star. No jump and no capture.
            self.pieces[piece] = new_piece_pos

        # Globs there are not own by enemy
        elif BORD_TILES[new_piece_pos] == TAILE_GLOB:
            if enemy_at_pos != NO_ENEMY:
                self.pieces[piece] = HOME_INDEX
            else:
                self.pieces[piece] = new_piece_pos

        # Globs there are own by enemy
        elif BORD_TILES[new_piece_pos] in LIST_TAILE_ENEMY_GLOBS:
            # Get the enemy there own the glob
            globs_enemy = LIST_TAILE_ENEMY_GLOBS.index(BORD_TILES[new_piece_pos])
            # Check if there is a enemy at the glob
            if enemy_at_pos != NO_ENEMY:
                # If there is a other enemy then send them home and move there
                if enemy_at_pos != globs_enemy:
                    move_enemy_home_from_poss.append(new_piece_pos)
                    self.pieces[piece] = new_piece_pos
                # If it is the same enemy there is there them move there
                else:
                    self.pieces[piece] = HOME_INDEX
            # If there ant any enemy at the glob then move there
            else:
                self.pieces[piece] = new_piece_pos

        # If it is a TAILE_FREE or if we move from a GLOB/STAR to a not done case
        elif BORD_TILES[old_piece_pos] == TAILE_FREE or \
                BORD_TILES[new_piece_pos] == TAILE_FREE or \
                BORD_TILES[old_piece_pos] == TAILE_GLOB or \
                BORD_TILES[old_piece_pos] == TAILE_STAR:
            if enemy_at_pos != NO_ENEMY:
                move_enemy_home_from_poss.append(new_piece_pos)
            self.pieces[piece] = new_piece_pos

        # If the case was not caught then there is a error
        else:
            print("\nold_piece_pos:", old_piece_pos, "\nnew_piece_pos", new_piece_pos,
                  "\nBORD_TILES[old_piece_pos]:", BORD_TILES[old_piece_pos],
                  "\nBORD_TILES[new_piece_pos]:", BORD_TILES[new_piece_pos], "\ndice:", dice)
            raise RuntimeError("The new_piece_pos case was not handel")

        # Check if there is any enemy there has to be moved
        if len(move_enemy_home_from_poss):
            # Go through the pos where enemy has to be moved from
            for pos in move_enemy_home_from_poss:
                # Get the enemy at the pos
                enemy_at_pos, enemy_pieces_at_pos = get_enemy_at_pos(pos, enemys)
                # Check if there was a enemy at the pos
                if enemy_at_pos != NO_ENEMY:
                    # If there is only one enemy then move the enemy home.
                    if not do_not_check_rule_a and not PLAY_WITH_RULE_A or len(enemy_pieces_at_pos) == 1:
                        for enemy_piece in enemy_pieces_at_pos:
                            enemys_new[enemy_at_pos][enemy_piece] = HOME_INDEX
                    # If there is more than one then move own piece home
                    else:
                        self.pieces[piece] = HOME_INDEX

        return enemys_new

    def set_all_pieces_to_home(self):
        """
        Sets all the players pieces to the home index
        """
        self.pieces = []
        for i in range(self.number_of_pieces):
            self.pieces.append(HOME_INDEX)
