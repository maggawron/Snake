import numpy as np
from tf_agents.environments import py_environment
from tf_agents.specs import array_spec
from tf_agents.trajectories import time_step as ts
from State import State
import copy
import math

"""Convert snake game into TensorFlow environment"""


class Environment(py_environment.PyEnvironment):

    def __init__(self):
        self._start_new_game()

        # Action is a snake move: 0: forward, 1: left, 2: right
        self._action_spec = array_spec.BoundedArraySpec(
            shape=(), dtype=np.int32, minimum=0, maximum=2, name='action')

        # Observation is the state of the environment.
        self._observation_spec = array_spec.BoundedArraySpec(
            shape=(self.stan.game_length, self.stan.game_width, 5,), dtype=np.float32, minimum=0, maximum=1, name='observation')

    def _start_new_game(self):
        self.stan = State()
        # self.stan = copy.deepcopy(Backend.state_const)
        self.state = self._update_state()
        self.episode_ended = False
        self.action_count = 0
        self.moves_from_last_apple = 0
        self.distance_to_apple = self._cal_dist_to_apple()
        self.all_moves = []
        self.reward_print = 0
        self.total_reward = 0

    def _cal_dist_to_apple(self):
        min_dist = math.sqrt(self.stan.game_length ** 2 + self.stan.game_width ** 2)
        for apple in self.stan.apple_loc:
            distance = math.sqrt(
                (self.stan.snake_loc[-1][0] - apple[0]) ** 2 + (self.stan.snake_loc[-1][1] - apple[1]) ** 2)
            min_dist = min(min_dist, distance)
        return min_dist

    def action_spec(self):
        return self._action_spec

    def observation_spec(self):
        return self._observation_spec

    # Coding of objects on the board
    def _update_state(self):
        state = []
        for row in range(self.stan.game_length):
            line = []
            for col in range(self.stan.game_width):
                if (row, col) in self.stan.apple_loc:
                    line.append([1, 0, 0, 0, 0])
                elif (row, col) in self.stan.obstacle_loc \
                        or row == 0 or col == 0 or row == self.stan.game_length - 1 or col == self.stan.game_width - 1:
                    line.append([0, 1, 0, 0, 0])
                # TODO make it faster
                elif (row, col) in self.stan.snake_loc[-1]:
                    line.append([0, 0, 1, 0, 0])
                elif (row, col) in self.stan.snake_loc[:-1]:
                    line.append([0, 0, 0, 1, 0])
                else:
                    line.append([0, 0, 0, 0, 1])
            state.append(line)
        return state

    # Calls generate_game method from State to initialize a random game from scratch
    def _reset(self):
        self._start_new_game()
        return ts.restart(np.array(self.state, dtype=np.float32))

    def _decode_action(self, action):
        if action == 0:
            return self.stan.prev_key
        # Turn left
        turn_left = {"right": "up", "up": "left", "left": "down", "down": "right"}
        if action == 1:
            return turn_left[self.stan.prev_key]
        # Turn right
        turn_right = {"right": "down", "up": "right", "left": "up", "down": "left"}
        if action == 2:
            return turn_right[self.stan.prev_key]
        return "right"

    def _step(self, action):
        if self.episode_ended:
            return self.reset()

        key = self._decode_action(action)
        self.stan.move_snake(key)
        self.state = self._update_state()

        self.episode_ended = not self.stan.check_if_snake_lives()
        # Make a copy of lists, so that every move is different.
        move = copy.deepcopy((self.stan.snake_loc, self.stan.apple_loc,
                              self.stan.obstacle_loc, self.reward_print,
                              self.total_reward, self.stan.points))
        self.all_moves.append(move)
        self.total_reward += self.reward_print
        if self.episode_ended or self.action_count > 500 or self.moves_from_last_apple > 50:
            self.episode_ended = True
            self.reward_print = -100
            return ts.termination(np.array(self.state, dtype=np.float32), reward=-100)

        self.distance_to_apple = self._cal_dist_to_apple()
        self.action_count += 1
        self.moves_from_last_apple += 1
        if self.stan.ate_apple:
            self.moves_from_last_apple = 0
            self.reward_print = 30
            self.stan.ate_apple = False
            return ts.transition(np.array(self.state, dtype=np.float32), reward=30)

        if len(self.all_moves) >= 5 and self.all_moves[-1][0][-1] == self.all_moves[-5][0][-1]:
            self.reward_print = -5
            return ts.transition(np.array(self.state, dtype=np.float32), reward=-5)
        """
        if self.distance_to_apple >= prev_dist:
            return ts.transition(np.array(self.state, dtype=np.float32), reward=-self.distance_to_apple / 100)
        return ts.transition(np.array(self.state, dtype=np.float32), reward=1-self.distance_to_apple/100)
        """
        self.reward_print = -0.005
        return ts.transition(np.array(self.state, dtype=np.float32), reward=-0.005)

    def _alternative_state(self):
        state = []
        head_row, head_col = self.stan.snake_loc[-1]
        vert_distance = (self.stan.apple_loc[0][0] - head_row) / self.width
        hor_distance = (self.stan.apple_loc[0][1] - head_col) / self.length
        state.append(vert_distance)
        state.append(hor_distance)
        obstacles_set = set(self.stan.obstacle_loc) | set(self.stan.snake_loc[:-1])
        direction = {"down": [(0, -1), (1, 0), (-1, 0)], "up": [(0, 1), (-1, 0), (1, 0)],
                     "left": [(-1, 0), (0, -1), (0, 1)], "right": [(1, 0), (0, 1), (0, -1)]}
        for act in range(3):
            head_row_straight = head_row + direction[self.stan.prev_key][act][0]
            head_col_straight = head_col + direction[self.stan.prev_key][act][1]
            if (head_row_straight, head_col_straight) in obstacles_set \
                    or head_row_straight == 0 or head_col_straight == 0 \
                    or head_row_straight == self.length - 1 or head_col_straight == self.width - 1:
                state.append(1)
            else:
                state.append(0)
        return state
