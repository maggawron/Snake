#from multiprocessing import set_start_method
#set_start_method("spawn")

import numpy as np
import tf_agents
from tf_agents.environments import py_environment
from tf_agents.specs import array_spec
from tf_agents.trajectories import time_step as ts
import Backend
from tf_agents import utils
import copy
import math

# Convert snake game into TensorFlow environment
class Environment(py_environment.PyEnvironment):

    def __init__(self):
        self.all_moves = []
        self.length = 25  #TODO
        self.width = 25  #TODO
        #self.stan = Backend.State(self.width, self.length)
        self.stan = copy.deepcopy(Backend.state_const) #TODO

        # Action is a snake move: 0: forward, 1: left, 2: right
        self._action_spec = array_spec.BoundedArraySpec(
            shape=(), dtype=np.int32, minimum=0, maximum=2, name='action')

        # Observation is the state of the environment.
        # The observation specification has specifications of observations provided by the environment.
        # As the board has width * length positions, the shape of an observation is (1, width * length).
        #self._observation_spec = array_spec.BoundedArraySpec(
        #    shape=(self.length, self.width, 5), dtype=np.float32, minimum=0, maximum=1, name='observation') #TODO

        self._observation_spec = array_spec.BoundedArraySpec(
           shape=(self.length, self.width, 5,), dtype=np.float32, minimum=0, maximum=1, name='observation') #TODO

        #self._reward_spec = array_spec.BoundedArraySpec(shape=(), dtype=np.float32, name='reward')

        # Generated in a separate method _update_state
        self.state = self._update_state() #TODO
        self.episode_ended = False
        self.reset_count = 0
        self.action_count = 0
        self.apple_eaten_count = 0
        self.moves_from_last_apple = 0
        self.prev_moves = []
        self.distance_to_apple = self.cal_dist_to_apple()

    def cal_dist_to_apple(self):
        for apple in self.stan.apple_loc:
            min_dist = math.sqrt(self.length ** 2 + self.width ** 2)
            distance = math.sqrt((self.stan.snake_loc[-1][0] - apple[0]) ** 2 + (self.stan.snake_loc[-1][1] - apple[1]) ** 2)
            min_dist = min(min_dist, distance)
        return min_dist


    def reward_spec(self):
        return array_spec.BoundedArraySpec(shape=(), dtype=np.float32, minimum=-100, maximum=30, name='reward')

    def action_spec(self):
        return self._action_spec

    def observation_spec(self):
        return self._observation_spec

    # Coding of objects on the board
    def _update_state(self):
        state = []
        obstacles_set = set(self.stan.obstacle_loc)
        apples_set = set(self.stan.apple_loc)
        for row in range(self.length):
            line = []
            for col in range(self.width):
                if (row, col) in apples_set:
                    line.append([1, 0, 0, 0, 0])
                elif (row, col) in obstacles_set \
                         or row == 0 or col == 0 or row == self.length - 1 or col == self.width - 1:
                    line.append([0, 1, 0, 0, 0])
                #TODO make it faster
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
        #self.stan = Backend.State(self.width, self.length)
        self.stan = copy.deepcopy(Backend.state_const) #TODO
        self.state = self._update_state() #TODO
        self.episode_ended = False
        self.reset_count += 1
        self.action_count = 0
        self.apple_eaten_count = 0
        self.moves_from_last_apple = 0
        self.distance_to_apple = self.cal_dist_to_apple()
        self.prev_moves = []
        return ts.restart(np.array(self.state, dtype=np.float32))

    def _decode_action(self, action):
        if action == 0:
            return self.stan.prev_key
        # Turn left
        if action == 1:
            if self.stan.prev_key == "right": return "up"
            if self.stan.prev_key == "up": return "left"
            if self.stan.prev_key == "left": return "down"
            if self.stan.prev_key == "down": return "right"
        # Turn right
        if action == 2:
            if self.stan.prev_key == "right": return "down"
            if self.stan.prev_key == "up": return "right"
            if self.stan.prev_key == "left": return "up"
            if self.stan.prev_key == "down": return "left"
        print(action)
        return "right"

    def alternative_state(self):
        state = []
        head_row, head_col = self.stan.snake_loc[-1]
        vert_distance = (self.stan.apple_loc[0][0] - head_row)/self.width
        hor_distance = (self.stan.apple_loc[0][1] - head_col)/self.length
        state.append(vert_distance)
        state.append(hor_distance)
        obstacles_set = set(self.stan.obstacle_loc) | set(self.stan.snake_loc[:-1])
        direction = {"down": [(0,-1), (1,0), (-1,0)], "up": [(0,1), (-1,0), (1,0)], "left": [(-1,0), (0,-1), (0,1)], "right": [(1,0), (0,1), (0,-1)]}
        for act in range(3):
            head_row_straight = head_row + direction[self.stan.prev_key][act][0]
            head_col_straight = head_col + direction[self.stan.prev_key][act][1]
            if (head_row_straight, head_col_straight) in obstacles_set \
                    or head_row_straight == 0 or head_col_straight == 0 \
                    or head_row_straight == self.length - 1 or head_col_straight == self.width - 1:
                state.append(1)
            else:
                state.append(0)

        #print("State", state)
        return state

    def _step(self, action):
        if self.episode_ended:
            return self.reset()

        key = self._decode_action(action)

        self.stan.move_snake(key)
        #print("Snake loc:", self.stan.snake_loc, "apple loc", self.stan.apple_loc)
        self.state = self._update_state() #TODO
        # assert self.state.count(2) == 10

        self.episode_ended = not self.stan.check_if_snake_lives()
        # Make a copy of lists, so that every move is different.
        move = copy.deepcopy((self.stan.snake_loc, self.stan.apple_loc, self.stan.obstacle_loc))
        self.all_moves.append(move)
        prev_dist = self.distance_to_apple
        self.distance_to_apple = self.cal_dist_to_apple()
        self.prev_moves.append(action)
        if len(self.prev_moves) >= 4:
            self.prev_moves.pop(0)

        if self.episode_ended or self.action_count > 500 or self.moves_from_last_apple > 25:
            self.episode_ended = True
            return ts.termination(np.array(self.state, dtype=np.float32), reward=-100)
        self.action_count += 1
        self.moves_from_last_apple += 1
        if self.stan.ate_apple:
            self.moves_from_last_apple = 0
            self.apple_eaten_count += 1
            return ts.transition(np.array(self.state, dtype=np.float32), reward=30)
        if len(self.prev_moves) == 4 and self.prev_moves[0] == self.prev_moves[1] == self.prev_moves[2] == self.prev_moves[3] and self.prev_moves[2] > 0:
            return ts.transition(np.array(self.state, dtype=np.float32), reward=-15)
        """
        if self.distance_to_apple >= prev_dist:
            return ts.transition(np.array(self.state, dtype=np.float32), reward=-self.distance_to_apple / 100)
        return ts.transition(np.array(self.state, dtype=np.float32), reward=1-self.distance_to_apple/100)
        """
        return ts.transition(np.array(self.state, dtype=np.float32), reward=-0.005)
