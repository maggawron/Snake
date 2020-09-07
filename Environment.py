import numpy as np
from tf_agents.environments import py_environment
from tf_agents.specs import array_spec
from tf_agents.trajectories import time_step as ts
import Backend
import copy

# Convert snake game into TensorFlow environment
class Environment(py_environment.PyEnvironment):

    def __init__(self):
        self.all_moves = []
        self.length = 25
        self.width = 25
        self.stan = Backend.State(self.width, self.length)

        # Action is a snake move: 0: forward, 1: left, 2: right
        self._action_spec = array_spec.BoundedArraySpec(
            shape=(), dtype=np.int32, minimum=0, maximum=2, name='action')

        # Observation is the state of the environment.
        # The observation specification has specifications of observations provided by the environment.
        # As the board has width * length positions, the shape of an observation is (1, width * length).
        self._observation_spec = array_spec.BoundedArraySpec(
            shape=(self.length * self.width,), dtype=np.int32, minimum=0, maximum=100, name='observation')

        self._reward_spec = array_spec.BoundedArraySpec(shape=(), dtype=np.float32, name='reward')

        # Generated in a separate method _update_state
        self.state = self._update_state()
        self.episode_ended = False
        self.reset_count = 0
        self.action_count = 0
        self.apple_eaten_count = 0
        self.moves_from_last_apple = 0

    def reward_spec(self):
        return self._reward_spec

    def action_spec(self):
        return self._action_spec

    def observation_spec(self):
        return self._observation_spec

    # Coding of objects on the board
    # Empty field: 0, Obstacle: 1, Apple: 2, Snake: 3, 4 ...,
    def _update_state(self):
        state = []
        counter = 3
        for row in range(self.length):
            for col in range(self.width):
                if (row, col) in set(self.stan.apple_loc):
                    state.append(2)
                elif (row, col) in set(
                        self.stan.obstacle_loc) or row == 0 or col == 0 or row == self.length - 1 or col == self.width - 1:
                    state.append(1)
                elif (row, col) in set(self.stan.snake_loc):
                    #TODO bug
                    state.append(counter)
                    counter += 1
                else:
                    state.append(0)
        return state

    # Calls generate_game method from State to initialize a random game from scratch
    def _reset(self):
        self.stan = Backend.State(self.width, self.length)
        self.state = self._update_state()
        self.episode_ended = False
        self.reset_count += 1
        self.action_count = 0
        self.apple_eaten_count = 0
        self.moves_from_last_apple = 0
        return ts.restart(np.array(self.state, dtype=np.int32), reward_spec=self._reward_spec)

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

    def _step(self, action):
        if self.episode_ended:
            return self.reset()

        key = self._decode_action(action)

        self.stan.move_snake(key)
        self.state = self._update_state()
        # assert self.state.count(2) == 10

        self.episode_ended = not self.stan.check_if_snake_lives()
        # Make a copy of lists, so that every move is different.
        move = copy.deepcopy((self.stan.snake_loc, self.stan.apple_loc, self.stan.obstacle_loc))
        self.all_moves.append(move)

        if self.episode_ended or self.action_count > 500 or self.moves_from_last_apple > 25:
            self.episode_ended = True
            return ts.termination(np.array(self.state, dtype=np.int32), reward=-10)
        self.action_count += 1
        self.moves_from_last_apple += 1
        if self.stan.ate_apple:
            self.moves_from_last_apple = 0
            self.apple_eaten_count += 1
            return ts.transition(np.array(self.state, dtype=np.int32), reward=100)
        return ts.transition(np.array(self.state, dtype=np.int32), reward=-0.05)

