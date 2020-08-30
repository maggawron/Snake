import Tensorflow as tf
import numpy as np
from tensorflow import keras
from DQN import DQNAgent
from random import randint
from keras.utils import to_categorical
from tf_agents.environments import py_environment
from tf_agents.environments import tf_environment
from tf_agents.environments import tf_py_environment
from tf_agents.environments import utils
from tf_agents.specs import array_spec
from tf_agents.environments import wrappers
from tf_agents.environments import suite_gym
from tf_agents.trajectories import time_step as ts
import main

# Define model hyperparameters
num_iterations = 20000
initial_collect_steps = 1000
collect_steps_per_iteration = 1
replay_buffer_max_length = 100000

batch_size = 64
learning_rate = 1e-3
log_interval = 200

num_eval_episodes = 10
eval_interval = 1000


# Convert snake game into TensorFlow environment
class Environment(py_environment.PyEnvironment):

    def __init__(self):
        # Action is a snake move: 0: up, 1: down, 2: left, 3: right, 4: None
        self._action_spec = array_spec.BoundedArraySpec(
            shape=(), dtype=np.int32, minimum=0, maximum=4, name='action')
        self.length = main.main().length
        self.width = main.main().width
        self.snake_loc = set(main.State(self.width, self.length).snake_loc)
        self.obstacle_loc = set(main.State(self.width, self.length).obstacle_loc)
        self.apple_loc = set(main.State(self.width, self.length).apple_loc)
        # Observation is the state of the environment.
        # The observation specification has specifications of observations provided by the environment.
        # As the board has width * length positions, the shape of an observation is (1, width * length).
        # 0: empty field
        # 1: obstacle / snake
        # 2: apple
        self._observation_spec = array_spec.BoundedArraySpec(
            shape=(1, self.length * self.width), dtype=np.int32, minimum=0, maximum=2, name='observation')

        # Generated in a separate method _update_state
        self.state = self._update_state()
        self._episode_ended = False

        # Do we need this?
        self.reward = 0  # -1 point for move, +10 points for apple, -20 points for crash

    def _update_state(self):
        self.state = []
        for row in range(self.length):
            for col in range(self.width):
                if (row, col) in self.apple_loc:
                    self.state.append(2)
                elif (row, col) in self.obstacle_loc:
                    self.state.append(1)
                elif (row, col) in self.snake_loc:
                    self.state.append(1)
                else:
                    self.state.append(0)
        return self.state

    def action_spec(self):
        return self._action_spec

    def observation_spec(self):
        return self._observation_spec

    def _reset(self):
        # state at the start of the game

        #Need to write a function that will reset a game
        self.state = self._update_state()

        self._episode_ended = False
        return ts.restart(np.array([self._state], dtype=np.int32))

    def _step(self, action):
        #Do we need this?
        self.reward = array_spec.BoundedArraySpec(
            shape=(), dtype=np.int32, minimum=0, name='reward')

        if self._episode_ended:
            return self.reset()
        if self.__is_spot_empty(action):
            self._state[action] = 1

            if self.__all_spots_occupied():
                self._episode_ended = True
                return ts.termination(np.array([self._state], dtype=np.int32), 1)
            else:
                return ts.transition(np.array([self._state], dtype=np.int32), reward=0.05, discount=1.0)
        else:
            self._episode_ended = True
            return ts.termination(np.array([self._state], dtype=np.int32), -1)

    #Create train and evaluation evirornments for Tensorflow
    train_py_env = Environment(py_environment.PyEnvironment)
    train_env = tf_py_environment.TFPyEnvironment(train_py_env)

    eval_py_env = Environment(py_environment.PyEnvironment)
    eval_env = tf_py_environment.TFPyEnvironment(eval_py_env)