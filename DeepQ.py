import tensorflow as tf
import numpy as np
import tf_agents
from tf_agents.agents.dqn import dqn_agent
from tf_agents.networks import q_network
from tf_agents.environments import py_environment
from tf_agents.environments import tf_py_environment
from tf_agents.specs import array_spec
from tf_agents.trajectories import time_step as ts
from tf_agents.utils import common
from tf_agents.specs import BoundedArraySpec
from tf_agents.environments import utils
from tf_agents.policies import random_tf_policy
from tf_agents.replay_buffers import tf_uniform_replay_buffer, replay_buffer
from tf_agents.trajectories import trajectory
import Backend
import state
import Screen

# Define model hyperparameters
num_iterations = 20000
initial_collect_steps = 1000
collect_steps_per_iteration = 10
replay_buffer_max_length = 100

batch_size = 1
learning_rate = 3e-5
log_interval = 100

num_eval_episodes = 10
eval_interval = 1000


# Convert snake game into TensorFlow environment
class Environment(py_environment.PyEnvironment):

    def __init__(self):
        self.stan = state.State(50, 10)
        # Action is a snake move: 0: forward, 1: left, 2: right
        self._action_spec = array_spec.BoundedArraySpec(
            shape=(), dtype=np.int32, minimum=0, maximum=2, name='action')
        self.length = self.stan.length
        self.width = self.stan.width
        self.snake_loc = self.stan.snake_loc
        self.obstacle_loc = self.stan.obstacle_loc
        self.apple_loc = self.stan.apple_loc
        self.prev_key = self.stan.prev_key

        # Observation is the state of the environment.
        # The observation specification has specifications of observations provided by the environment.
        # As the board has width * length positions, the shape of an observation is (1, width * length).
        self._observation_spec = array_spec.BoundedArraySpec(
            shape=(self.length * self.width,), dtype=np.int32, minimum=0, maximum=100, name='observation')

        self._reward_spec = array_spec.BoundedArraySpec(shape=(), dtype=np.float32, name='reward')

        # Generated in a separate method _update_state
        self.state = self._update_state()
        self.episode_ended = False
        self.apple_eaten = self.stan.ate_apple
        self.reset_count = 0
        self.action_count = 0
        self.apple_eaten_count = 0

    # Coding of objects on the board
    # Empty field: 0, Obstacle: 1, Apple: 2, Snake: 3, 4 ...,
    def _update_state(self):
        self.state = []
        counter = 3
        for row in range(self.length):
            for col in range(self.width):
                if (row, col) in set(self.apple_loc):
                    self.state.append(2)
                elif (row, col) in set(
                        self.obstacle_loc) or row == 0 or col == 0 or row == self.length - 1 or col == self.width - 1:
                    self.state.append(1)
                elif (row, col) in set(self.snake_loc):
                    self.state.append(counter)
                    counter += 1
                else:
                    self.state.append(0)
        return self.state

    def reward_spec(self):
        return self._reward_spec

    def action_spec(self):
        return self._action_spec

    def observation_spec(self):
        return self._observation_spec

    def episode_finished(self):
        # print("snake", self.snake_loc, "obstacle", self.obstacle_loc)
        return not Backend.check_if_snake_lives(self.snake_loc, self.obstacle_loc, self.width, self.length)

    # Calls generate_game method from State to initialize a random game from scratch
    def _reset(self):
        self.stan = state.State(50, 10)
        self.snake_loc, self.apple_loc, self.obstacle_loc = state.State.generate_game(self.stan)
        self.state = self._update_state()
        self.episode_ended = False
        self.reset_count += 1
        self.action_count = 0
        self.apple_eaten_count = 0
        return ts.restart(np.array(self.state, dtype=np.int32), reward_spec=self._reward_spec)

    def reset(self):
        self._current_time_step = self._reset()
        #print("Called reset function externally", self._current_time_step, self.episode_ended)
        return self._current_time_step

    def decode_action(self, action):
        if action == 0:
            return self.prev_key
        # Turn left
        if action == 1:
            if self.prev_key == "right": return "up"
            if self.prev_key == "up": return "left"
            if self.prev_key == "left": return "down"
            if self.prev_key == "down": return "right"
        # Turn right
        if action == 2:
            if self.prev_key == "right": return "down"
            if self.prev_key == "up": return "right"
            if self.prev_key == "left": return "up"
            if self.prev_key == "down": return "left"

    def _step(self, action):
        if self.episode_ended:
            return self.reset()

        action_decoded = self.decode_action(action)
        self.prev_key, self.snake_loc, self.apple_eaten, self.apple_loc = state.move_snake(
            self.prev_key, action_decoded, self.snake_loc, self.apple_loc, self.obstacle_loc, self.width, self.length)

        self.state = self._update_state()
        assert self.state.count(2) == 10
        self.episode_ended = self.episode_finished()

        if self.episode_ended or self.action_count > 500:
            self.episode_ended = True
            return ts.termination(np.array(self.state, dtype=np.int32), reward=-10)
        self.action_count += 1
        if self.apple_eaten:
            self.apple_eaten_count += 1
            return ts.transition(np.array(self.state, dtype=np.int32), reward=100, discount=0.3)
        return ts.transition(np.array(self.state, dtype=np.int32), reward=-0.05, discount=0.3)


def compute_avg_return(environment, policy, num_episodes=10):
    total_return = 0.0
    for _ in range(num_episodes):
        time_step = environment.reset()
        episode_return = 0.0
        while not time_step.is_last():
            action_step = policy.action(time_step)
            time_step = environment.step(action_step.action)
            episode_return += time_step.reward
            #print(f"Move reward: {time_step.reward.numpy()[0]}")
        total_return += episode_return

    avg_ret = total_return / num_episodes
    return avg_ret.numpy()[0]

# Data collection


# Add each step of the agent to replay buffer
def collect_step(environment, policy, buffer):
    time_step = environment.current_time_step()
    action_step = policy.action(time_step)
    next_time_step = environment.step(action_step.action)
    traj = trajectory.from_transition(time_step, action_step, next_time_step)

    # Add trajectory to the replay buffer
    buffer.add_batch(traj)


def main():
    # Create train and evaluation environments for Tensorflow
    train_py_env = Environment()
    train_env = tf_py_environment.TFPyEnvironment(train_py_env)

    eval_py_env = Environment()
    eval_env = tf_py_environment.TFPyEnvironment(eval_py_env)

    # utils.validate_py_environment(train_py_env, episodes=5)

    # Set up an agent
    # Decide on layers of a network
    fc_layer_params = (500, 1000, 256, 100, 20, 3)

    # QNetwork predicts QValues (expected returns) for all actions based on observation on the given environment
    q_net = q_network.QNetwork(train_env.observation_spec(), train_env.action_spec(), fc_layer_params=fc_layer_params)

    # Initialize DQN Agent on the train environment steps, actions, QNetwork, Adam Optimizer, loss function & train step counter
    optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=learning_rate)

    # Variable maintains shared, persistent state manipulated by a program.
    # 0 is the initial value.
    # After construction, the type and shape of the variable are fixed.
    train_step_counter = tf.Variable(0)
    agent = dqn_agent.DqnAgent(
        train_env.time_step_spec(),
        train_env.action_spec(),
        q_network=q_net,
        optimizer=optimizer,
        td_errors_loss_fn=common.element_wise_squared_loss,
        train_step_counter=train_step_counter)

    agent.initialize()

    # Policies

    """A policy defines the way an agent acts in an environment. 
    Typically, the goal of RL is to train the underlying model until the policy produces the desired outcome.
    
    Agents contain two policies:
    agent.policy — The main policy that is used for evaluation and deployment.
    agent.collect_policy — A second policy that is used for data collection.
    """

    # tf_agents.policies.random_tf_policy creates a policy which will randomly select an action for each time_step (independent of agent)
    random_policy = random_tf_policy.RandomTFPolicy(train_env.time_step_spec(), train_env.action_spec())

    # Baseline average return of the moves based on random_policy (random actions of an agent)
    print(compute_avg_return(eval_env, random_policy, num_eval_episodes))

    # Replay buffer

    # The replay buffer keeps track of data collected from the environment.
    # This tutorial uses tf_agents.replay_buffers.tf_uniform_replay_buffer.TFUniformReplayBuffer, as it is the most common.
    # The constructor requires the specs for the data it will be collecting.
    # This is available from the agent using the collect_data_spec method.
    # The batch size and maximum buffer length are also required.

    replay_buffer = tf_uniform_replay_buffer.TFUniformReplayBuffer(
        data_spec=agent.collect_data_spec,
        batch_size=batch_size,
        max_length=replay_buffer_max_length)

    # The agent needs access to the replay buffer.
    # This is provided by creating an iterable tf.data.Dataset pipeline which will feed data to the agent.
    # Each row of the replay buffer only stores a single observation step.
    # But since the DQN Agent needs both the current and next observation to compute the loss,
    # the dataset pipeline will sample two adjacent rows for each item in the batch (num_steps=2).
    # This dataset is also optimized by running parallel calls and prefetching data.

    dataset = replay_buffer.as_dataset(
        num_parallel_calls=3,
        sample_batch_size=batch_size,
        single_deterministic_pass=False,
        num_steps=2).prefetch(3)
    iterator = iter(dataset)

    # Train the agent

    # Reset the train step
    agent.train_step_counter.assign(0)

    # Evaluate the agent's policy once before training.
    avg_return = compute_avg_return(eval_env, agent.policy, num_eval_episodes)
    returns = [avg_return]

    for _ in range(num_iterations):

        # Collect a few steps using collect_policy and save to the replay buffer.
        for _ in range(collect_steps_per_iteration):
            collect_step(train_env, agent.collect_policy, replay_buffer)

        # Sample a batch of data from the buffer and update the agent's network.
        experience, unused_info = next(iterator)
        train_loss = agent.train(experience).loss

        step = agent.train_step_counter.numpy()
        if step % log_interval == 0:
            avg_return = compute_avg_return(eval_env, agent.policy, num_eval_episodes)
            print('step = {0}: loss = {1}, Average Return: {2}'.format(step, train_loss, avg_return))


if __name__ == "__main__":
    main()