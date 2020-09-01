import tensorflow as tf
import numpy as np
from tf_agents.agents.dqn import dqn_agent
from tf_agents.networks import q_network
from tf_agents.environments import py_environment
from tf_agents.environments import tf_py_environment
from tf_agents.specs import array_spec
from tf_agents.trajectories import time_step as ts
from tf_agents.utils import common
from tf_agents.policies import random_tf_policy
from tf_agents.replay_buffers import tf_uniform_replay_buffer
from tf_agents.trajectories import trajectory
import Backend
import state

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
        self.stan = state.State(50, 10)
        # Action is a snake move: 0: forward, 1: left, 2: right
        self._action_spec = array_spec.BoundedArraySpec(
            shape=(), dtype=np.int32, minimum=0, maximum=2, name='action')
        self.length = self.stan.length
        self.width = self.stan.width
        self.snake_loc_set = set(self.stan.snake_loc)
        self.obstacle_loc_set = set(self.stan.obstacle_loc)
        self.apple_loc_set = set(self.stan.apple_loc)
        self.snake_loc = self.stan.snake_loc
        self.obstacle_loc = self.stan.obstacle_loc
        self.apple_loc = self.stan.apple_loc
        self.prev_key = self.stan.prev_key

        # Observation is the state of the environment.
        # The observation specification has specifications of observations provided by the environment.
        # As the board has width * length positions, the shape of an observation is (1, width * length).
        self._observation_spec = array_spec.BoundedArraySpec(
            shape=(1, self.length * self.width), dtype=np.int32, minimum=0, maximum=100, name='observation')

        # Generated in a separate method _update_state
        self.state = self._update_state()
        self.episode_ended = False
        self.apple_eaten = self.stan.ate_apple

    # Coding of objects on the board
    # Empty field: 0, Obstacle: 1, Apple: 2, Snake: 3, 4 ...,
    def _update_state(self):
        self.state = []
        counter = 3
        for row in range(self.length):
            for col in range(self.width):
                if (row, col) in self.apple_loc_set:
                    self.state.append(2)
                elif (row, col) in self.obstacle_loc_set:
                    self.state.append(1)
                elif (row, col) in self.snake_loc_set:
                    self.state.append(counter)
                    counter += 1
                else:
                    self.state.append(0)
        return self.state

    def action_spec(self):
        return self._action_spec

    def observation_spec(self):
        return self._observation_spec

    def episode_finished(self):
        return Backend.check_if_snake_lives(self.snake_loc, self.obstacle_loc, self.width, self.length)

    # Calls generate_game method from State to initialize a random game from scratch
    def _reset(self):
        state.State.generate_game(self.stan)
        self.stan = state.State(self.width, self.length)
        self.state = self._update_state()
        self._episode_ended = False
        return ts.restart(np.array([self.state], dtype=np.int32))

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
        if self._episode_ended:
            return self.reset()

        action_decoded = self.decode_action(action)
        state.State.move_snake(self.stan, action_decoded)
        self.state = self._update_state()
        # print("Episode ended:", self.episode_ended)
        self.episode_ended = self.episode_finished()

        if self.apple_eaten:
            return ts.transition(np.array([self.state], dtype=np.int32), reward=10)
        return ts.transition(np.array([self.state], dtype=np.int32), reward=-0.0005, discount=1.0)


# Create train and evaluation environments for Tensorflow
train_py_env = Environment()
train_env = tf_py_environment.TFPyEnvironment(train_py_env)

eval_py_env = Environment()
eval_env = tf_py_environment.TFPyEnvironment(eval_py_env)

# Set up an agent
fc_layer_params = (100, 20, 4)
q_net = q_network.QNetwork(train_env.observation_spec(), train_env.action_spec(), fc_layer_params=fc_layer_params)

optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=learning_rate)

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

eval_policy = agent.policy
collect_policy = agent.collect_policy
random_policy = random_tf_policy.RandomTFPolicy(train_env.time_step_spec(),
                                                train_env.action_spec())


def compute_avg_return(environment, policy, num_episodes=10):
    total_return = 0.0
    for _ in range(num_episodes):
        time_step = environment.reset()
        episode_return = 0.0

        while not time_step.is_last():
            action_step = policy.action(time_step)
            time_step = environment.step(action_step.action)
            episode_return += time_step.reward
            total_return += episode_return

    avg_return = total_return / num_episodes
    print(avg_return)
    return avg_return


compute_avg_return(eval_env, random_policy, num_eval_episodes)

# Replay buffer

replay_buffer = tf_uniform_replay_buffer.TFUniformReplayBuffer(
    data_spec=agent.collect_data_spec,
    batch_size=train_env.batch_size,
    max_length=replay_buffer_max_length)

agent.collect_data_spec
agent.collect_data_spec._fields

# Data collection

def collect_step(environment, policy, buffer):
    time_step = environment.current_time_step()
    action_step = policy.action(time_step)
    next_time_step = environment.step(action_step.action)
    traj = trajectory.from_transition(time_step, action_step, next_time_step)

    # Add trajectory to the replay buffer
    buffer.add_batch(traj)


def collect_data(env, policy, buffer, steps):
    for _ in range(steps):
        collect_step(env, policy, buffer)


collect_data(train_env, random_policy, replay_buffer, steps=100)

dataset = replay_buffer.as_dataset(
    num_parallel_calls=3,
    sample_batch_size=batch_size,
    num_steps=2).prefetch(3)

iterator = iter(dataset)

# (Optional) Optimize by wrapping some of the code in a graph using TF function.
agent.train = common.function(agent.train)

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
        print('step = {0}: loss = {1}'.format(step, train_loss))

    if step % eval_interval == 0:
        avg_return = compute_avg_return(eval_env, agent.policy, num_eval_episodes)
        print('step = {0}: Average Return = {1}'.format(step, avg_return))
        returns.append(avg_return)

print("dupa")
