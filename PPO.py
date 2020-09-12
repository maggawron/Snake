#from multiprocessing import set_start_method
#set_start_method("spawn")

import tf_agents
import tensorflow as tf
from tf_agents.agents.ppo import ppo_agent
from tf_agents.drivers import dynamic_episode_driver
from tf_agents.environments import parallel_py_environment
from tf_agents.environments import tf_py_environment

from tf_agents.metrics import tf_metrics
from tf_agents.networks.actor_distribution_rnn_network import ActorDistributionRnnNetwork
from tf_agents.networks.value_rnn_network import ValueRnnNetwork
from tf_agents.replay_buffers import tf_uniform_replay_buffer

from tensorflow.keras.optimizers import Adam

from tf_agents.trajectories.policy_step import PolicyStep

import os

import Environment

# Define model hyperparameters
num_environment_steps = 30000000
collect_episodes_per_iteration = 32
num_parallel_environments = 32
replay_buffer_capacity = 501  # Per-environment
# Params for train
num_epochs = 25
learning_rate = 4e-4
# Params for eval
num_eval_episodes = 3
eval_interval = 100


def evaluate_perf(env, policy, num_episodes):
    for episode in range(num_episodes):
        time_step = env.reset()
        state = policy.get_initial_state(env.batch_size)

        total_reward = 0
        while not time_step.is_last():
            policy_step: PolicyStep = policy.action(time_step, state)
            state = policy_step.state
            total_reward += time_step.reward
            time_step = env.step(policy_step.action)

    avg_reward = total_reward / num_episodes
    return avg_reward[0]

def main():
    tf.compat.v1.enable_v2_behavior()
    # Create train and evaluation environments for Tensorflow
    train_py_env = Environment.Environment()
    train_env = tf_py_environment.TFPyEnvironment(parallel_py_environment.ParallelPyEnvironment([Environment.Environment] * num_parallel_environments))
    #train_env = tf_py_environment.TFPyEnvironment(train_py_env)

    eval_py_env = Environment.Environment()
    eval_env = tf_py_environment.TFPyEnvironment(eval_py_env)

    optimizer = Adam(learning_rate=learning_rate, epsilon=1e-5)
    global_step = tf.compat.v1.train.get_or_create_global_step()

    # Initialize actor and value networks
    actor_net = ActorDistributionRnnNetwork(
        input_tensor_spec=train_env.observation_spec(),
        output_tensor_spec=train_env.action_spec(),
        conv_layer_params=[(3, 4, 1), (10, 8, 2), (5, 8, 2)],
        input_fc_layer_params=(256,),
        lstm_size=(256,),
        output_fc_layer_params=(128,),
        activation_fn=tf.nn.elu)

    value_net = ValueRnnNetwork(
        input_tensor_spec=train_env.observation_spec(),
        conv_layer_params=[(3, 4, 1), (10, 8, 2), (5, 8, 2)],
        input_fc_layer_params=(256,),
        lstm_size=(256,),
        output_fc_layer_params=(128,),
        activation_fn=tf.nn.elu)

    agent = ppo_agent.PPOAgent(
        train_env.time_step_spec(),
        train_env.action_spec(),
        optimizer,
        actor_net,
        value_net,
        num_epochs=num_epochs,
        train_step_counter=global_step,
        discount_factor=0.99,
        gradient_clipping=0.5,
        entropy_regularization=1e-2,
        importance_ratio_clipping=0.2,
        use_gae=True,
        use_td_lambda_return=True
    )

    agent.initialize()
    environment_steps_metric = tf_metrics.EnvironmentSteps()
    step_metrics = [tf_metrics.NumberOfEpisodes(), environment_steps_metric]

    replay_buffer = tf_uniform_replay_buffer.TFUniformReplayBuffer(agent.collect_data_spec,
                                                                   batch_size=num_parallel_environments,
                                                                   max_length=replay_buffer_capacity)

    collect_driver = dynamic_episode_driver.DynamicEpisodeDriver(train_env, agent.collect_policy,
                                                                 observers=[replay_buffer.add_batch] + step_metrics,
                                                                 num_episodes=collect_episodes_per_iteration)

    environment_steps_metric = tf_metrics.EnvironmentSteps()

    # Reset the train step
    agent.train_step_counter.assign(0)

    while environment_steps_metric.result() < num_environment_steps:
        collect_driver.run()

        trajectories = replay_buffer.gather_all()
        total_loss, unused_info = agent.train(experience=trajectories)

        replay_buffer.clear()

        global_step_val = global_step.numpy()

        if global_step_val % eval_interval == 0:

            avg_return = evaluate_perf(eval_env, agent.policy, num_eval_episodes)

            if not os.path.exists("eval_data"):
                os.makedirs("eval_data")
            path = os.path.join("eval_data", f'Eval_data.step{global_step_val // eval_interval}.txt')
            with open(path, 'w') as f:
                for move in eval_py_env.all_moves:
                    print(str(move), file=f)
            eval_py_env.all_moves = []
            print('step = {0}: loss = {1}, Avg return: {2}'.format(global_step_val, total_loss, avg_return))


if __name__ == "__main__":
    tf_agents.system.multiprocessing.handle_main(main)
