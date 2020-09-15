import time
import tf_agents
import tensorflow as tf
import sys
from tf_agents.agents.ppo import ppo_agent
from tf_agents.drivers import dynamic_episode_driver
from tf_agents.environments import parallel_py_environment
from tf_agents.environments import tf_py_environment

from tf_agents.metrics import tf_metrics
from tf_agents.networks.actor_distribution_rnn_network import ActorDistributionRnnNetwork
from tf_agents.networks.value_rnn_network import ValueRnnNetwork
from tf_agents.replay_buffers import tf_uniform_replay_buffer
from tf_agents.trajectories.policy_step import PolicyStep
from tf_agents.policies import policy_saver

from tensorflow.keras.optimizers import Adam
from tf_agents.utils import common


import os
import Environment

#TODO: Add PolicySaver & save configuration & hyperparameters, deque

# Define model hyperparameters
num_environment_steps = 3 * 10 ** 7
collect_episodes_per_iteration = 32
num_parallel_environments = 1
replay_buffer_capacity = 501  # Per-environment
# Params for train
num_epochs = 20
learning_rate = 4e-4
# Params for eval
num_eval_episodes = 3
eval_interval = 1
policy_saver_interval = 2 # 1000


def eval_path(global_step_val, eval_interval):
    if not os.path.exists("eval_data"):
        os.makedirs("eval_data")
    return os.path.join("eval_data", f'Eval_data.step{global_step_val // eval_interval}.txt')


def evaluate_perf(f, env, policy, num_episodes):
    total_reward = 0
    for episode in range(num_episodes):
        time_step = env.reset()
        state = policy.get_initial_state(env.batch_size)
        while not time_step.is_last():
            screen = time_step.observation
            policy_step = policy.action(time_step, state)
            state = policy_step.state
            total_reward += time_step.reward[0]
            time_step = env.step(policy_step.action)
            print((screen.numpy().tolist(),
                   time_step.reward[0].numpy().tolist(),
                   total_reward.numpy().tolist()),
                  file=f)
    avg_reward = total_reward / num_episodes
    return avg_reward


def main():
    if len(sys.argv) != 2:
        raise ValueError(f"Usage: ./{sys.argv[0]} experiment_name")
    experiment_name = sys.argv[1]

    tf.compat.v1.enable_v2_behavior()
    # Create train and evaluation environments for Tensorflow
    train_env = tf_py_environment.TFPyEnvironment(parallel_py_environment.ParallelPyEnvironment([Environment.Environment] * num_parallel_environments))

    eval_py_env = Environment.Environment()
    eval_env = tf_py_environment.TFPyEnvironment(eval_py_env)

    optimizer = Adam(learning_rate=learning_rate, epsilon=1e-5)

    global_step = tf.compat.v1.train.get_or_create_global_step()
    timed_at_step = global_step.numpy()

    # Initialize actor and value networks
    actor_net = ActorDistributionRnnNetwork(
        input_tensor_spec=train_env.observation_spec(),
        output_tensor_spec=train_env.action_spec(),
        conv_layer_params=[(3, 4, 1), (7, 4, 2), (5, 8, 2)],
        input_fc_layer_params=(128,),
        lstm_size=(128,),
        output_fc_layer_params=(64,),
        activation_fn=tf.nn.elu)

    value_net = ValueRnnNetwork(
        input_tensor_spec=train_env.observation_spec(),
        conv_layer_params=[(3, 4, 1), (7, 4, 2), (5, 8, 2)],
        input_fc_layer_params=(128,),
        lstm_size=(128,),
        output_fc_layer_params=(64,),
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
    step_metrics = [tf_metrics.NumberOfEpisodes(), tf_metrics.EnvironmentSteps()]

    replay_buffer = tf_uniform_replay_buffer.TFUniformReplayBuffer(agent.collect_data_spec,
                                                                   batch_size=num_parallel_environments,
                                                                   max_length=replay_buffer_capacity)

    collect_driver = dynamic_episode_driver.DynamicEpisodeDriver(train_env, agent.collect_policy,
                                                                 observers=[replay_buffer.add_batch] + step_metrics,
                                                                 num_episodes=collect_episodes_per_iteration)

    environment_steps_metric = tf_metrics.EnvironmentSteps()

    collect_time = 0
    train_time = 0

    # Reset the train step
    agent.train_step_counter.assign(0)

    saved_model_dir = os.path.join("saved_models", experiment_name)
    checkpoint_dir = os.path.join(saved_model_dir, 'checkpoint')
    train_checkpointer = common.Checkpointer(
        ckpt_dir=checkpoint_dir,
        max_to_keep=1,
        agent=agent,
        policy=agent.policy,
        #replay_buffer=replay_buffer,
        global_step=global_step
    )

    train_checkpointer.initialize_or_restore()
    global_step = tf.compat.v1.train.get_global_step()

    while environment_steps_metric.result() < num_environment_steps:
        start_time = time.time()
        collect_driver.run()
        collect_time += time.time() - start_time

        start_time = time.time()
        trajectories = replay_buffer.gather_all()
        total_loss, unused_info = agent.train(experience=trajectories)
        replay_buffer.clear()
        train_time += time.time() - start_time

        global_step_val = global_step.numpy()
        if global_step_val % eval_interval == 0:
            with open(eval_path(global_step_val, eval_interval), 'w') as f:
                avg_return = evaluate_perf(f, eval_env, agent.policy, eval_interval)
            steps_per_sec = ((global_step_val - timed_at_step) / (collect_time + train_time))
            print(f"step = {global_step_val}: loss = {total_loss}, Avg return: {avg_return}, {steps_per_sec:.3f} steps/sec, collect_time = {collect_time}, train_time = {train_time}")
            timed_at_step = global_step_val
            collect_time = 0
            train_time = 0

        if global_step_val % policy_saver_interval == 0:
            train_checkpointer.save(global_step_val)


if __name__ == "__main__":
    tf_agents.system.multiprocessing.handle_main(main)
