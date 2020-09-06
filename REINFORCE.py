import tensorflow as tf
from tf_agents.agents.reinforce import reinforce_agent
from tf_agents.networks import actor_distribution_network
from tf_agents.environments import tf_py_environment
from tf_agents.replay_buffers import tf_uniform_replay_buffer
from tf_agents.trajectories import trajectory
import Environment
import Screen

# Define model hyperparameters
num_iterations = 20000 # @param {type:"integer"}
collect_episodes_per_iteration = 2 # @param {type:"integer"}
replay_buffer_capacity = 100000 # @param {type:"integer"}

fc_layer_params = (500, 1000, 256, 100, 20, 3)

learning_rate = 1e-3 # @param {type:"number"}
log_interval = 50 # @param {type:"integer"}
num_eval_episodes = 10 # @param {type:"integer"}
eval_interval = 50 # @param {type:"integer"}

# Evaluation of a model performance
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
    return avg_return.numpy()[0]

def collect_episode(environment, policy, num_episodes, buffer):

    episode_counter = 0
    environment.reset()

    while episode_counter < num_episodes:
        time_step = environment.current_time_step()
        action_step = policy.action(time_step)
        next_time_step = environment.step(action_step.action)
        traj = trajectory.from_transition(time_step, action_step, next_time_step)

        # Add trajectory to the replay buffer
        buffer.add_batch(traj)

        if traj.is_boundary():
            episode_counter += 1

def main():
    # Create train and evaluation environments for Tensorflow
    train_py_env = Environment.Environment()
    train_env = tf_py_environment.TFPyEnvironment(train_py_env)

    eval_py_env = Environment.Environment()
    eval_env = tf_py_environment.TFPyEnvironment(eval_py_env)

    # utils.validate_py_environment(train_py_env, episodes=5)

    # Set up an agent
    # The algorithm that we use to solve an RL problem is represented as an Agent.
    # To create a REINFORCE Agent, we first need an Actor Network
    # that can learn to predict the action given an observation from the environment.
    #
    # We can easily create an Actor Network using the specs of the observations and actions.
    # We can specify the layers in the network which,
    # in this example, is the fc_layer_params argument set to a tuple of ints representing
    # the sizes of each hidden layer (see the Hyperparameters section above).

    actor_net = actor_distribution_network.ActorDistributionNetwork(
        train_env.observation_spec(),
        train_env.action_spec(),
        fc_layer_params=fc_layer_params)

    optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=learning_rate)

    train_step_counter = tf.compat.v2.Variable(0)

    tf_agent = reinforce_agent.ReinforceAgent(
        train_env.time_step_spec(),
        train_env.action_spec(),
        actor_network=actor_net,
        optimizer=optimizer,
        normalize_returns=True,
        train_step_counter=train_step_counter)
    tf_agent.initialize()

    # Replay buffer

    replay_buffer = tf_uniform_replay_buffer.TFUniformReplayBuffer(
        data_spec=tf_agent.collect_data_spec,
        batch_size=1,
        max_length=replay_buffer_capacity)

    # Reset the train step
    tf_agent.train_step_counter.assign(0)

    # Evaluate the agent's policy once before training.
    avg_return = compute_avg_return(eval_env, tf_agent.policy, num_eval_episodes)
    returns = [avg_return]

    for _ in range(num_iterations):

        # Collect a few episodes using collect_policy and save to the replay buffer.
        collect_episode(
            train_env, tf_agent.collect_policy, collect_episodes_per_iteration, replay_buffer)

        # Use data from the buffer and update the agent's network.
        experience = replay_buffer.gather_all()
        train_loss = tf_agent.train(experience)
        replay_buffer.clear()

        step = tf_agent.train_step_counter.numpy()

        if step % log_interval == 0:
            print('step = {0}: loss = {1}'.format(step, train_loss.loss))

        if step % eval_interval == 0:
            avg_return = compute_avg_return(eval_env, tf_agent.policy, num_eval_episodes)
            print('step = {0}: Average Return = {1}'.format(step, avg_return))
            returns.append(avg_return)

if __name__ == "__main__":
    main()