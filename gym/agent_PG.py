"""

 agent_PG.py (author: Anson Wong / git: ankonzoid)

 Policy gradients NN approach for solve the opengym cart pole environment.

"""
import numpy as np
import random
import tensorflow as tf
import matplotlib.pyplot as plt
import gym

def main():

    # ======================================
    #
    # Configure settings
    #
    # ======================================
    save_plot_reward_results = True  # rewards collected vs episode #

    # ======================================
    #
    #  Set up agent environment
    #
    # ======================================
    if 1:
        env_str = 'CartPole-v0'
        n_train_episodes = 1000  # training episodes
        reward_success = 200  # minimum reward to be considered success
    elif 0:
        env_str = 'MountainCar-v0'
        n_train_episodes = 1000  # training episodes
        reward_success = -110  # minimum reward to be considered success
    elif 0:
        env_str = 'LunarLander-v2'
        n_train_episodes = 1000  # training episodes
        reward_success = 200  # minimum reward to be considered success
    elif 0:
        env_str = 'AirRaid-ram-v0'
        n_train_episodes = 200  # training episodes
        reward_success = 2000  # minimum reward to be considered success
    else:
        print("Invalid environment given!")
        exit()

    save_folder = env_str
    env = gym.make(env_str)
    env = gym.wrappers.Monitor(env, save_folder, force=True)

    if 1:

        if 'n' in env.action_space.__dict__:
            print("env.action_space is Discrete. Go.")
            print(env.action_space)
        elif 'low' in env.action_space.__dict__ and 'high' in env.action_space.__dict__:
            print("env.action_space is Box. Stop.")
            exit()
        else:
            raise IOError("Invalid action space")

        if 'n' in env.observation_space.__dict__:
            print("env.observation_space is Discrete. Stop.")
            exit()
        elif 'low' in env.observation_space.__dict__ and 'high' in env.observation_space.__dict__:
            print("env.observation_space is Box. Go.")
            print(env.observation_space)
        else:
            raise IOError("Invalid observation space")

    policy_grad = policy_gradient(env)
    value_grad = value_gradient(env)
    sess = tf.InteractiveSession()
    sess.run(tf.global_variables_initializer())



    # ======================================
    #
    # Train agent
    #
    # ======================================
    reward_results = []
    print("Training the agent for {} episodes".format(n_train_episodes))
    for i in range(n_train_episodes):

        # Run episode and update policy gradient and value gradient
        reward_collected = run_episode(env, policy_grad, value_grad, sess,
                                       render = False, force_greedy = False)
        reward_results.append(reward_collected)

        # Report result of episode
        if reward_collected >= reward_success:
            print("[Train episode {}/{}] Success. Rewards collected = {}".format(i+1,n_train_episodes,reward_collected))
        else:
            print("[Train episode {}/{}] Fail. Rewards collected = {}".format(i+1,n_train_episodes,reward_collected))

    # Report lowest/highest results collected during training
    print("\nLowest rewards collected over {} episodes: {}".format(
        n_train_episodes, min(reward_results)))
    print("Highest rewards collected over {} episodes: {}\n".format(
        n_train_episodes, max(reward_results)))


    # ================================
    #
    # Show results
    #
    # =================================

    # Plot
    print("Saving collected rewards vs episode number plot...")
    if save_plot_reward_results:
        plt.plot(reward_results)
        plt.xlabel('Episode #')
        plt.ylabel('Reward collected')
        plt.title('Rewards over time')
        plt.savefig(save_folder + '/' + env_str + '_results.png', bbox_inches='tight')

    # Show some test example episodes
    n_test_episodes = 5
    print("Showing {} test examples with trained agent...".format(n_test_episodes))
    for i in range(n_test_episodes):

        reward_collected = run_episode(env, policy_grad, value_grad, sess,
                                       render = True, force_greedy = True)

        if reward_collected >= reward_success:
            print("[Test episode {}/{}] Success. Rewards collected = {}".format(i+1,n_test_episodes,reward_collected))
        else:
            print("[Test episode {}/{}] Fail. Rewards collected = {}".format(i+1,n_test_episodes,reward_collected))


# ==========================================
#
# Side functions
#
# ==========================================

# softmax function = exp(xi - max(x))/sum(exp(xi-max(x))
def softmax(x):
    e_x = np.exp(x - np.max(x))  # array of values between (0,1]
    out = e_x / e_x.sum()  # normalize to make a probability
    return out

# policy gradient network
def policy_gradient(env):

    # Retrieve observation and action space dimension
    n_observation_space = env.observation_space.shape[0]
    n_action_space = env.action_space.n

    with tf.variable_scope("policy"):

        # Set state
        state = tf.placeholder("float", [None, n_observation_space])

        # Set actions
        actions = tf.placeholder("float", [None, n_action_space])

        # Set advantages
        params = tf.get_variable("policy_parameters",
                                 [n_observation_space, n_action_space])
        advantages = tf.placeholder("float", [None, 1])
        linear = tf.matmul(state, params)  # Combine

        # Set probabilities
        probabilities = tf.nn.softmax(linear)
        good_probabilities = tf.reduce_sum(tf.multiply(probabilities, actions), reduction_indices=[1])
        eligibility = tf.log(good_probabilities) * advantages
        loss = -tf.reduce_sum(eligibility)

        # Set optimizer
        learning_rate = 0.1
        optimizer = tf.train.AdamOptimizer(learning_rate).minimize(loss)

        return probabilities, state, actions, advantages, optimizer

# value gradient
def value_gradient(env):

    # Retrieve observation and action space dimension
    n_observation_space = env.observation_space.shape[0]
    n_action_space = env.action_space.n

    with tf.variable_scope("value"):

        # Set state
        state = tf.placeholder("float", [None, n_observation_space])

        # Set new values
        newvals = tf.placeholder("float", [None, 1])

        # Set calculated
        w1 = tf.get_variable("w1", [n_observation_space, n_action_space])  # Value gradient is *w1+b1, Relu, *w2+b2. 4, 2, 1.
        b1 = tf.get_variable("b1", [n_action_space])
        h1 = tf.nn.relu(tf.matmul(state, w1) + b1)
        w2 = tf.get_variable("w2", [n_action_space, 1])
        b2 = tf.get_variable("b2", [1])
        calculated = tf.matmul(h1, w2) + b2
        diffs = calculated - newvals

        # Set loss
        loss = tf.nn.l2_loss(diffs)

        # Set optimizer
        learning_rate = 0.1
        optimizer = tf.train.AdamOptimizer(learning_rate).minimize(loss)

        return calculated, state, newvals, optimizer, loss

# run episode
def run_episode(env, policy_grad, value_grad, sess, render=False, force_greedy=False):

    # Retrieve observation and action space dimension
    n_observation_space = env.observation_space.shape[0]
    n_action_space = env.action_space.n

    # Retrieve current values from policy grad and value grad
    pl_calculated, pl_state, pl_actions, pl_advantages, pl_optimizer = policy_grad
    vl_calculated, vl_state, vl_newvals, vl_optimizer, vl_loss = value_grad

    # Initialize variables
    observation = env.reset()
    totalreward = 0
    states = []
    actions = []
    advantages = []
    transitions = []
    update_vals = []

    #
    # Run episode by following policy
    #
    n_steps_per_episode = 1000000
    for t in range(n_steps_per_episode):

        # Render
        if render:
            env.render()

        # Calculate policy
        obs_vector = np.expand_dims(observation, axis=0)
        probs = sess.run(pl_calculated, feed_dict={pl_state: obs_vector})  # probability for available actions

        # Perform action
        action = np.argmax(probs)
        epsilon = 0.8  # exploration probability
        if not force_greedy:
            if random.uniform(0, 1) > epsilon:
                action = random.randint(0, n_action_space-1)

        # Record the transition
        states.append(observation)
        actionblank = np.zeros(n_action_space)
        actionblank[action] = 1
        actions.append(actionblank)

        # Take the action in the environment
        old_observation = observation
        observation, reward, done, info = env.step(action)
        transitions.append((old_observation, action, reward))
        totalreward += reward

        # Done?
        if done:
            break

    #
    # After finishing the episode, go through history to make update
    #
    for index, trans in enumerate(transitions):
        obs, action, reward = trans

        # Calculate discounted Monte Carlo return
        future_reward = 0
        future_transitions = len(transitions) - index
        decrease = 1
        for index2 in range(future_transitions):
            future_reward += transitions[(index2) + index][2] * decrease
            decrease = decrease * 0.99
        obs_vector = np.expand_dims(obs, axis=0)
        currentval = sess.run(vl_calculated, feed_dict={vl_state: obs_vector})[0][0]

        # Advantage: how much better was this action than normal?
        advantages.append(future_reward - currentval)

        # Update the value function towards new return
        update_vals.append(future_reward)

    #
    # Update value function
    #
    update_vals_vector = np.expand_dims(update_vals, axis=1)
    sess.run(vl_optimizer, feed_dict={vl_state: states, vl_newvals: update_vals_vector})

    #
    # Update policy function
    #
    advantages_vector = np.expand_dims(advantages, axis=1)
    sess.run(pl_optimizer, feed_dict={pl_state: states, pl_advantages: advantages_vector, pl_actions: actions})

    #
    # Return the rewards collected from this episode
    #
    return totalreward

#
# Driver
#
if __name__ == "__main__":
    main()
