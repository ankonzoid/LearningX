"""

 multiarmed_bandit.py (author: Anson Wong / git: ankonzoid)

 Attempts to solve the multi-armed bandit problem by finding the best bandit
 and maximizing the total reward over fixed number of episodes received by
 the agent. We use the optimization method of policy gradients on a feed-forward
 neural network approximation of the policy.

 Note that there are no explicit state value functions, we are using a policy-based
 optimization method to find the best policy.

"""
import numpy as np
import random
from numpy import linalg as LA
import tensorflow as tf

def main():

    # ================================================
    #
    # Set up agent behaviour and rigged bandits environment
    #
    # ================================================
    total_episodes = 40000  # total number of trials
    epsilon = 0.6  # exploratory probability
    bandits = [0.6, 0.1, 0.9, 0.3, 0.7]  # bandit probability of success


    # ================================================
    #
    # Set up agent and policy objective function J(p)
    #
    # ================================================
    n_bandits = len(bandits)
    tf.reset_default_graph()

    # Create feed-forward NN to prepare for our policy objective function definition
    # W: stores output weights for each bandit
    # w_action: the weight value for a particular action
    W = tf.Variable(tf.ones([n_bandits]))  # initialize n_bandits weights of 1
    argmax_W = tf.argmax(W, 0)  # gives bandit with highest weight
    action_holder = tf.placeholder(shape=[1], dtype=tf.int32)
    w_action = tf.slice(W, action_holder, [1])  # action weight value

    # Define policy objective function and compile model
    reward_holder = tf.placeholder(shape=[1], dtype=tf.float32)
    loss = -(w_action*reward_holder)  # set policy objective function
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.0001)  # optimizer
    update = optimizer.minimize(loss)  # minimize policy objective function


    # ================================================
    #
    # Train agent
    #
    # ================================================
    np.set_printoptions(precision=2)  # set printing precision
    total_reward = np.zeros(n_bandits)  # scoreboard for bandits to 0
    init = tf.initialize_all_variables()  # initialize variables
    with tf.Session() as sess:

        sess.run(init)
        episode = 1
        n_explore = 0
        n_greedy = 0
        while episode < total_episodes:

            ### Choose an action
            if random.uniform(0,1) < epsilon:
                action = np.random.randint(n_bandits)  # explore
                action_str = "explore"
                n_explore += 1
            else:
                action = sess.run(argmax_W)  # greedy
                action_str = "greedy"
                n_greedy += 1

            ### Receive reward for action
            reward = get_reward(bandits[action])

            ### Update policy (NN approximation) using policy gradients
            ### Feed in reward and action -> output of responsible weight, and W
            _, w_action_output, W_output = sess.run([update, w_action, W],
                                   feed_dict={reward_holder: [reward],
                                              action_holder: [action]})


            # Update our running tally of scores.
            total_reward[action] += reward

            if episode % (total_episodes/10) == 0:
                print("[Episode {0} ({1}%)] pick bandit {2} ({3}) -> reward = {4}".format(
                    episode, 100*episode/total_episodes, action + 1, action_str, reward))
                print("  W_output = {0}".format(W_output))
                print("  n_explore = {0}, n_greedy = {1}".format(n_explore, n_greedy))

                # Predict the best bandit as of now
                best_bandit_groundtruth = np.argmax(np.array(bandits)) + 1  # from ground truth
                best_bandit_prediction = np.argmax(W_output) + 1  # from agent training

                print("  Agent redicts bandit {1}".format(episode, best_bandit_prediction),
                    "to be best (ground truth = {0})".format(best_bandit_groundtruth))
                print("")

            episode += 1


"""
 get reward from bandit
"""
def get_reward(bandit):
    r = random.uniform(0,1)  # sample from uniform distrib
    if r < bandit:
        return 1  # success
    else:
        return -1  # failure

# Driver file
if __name__ == '__main__':
    main()