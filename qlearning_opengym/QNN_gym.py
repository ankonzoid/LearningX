"""

 QNN.py  (author: Anson Wong / git: ankonzoid)

 Neural network Q-learning on a custom gym environment.
 The gym environment provides us with the state space, actions, and rewards(s,a).

"""
import numpy as np
import matplotlib.pyplot as plt

import tensorflow as tf

import gym

# =================================================
#
# Set up environment
#
# =================================================
save_video = False
env_str = 'Taxi-v2'
env = gym.make(env_str)
if save_video:
    env = gym.wrappers.Monitor(env, 'videos/' + env_str, force=True,
                               video_callable=lambda episode_id: episode_id%100==0)

n_states = env.observation_space.n
n_actions = env.action_space.n
print("n_states = {0}".format(n_states))
print("n_actions = {0}".format(n_actions))


# =================================================
#
# Set up Q neural network
#
# =================================================
tf.reset_default_graph()

# Set feed-forward part of the network used to choose actions
input = tf.placeholder(shape=[1, n_states], dtype=tf.float32)  # input layer
W = tf.Variable(tf.random_uniform([n_states, n_actions], 0, 0.01))  # weights
Qout = tf.matmul(input, W)  # set NN for Q value
a_maxQout = tf.argmax(Qout, 1)  # add layer to NN for greedy action from predicted Q

# Obtain loss via sum of squares difference between target and predicted Q values
nextQ = tf.placeholder(shape=[1, n_actions], dtype=tf.float32)  # output layer to feed in target Q-values
loss = tf.reduce_sum(tf.square(nextQ - Qout))  # loss
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.1)  # optimizer
updateModel = optimizer.minimize(loss)  # compile optimizer with loss

# Set initialization for all tensorflow objects above
init = tf.initialize_all_variables()


# =================================================
#
# Generate episodes, and update Q neural network
#
# =================================================
# Set learning parameters
gamma = 0.99  # discount factor
epsilon = 0.1  # epsilon-greedy
n_episodes = 2000

niter_total_history = []
reward_total_history = []
with tf.Session() as sess:
    sess.run(init)

    # Generate episodes
    for i in range(n_episodes):

        # Reset environment and get first new observation
        s = env.reset()
        reward_total = 0
        terminate = False
        iter = 0

        # ===============================================
        #
        # The Q-NN training
        #
        # ===============================================
        max_iter = 999  # maximum number of action iterations per episode
        while iter < max_iter:

            env.render()
            #print(s)
            iter += 1  # iteration within episode

            ###
            ### Select action a based on epsilon-greedy selection from current Q(s,:)
            ###

            # Note s is an integer, to feed into Q-NN -> s_hot one-hot vector
            # Feed s_hot into Qout (NN) -> get Qs = [Q(s)]
            # Feed s_hot into a_maxQout (NN) -> get a = [argmax(Q(s))]
            # Note 1: a[0] is an integer
            # Note 2: Qs[0] is a list of length n_actions
            s_hot = np.identity(n_states)[s:s + 1]  # one-hot vector of current state
            a, Qs = sess.run([a_maxQout, Qout], feed_dict={input:s_hot})
            if np.random.rand(1) < epsilon:  # introduce epsilon-chance of exploring
                a[0] = env.action_space.sample()  # sample action randomly if explore

            ###
            ### Update our Q neural network by training with:
            ###  input: current state (s_hot)
            ###  output labels: current Q Bellman-updated values (targetQ)
            ###

            # Take action -> get new state, and reward from environment
            s_next, reward, terminal, _ = env.step(a[0])

            # Feed in s_next into Q-NN -> get Qs_next_max
            s_next_hot = np.identity(n_states)[s_next:s_next+1]  # one-hot vector for next state
            Qs_next = sess.run(Qout, feed_dict={input:s_next_hot})
            Qs_next_max = np.max(Qs_next)

            # Set targetQ to be current Q(s,a) updated with Bellman equations
            targetQ = Qs
            targetQ[0][a[0]] = reward + gamma*Qs_next_max

            _, W1 = sess.run([updateModel, W], feed_dict={input:s_hot, nextQ:targetQ})

            reward_total += reward  # update total reward
            s = s_next  # update state


            # Break to next episode upon reaching terminal state.
            # Also reduce (epsilon) exploration chance after each episode
            if terminal == True:
                epsilon = 1./((i/50) + 10)
                break

        # Print episode information
        print("Episode {0}/{1}: iter = {2}, reward = {3}".format(
            i+1, n_episodes, iter, reward_total))

        # Append iterations and rewards of episode
        niter_total_history.append(iter)
        reward_total_history.append(reward_total)

# Print
n_splits = 10
n_episodes_per_split = int(n_episodes/n_splits)
niter_total_history_split = []
reward_total_history_split = []
for i in range(n_splits):
    ind_start = i*n_episodes_per_split
    ind_end = (i+1)*n_episodes_per_split

    niter_total_history_split.append(
        sum(niter_total_history[ind_start:ind_end])/
        len(range(ind_start,ind_end)))

    reward_total_history_split.append(
        sum(reward_total_history[ind_start:ind_end]) /
        len(range(ind_start, ind_end)))

print("\n")
print("Average iterations: ", niter_total_history_split)
print("Average rewards: ", reward_total_history_split)
print("Lowest/highest episodic reward: {} / {}".format(min(reward_total_history),
                                                     max(reward_total_history)))

plt.figure(1)
plt.plot(reward_total_history)
plt.xlabel("episode #")
plt.ylabel("total rewards collected")
plt.savefig("plots/" + env_str + "_QNN_totreward.png")

plt.figure(2)
plt.plot(niter_total_history)
plt.xlabel("episode #")
plt.ylabel("number of actions per episode")
plt.savefig("plots/" + env_str + "_QNN_nactions.png")
#plt.show()
