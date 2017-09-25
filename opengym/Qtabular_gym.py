"""

 Qtabular_gym.py  (author: Anson Wong / git: ankonzoid)

 Tabular Q-learning on a custom gym environment.
 The gym environment provides us with the state space, actions, and rewards(s,a).

"""
import numpy as np
import matplotlib.pyplot as plt

import gym

# ====================================================
#
# Set up environment
#
# ====================================================
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


# ====================================================
#
# Initialize Q-table
#
# ====================================================
# Initialize Q-table with all zeros
Q = np.zeros([n_states, n_actions])


# ====================================================
#
# Generate episodes and update Q-table
#
# ====================================================
# Set learning parameters
beta = 0.8  # learning rate
gamma = 0.95  # discount factor
n_episodes = 2000

niter_total_history = []
reward_total_history = []
for i in range(n_episodes):

    # Reset environment and get initial state
    s = env.reset()
    reward_total = 0
    terminate = False
    iter = 0

    # The Q-Table learning algorithm
    max_iter = 999  # maximum number of action iterations per episode
    while iter < max_iter:

        env.render()
        #print(s)
        iter += 1  # iteration within episode

        ###
        ### Choose an action by greedily (with exploration noise) picking from Q table
        ###
        exploration_noise = (1./(i+1))
        a = np.argmax(Q[s,:] + np.random.randn(1,n_actions)*exploration_noise)


        ###
        ### Update Q-Table
        ###

        # Take action a -> get new state s1, reward r,
        s_next, reward, terminal, _ = env.step(a)

        # Update Q(s,a) using Bellman equations
        Qnext_max = np.max(Q[s_next,:])
        Q[s,a] = Q[s,a] + beta*(reward + gamma*Qnext_max - Q[s,a])

        reward_total += reward  # update total reward
        s = s_next  # update state


        # Break to next episode upon reach terminal state
        if terminal == True:
            break

    # Print episode information
    print("Episode {0}/{1}: iter = {2}, reward = {3}".format(
        i+1, n_episodes, iter, reward_total))

    # Remember the episodic reward history
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


print("Final Q table")
print(Q)


print("\n")
print("Average iterations: ", niter_total_history_split)
print("Average rewards: ", reward_total_history_split)
print("Lowest/highest episodic reward: {} / {}".format(
    min(reward_total_history), max(reward_total_history)))

plt.figure(1)
plt.plot(reward_total_history)
plt.xlabel("episode #")
plt.ylabel("total rewards collected")
plt.savefig("plots/" + env_str + "_Qtab_totreward.png")

plt.figure(2)
plt.plot(niter_total_history)
plt.xlabel("episode #")
plt.ylabel("number of actions per episode")
plt.savefig("plots/" + env_str + "_Qtab_nactions.png")
#plt.show()


