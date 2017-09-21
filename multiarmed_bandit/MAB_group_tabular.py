"""

 MAB_group.py  (author: Anson Wong / git: ankonzoid)

 Classical epsilon-greedy agent solving the multi-armed bandit problem
 with the additional option of choosing a group of bandits (rather than
 the conventional single bandit choice) to maximize the total rewards for.

 Given a set of bandits with a probability distribution of success, we
 maximize our collection of rewards with an agent that explores with
 epsilon probability, and exploits the action of highest value estimate
 for the remaining probability. Experiments are executed many times
 and averaged out and plotted as an averaged reward history.

 The update rule for the values is via an incremental implementation of:

   V(a;k+1) = V(a;k) + alpha*(R(a) - V(a;k))

 where
   k = # of times action "a" (essentially bandit here) was chosen in the past
   V(a;k) = value of action "a"
   R(a) = reward for choosing action (bandit) "a"
   alpha = 1/k

 Note that the reward R(a) is stochastic in this example and follows the probability
 of the distributions provided by the user in the variable "bandits".

"""
import numpy as np
import matplotlib.pyplot as plt

def main():
    # =========================
    # Settings
    # =========================
    bandits = [0.10, 0.50, 0.60, 0.80, 0.10, 0.25, 0.60, 0.45, 0.75, 0.65]  # bandit probabilities of success
    N_choices = 4  # number of bandits to pull simultaneously (without replacement)

    N_experiments = 1000  # number of experiments to perform
    N_episodes = 2000  # number of episodes per experiment
    epsilon = 0.1  # probability of random exploration (fraction)

    save_fig = True  # if false -> plot, if true save as file in same directory

    # =========================
    # Pre-processing
    # =========================
    N_bandits = len(bandits)
    if N_choices >= N_bandits:
        raise IOError("error: N_choices = {}, N_bandits = {}!".format(N_choices, N_bandits))
    action_space = (N_bandits,) * N_choices  # shape dimensions of action space

    # =========================
    # Define bandit class
    # =========================
    class Bandit:
        def __init__(self, bandits):
            self.prob = bandits  # probabilities of success
            self.n = np.zeros(action_space, dtype=np.int)  # number of times action each action (bandit) was used
            self.V = np.zeros(action_space, dtype=np.float)  # estimated value

        def get_reward(self, action):
            reward = 0
            # Loop through every action
            for action_i in action:
                rand = np.random.random()
                if rand < self.prob[action_i]:
                    reward += 1  # success
                else:
                    reward += 0  # failure
            return reward

        # choose action based on epsilon-greedy agent
        def choose_action(self, epsilon, iter):
            if (np.random.random() < epsilon) or (iter == 0):  # random float in [0.0,1.0)
                # Choose random indices from bandits without replacement
                action_explore = tuple(np.random.choice(N_bandits, N_choices, replace=False))  # explore
                action_explore = tuple(np.sort(action_explore))  # sort (to make it unique)
                return action_explore
            else:
                action_greedy = np.unravel_index(self.V.argmax(), self.V.shape)  # exploit
                action_greedy = tuple(np.sort(action_greedy))  # sort (to make it unique)
                return action_greedy

        def update_V(self, action, reward):
            # Update action counter
            self.n[action] += 1
            # Update V
            alpha = 1./self.n[action]
            self.V[action] += alpha * (reward - self.V[action])

    # =========================
    # Define out experiment procedure
    # =========================
    def experiment(bandit, Npulls, epsilon):
        action_history = []
        reward_history = []
        for iter in range(Npulls):
            # Choose action, collect reward, and update value estimates
            action = bandit.choose_action(epsilon, iter)  # choose action (we use epsilon-greedy approach)
            reward = bandit.get_reward(action)  # pick up reward for chosen action
            bandit.update_V(action, reward)  # update our value V estimates for (reward, action)
            # Track action and reward history
            action_history.append(action)
            reward_history.append(reward)
        return (np.array(action_history), np.array(reward_history))

    # =========================
    #
    # Start multi-armed bandit simulation
    #
    # =========================
    print("Running multi-armed bandit simulation with epsilon = {}".format(epsilon))
    reward_history_avg = np.zeros(N_episodes)  # reward history experiment-averaged
    action_history_sum = np.zeros((N_episodes, N_bandits))  # sum action history
    for i in range(N_experiments):
        # Initialize our bandit configuration
        bandit = Bandit(bandits)
        # Perform experiment with epsilon-greedy agent (updates V, and reward history)
        (action_history, reward_history) = experiment(bandit, N_episodes, epsilon)
        # Print
        if (i+1) % (N_experiments/20) == 0:
            print("[{}/{}]".format(i + 1, N_experiments))
            print("  action history = {} ... {}".format(action_history[0], action_history[-1]))
            print("  reward history = {} ... {}".format(reward_history[0], reward_history[-1]))
            print("  average reward = {}".format(np.sum(reward_history)/len(reward_history)))
            print("")
        # Sum up experiment reward (later to be divided to represent an average)
        reward_history_avg += reward_history
        # Sum up action history
        for j, (a) in enumerate(action_history):
            action_history_sum[j][list(a)] += 1

    reward_history_avg /= np.float(N_experiments)
    print("reward history avg = {}".format(reward_history_avg))

    # =========================
    # Plot average reward history results
    # =========================
    plt.plot(reward_history_avg)
    plt.xlabel("iteration #")
    plt.ylabel("average reward (over {} experiments)".format(N_experiments))
    plt.title("MAB group ({}) w/ epsilon-greedy agent (epsilon = {})".format(N_choices, epsilon))
    if save_fig:
        output_file = "results/MAB_group_rewardhistoryavg.pdf"
        plt.savefig(output_file, bbox_inches="tight")
    else:
        plt.show()

    # =========================
    # Plot action history results
    # =========================
    action_history_img = np.transpose(action_history_sum / np.amax(action_history_sum))
    plt.figure(figsize=(16, 16))
    plt.imshow(action_history_img)  # (ypixels, xpixels)
    ax = plt.gca()
    ax.axes.get_xaxis().set_visible(False)
    ax.axes.get_yaxis().set_visible(False)
    plt.gray()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_linewidth(0.0)
    ax.spines['left'].set_linewidth(0.0)
    if save_fig:
        output_file = "results/MAB_group_actionhistory.pdf"
        plt.savefig(output_file, bbox_inches="tight")
    else:
        plt.show()


# Driver
if __name__ == "__main__":
    main()