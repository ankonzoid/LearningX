"""

 gridworld_CNN.py  (author: Anson Wong / git: ankonzoid)

"""
import numpy as np
import random
import matplotlib.pyplot as plt

from keras.models import Sequential
from keras.layers import Dense, Reshape, Flatten
from keras.optimizers import Adam
from keras.layers.convolutional import Convolution2D

# ==============================
#
# Agent Class
#
# ==============================
class Agent():

    def __init__(self, env, agent_info):
        # Training Parameters
        self.epsilon = agent_info["epsilon"]
        self.epsilon_decay = agent_info["epsilon_decay"]
        self.gamma = agent_info["gamma"]
        self.learning_rate = agent_info["learning_rate"]

        # Memory
        self.clear_memory()

        # Extra parameters
        self.episode = 0
        self.epsilon_effective = self.epsilon

        # Q function
        self.Q = self._build_Q(env)

    # ==================
    # Policy
    # ==================

    def get_action(self, state, env):
        # Reshape 2D state to 3D slice for NN input
        state_Q_input = state.reshape([1, env.Ny, env.Nx])
        # Forward-pass state into Q network
        # Q(s) = [Q(a_1), ..., Q(a_n)]
        Q_state = self.Q.predict(state_Q_input, batch_size=1).flatten()
        # Set zero to the states that are not physically allowed
        N_actions = len(Q_state)
        Q_allowed = []
        actions_allowed = []
        for action in range(N_actions):
            if env.is_allowed_action(state, action):
                Q_allowed.append(action)
                actions_allowed.append(action)
        # Check that there exists non-zero action value
        if np.sum(actions_allowed) == 0:
            raise IOError("Error: at state with no possible actions!")
        # Compute probabilities for each state
        prob = Q_state / np.sum(Q_state)  # action probabilities

        # Sample action based on action probabilities
        #action = np.random.choice(env.action_size, 1, p=prob)[0]
        # Epsilon-greedy selection
        rand = random.uniform(0, 1)
        self.epsilon_effective = self.epsilon * np.exp(-self.epsilon_decay*self.episode)
        choice = ""
        if rand < self.epsilon_effective:
            action = np.random.choice(actions_allowed)
            choice = "explore"
        else:
            Q_max = max(Q_allowed)
            actions_Qmax_allowed = []
            for (action, Q) in zip(actions_allowed, Q_allowed):
                if Q == Q_max:
                    actions_Qmax_allowed.append(action)
            action = np.random.choice(actions_Qmax_allowed)
            choice = "greedy"

        print()
        print(choice)
        print(prob)
        print(action)

        return action, prob

    # ===================
    # Q functions
    # ===================

    def _build_Q(self, env):
        # Build Q(s) function that outputs [Q(a_1), Q(a_2), ..., Q(a_n)]
        Q = Sequential()
        # Reshape 2D to 3D slice
        Q.add(Reshape((1, env.Ny, env.Nx), input_shape=(env.Ny, env.Nx)))
        Q.add(Flatten())
        Q.add(Dense(64, activation="relu", kernel_initializer="he_uniform"))
        #Q.add(Dense(128, activation="relu", kernel_initializer="he_uniform"))
        #Q.add(Convolution2D(32, (6, 6), strides=(3, 3), padding="same", activation="relu", kernel_initializer="he_uniform"))
        #Q.add(Flatten())
        #Q.add(Dense(64, activation="relu", kernel_initializer="he_uniform"))
        #Q.add(Dense(32, activation="relu", kernel_initializer="he_uniform"))
        Q.add(Dense(env.action_size, activation="softmax"))
        # Select optimizer and loss function
        #opt = Adam(lr=self.learning_rate)
        Q.compile(loss="mean_squared_error", optimizer="Adam")
        #Q.compile(loss="mean_squared_error", optimizer=opt)
        # Print architecture of Q network
        Q.summary()
        return Q

    def update_Q(self):
        states = self.state_memory
        actions = self.action_memory
        rewards = self.reward_memory
        probs = self.prob_memory
        gradients = self.gradient_memory

        gradients = np.vstack(gradients)  # stack as row vector matrix
        rewards = np.vstack(rewards)  # stack as row vector matrix
        rewards = self.apply_discount(rewards, self.gamma)  # apply discount factors to reward history
        rewards = rewards / np.std(rewards - np.mean(rewards))
        gradients *= rewards

        # Construct training data
        X = np.squeeze(np.vstack([states]))
        # Construct labels by adding err
        error = self.learning_rate * np.squeeze(np.vstack([gradients]))
        Y = probs + error

        # Train Q network
        self.Q.train_on_batch(X, Y)

    def load_Q(self, filename):
        self.Q.load_weights(filename)

    def save_Q(self, filename):
        self.Q.save_weights(filename)

    # ===================
    # Rewards
    # ===================

    def apply_discount(self, rewards, gamma):
        # If rewards = [r(0), r(1), r(2), ..., r(n-2), r(n-1), r(n)]
        # Then discounted rewards = [, ..., gamma^2*r(n) + gamma*r(n-1) + r(n-2), gamma*r(n) + r(n-1), r(n)]
        discounted_rewards = np.zeros(rewards.shape)
        rsum = 0
        for t in reversed(range(0, rewards.size)):
            # Traverse rewards right -> left (most recent -> least recent)
            # If rewards[t] is non-zero, then discounted_rewards[t] = rewards[t]
            # If rewards[t] is zero, then discounted_rewards[t] = gamma * rsum
            if rewards[t] != 0:
                rsum = 0
            rsum = rsum * gamma + rewards[t]
            discounted_rewards[t] = rsum
        return discounted_rewards

    # ===================
    # Memory
    # ===================

    def compute_gradient(self, action, prob):
        y = np.zeros([prob.shape[0]])
        y[action] = 1
        gradient = np.array(y).astype('float32') - prob
        return gradient

    def append_to_memory(self, state, action, prob, reward, gradient):
        self.state_memory.append(state)
        self.action_memory.append(action)
        self.prob_memory.append(prob)
        self.reward_memory.append(reward)
        self.gradient_memory.append(gradient)

    def clear_memory(self):
        self.state_memory = []
        self.action_memory = []
        self.prob_memory = []
        self.reward_memory = []
        self.gradient_memory = []

# ==============================
#
# Environment Class
#
# ==============================
class Environment():

    def __init__(self, env_info):
        # Environment settings
        self.Ny = env_info["Ny"]
        self.Nx = env_info["Nx"]

        # State and action space
        self.action_dict = {"up": 0, "right": 1, "down": 2, "left": 3}
        self.action_coords = np.array([[-1,0], [0,1], [1,0], [0,-1]], dtype=np.int)
        self.N_actions = len(self.action_coords)

        self.state_dim = (self.Ny, self.Nx)  # tuple of integers
        self.action_dim = (self.N_actions,)  # tuple of integers

        self.state_size = np.prod(np.array(list(self.state_dim), dtype=np.int))  # integer
        self.action_size = np.prod(np.array(list(self.action_dim), dtype=np.int))  # integer

        # Check
        if len(self.action_dict.keys()) != self.N_actions:
            raise IOError("Error: Inconsistent action dimensions!")

    # ===================
    # Starting and terminal state
    # ===================

    def starting_state(self):
        # 2D zero grid with a 1 at top-left corner
        state = np.zeros((self.Ny, self.Nx), dtype=np.int)
        state[0, 0] = 1
        return state

    def is_terminal_state(self, state):
        idx = np.argwhere(np.array(state) == 1)[0]
        if (idx[0], idx[1]) == (self.Ny-1, self.Nx-1):
            return True
        else:
            return False

    # ======================
    # Rewards
    # ======================

    def get_reward(self, state, action):
        reward = -0.01
        idx = np.argwhere(np.array(state) == 1)[0]
        if (idx[0] == self.Ny-2) and (idx[1] == self.Nx-1) and (action == self.action_dict["down"]):
            reward = 100
        if (idx[0] == self.Ny-1) and (idx[1] == self.Nx-2) and (action == self.action_dict["right"]):
            reward = 100
        return reward

    # ======================
    # Apply action
    # ======================

    def is_allowed_action(self, state, action):
        # Find index of agent location in 2D image
        idx_list = np.argwhere(np.array(state) == 1)
        if len(idx_list) != 1:
            raise IOError("Error: Invalid state!")
        idx_new = idx_list[0] + self.action_coords[action]
        # Check
        if (idx_new[0] < 0) or (idx_new[0] >= self.Ny):
            return False
        if (idx_new[1] < 0) or (idx_new[1] >= self.Nx):
            return False
        # If it makes it here, then it is allowed
        return True

    def perform_action(self, state, action):
        # Find index of agent location in 2D image
        idx = np.argwhere(np.array(state) == 1)
        if len(idx) != 1:
            raise IOError("Error: Invalid state!")
        idx_new = idx[0] + self.action_coords[action]
        # Check
        if not self.is_allowed_action(state, action):
            raise IOError("Trying to perform unallowed action")
        # Create new state
        state_new = np.zeros(state.shape, dtype=np.int)
        state_new[tuple(idx_new)] = 1
        return state_new


def main():
    # ==============================
    # Settings
    # ==============================
    N_episodes_train = 10000

    env_info = {"Ny": 3, "Nx": 3}
    agent_info = {"gamma": 0.95, "learning_rate": 0.001, "epsilon": 1.0, "epsilon_decay": 3/N_episodes_train}

    # ==============================
    # Setup environment and agent
    # ==============================
    env = Environment(env_info)
    agent = Agent(env, agent_info)

    # ==============================
    # Train agent
    # ==============================
    for episode in range(N_episodes_train):

        iter = 0
        state = env.starting_state()
        while env.is_terminal_state(state) == False:
            # Pick an action by sampling Q(state) probabilities
            action, prob = agent.get_action(state, env)
            # Collect reward and observe next state
            reward = env.get_reward(state, action)
            state_new = env.perform_action(state, action)
            # Compute loss function
            gradient = agent.compute_gradient(action, prob)
            # Append quantities to memory
            agent.append_to_memory(state, action, prob, reward, gradient)
            # Transition to next state
            state = state_new
            iter += 1

        # Update Q using memory
        agent.update_Q()
        agent.episode += 1

        # Print
        print("[episode {}] iter = {}, epsilon_effective = {} reward = {}".format(episode, iter, round(agent.epsilon_effective, 4), sum(agent.reward_memory)))

        if 0:
            print("{}".format(agent.state_memory))
            print("{}".format(agent.action_memory))
            exit()

        # Clear memory for next episode
        agent.clear_memory()



# Driver
if __name__ == "__main__":
    main()