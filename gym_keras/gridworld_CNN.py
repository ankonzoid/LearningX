"""

 gridworld_CNN.py  (author: Anson Wong / git: ankonzoid)

"""
import numpy as np
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
        self.gamma = agent_info["gamma"]
        self.learning_rate = agent_info["learning_rate"]

        # Memory
        self.clear_memory()

        # Q function
        self.Q = self._build_Q(env)

    # ==================
    # Policy
    # ==================

    def get_action(self, state, env):
        # Reshape 2D state to 3D slice for NN input
        state = state.reshape([1, env.Ny, env.Nx])
        Q_state = self.Q.predict(state, batch_size=1).flatten()  # forward pass -> Q(s) = [Q(a_1), ..., Q(a_n)]
        prob = Q_state / np.sum(Q_state)  # action probabilities
        action = np.random.choice(env.action_size, 1, p=prob)[0]  # sample action based on action probabilities
        return action, prob

    # ===================
    # Q functions
    # ===================

    def _build_Q(self, env):
        # Build Q(s) function that outputs [Q(a_1), Q(a_2), ..., Q(a_n)]
        Q = Sequential()
        # Reshape 2D to 3D slice
        Q.add(Reshape((1, env.Ny, env.Nx), input_shape=(env.Ny, env.Nx)))
        Q.add(Convolution2D(32, (6, 6), strides=(3, 3), padding="same", activation="relu", kernel_initializer="he_uniform"))
        Q.add(Flatten())
        Q.add(Dense(64, activation="relu", kernel_initializer="he_uniform"))
        Q.add(Dense(32, activation="relu", kernel_initializer="he_uniform"))
        Q.add(Dense(env.action_size, activation="softmax"))
        # Select optimizer and loss function
        opt = Adam(lr=self.learning_rate)
        Q.compile(loss="categorical_crossentropy", optimizer=opt)
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
        discounted_rewards = np.zeros(rewards.shape)
        running_add = 0
        # Traverse rewards right -> left (most recent -> least recent)
        for t in reversed(range(0, rewards.size)):
            if rewards[t] != 0:
                running_add = 0
            running_add = running_add * gamma + rewards[t]
            discounted_rewards[t] = running_add
        return discounted_rewards

    # ===================
    # Memory
    # ===================

    def _compute_gradient(self, action, prob):
        y = np.zeros([prob.shape[0]])
        y[action] = 1
        gradient = np.array(y).astype('float32') - prob
        return gradient

    def append_to_memory(self, state, action, prob, reward):
        self.state_memory.append(state)
        self.action_memory.append(action)
        self.prob_memory.append(prob)
        self.reward_memory.append(reward)
        self.gradient_memory.append(self._compute_gradient(action, prob))

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
        state[50, 50] = 1
        return state

    def is_terminal_state(self, state):
        idx = np.argwhere(state == 1)[0]
        if (idx[0], idx[1]) == (self.Ny-1, self.Nx-1):
            return True
        else:
            return False

    # ======================
    # Rewards
    # ======================

    def get_reward(self, state, action):
        reward = -1
        idx = np.argwhere(state == 1)[0]
        if ((idx[0],idx[1]) == (self.Ny-2, self.Nx-1)) and (action == self.action_dict["right"]):
            reward = 100
        if ((idx[0],idx[1]) == (self.Ny-1, self.Nx-2)) and (action == self.action_dict["down"]):
            reward = 100
        return reward

    # ======================
    # Apply action
    # ======================

    def perform_action(self, state, action):
        # Find index of agent location in 2D image
        idx = np.argwhere(state == 1)
        if len(idx) != 1:
            raise IOError("Error: Invalid state!")
        idx_new = idx[0] + self.action_coords[action]
        # Check
        if (idx_new[0] < 0) or (idx_new[0] >= self.Ny):
            raise IOError("Error: Performed invalid action!")
        if (idx_new[1] < 0) or (idx_new[1] >= self.Nx):
            raise IOError("Error: Performed invalid action!")
        # Create new state
        state_new = np.zeros(state.shape, dtype=np.int)
        state_new[tuple(idx_new)] = 1
        return state_new


def main():
    # ==============================
    # Settings
    # ==============================
    N_episodes_train = 100

    env_info = {"Ny": 100, "Nx": 100}
    agent_info = {"gamma": 0.99, "learning_rate": 0.001}

    A = np.zeros([5])
    print(A)
    print(A.shape)

    exit()

    # ==============================
    # Setup environment and agent
    # ==============================
    env = Environment(env_info)
    agent = Agent(env, agent_info)

    # ==============================
    # Train agent
    # ==============================
    for episode in range(N_episodes_train):
        print("episode {}".format(episode))

        iter = 0
        state = env.starting_state()
        while env.is_terminal_state(state) == False:
            print("iter {}".format(iter))
            # Pick an action by sampling Q(state) probabilities
            action, prob = agent.get_action(state, env)
            # Collect reward and observe next state
            reward = env.get_reward(state, action)
            state_new = env.perform_action(state, action)
            # Append to memory (states, actions, probs, rewards, gradients)
            agent.append_to_memory(state, action, prob, reward, env)
            # Update Q using appended memory
            agent.update_Q()
            # Transition to next state
            state = state_new
            iter += 1

        # Clear memory for next episode
        agent.clear_memory()



# Driver
if __name__ == "__main__":
    main()