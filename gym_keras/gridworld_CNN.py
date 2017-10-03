"""

 pong.py  (author: Anson Wong / git: ankonzoid)

"""
import numpy as np
import matplotlib.pyplot as plt

from keras.models import Sequential
from keras.layers import Dense, Reshape, Flatten
from keras.optimizers import Adam
from keras.layers.convolutional import Convolution2D

class Agent():

    def __init__(self, env):
        # Training Parameters
        self.gamma = 0.99
        self.learning_rate = 0.001

        # Memory
        self.states_history_episode = []
        self.rewards_history_episode = []
        self.prob_actions_history_episode = []

        # Q function
        self.Q = self._build_Q(env)
        self.Q.summary()

    def _build_Q(self, env):
        # Build Q(s) function that outputs [Q(a_1), Q(a_2), ..., Q(a_n)]
        Q = Sequential()
        # Reshape 2D to 3D slice
        Q.add(Reshape((1, env.Ny, env.Nx),
                      input_shape=(env.Ny, env.Nx)))
        Q.add(Convolution2D(32, (6, 6), strides=(3, 3), padding="same",
                            activation="relu", kernel_initializer="he_uniform"))
        Q.add(Flatten())
        Q.add(Dense(64, activation="relu", kernel_initializer="he_uniform"))
        Q.add(Dense(32, activation="relu", kernel_initializer="he_uniform"))
        Q.add(Dense(env.action_size, activation="softmax"))
        opt = Adam(lr=self.learning_rate)
        Q.compile(loss="categorical_crossentropy", optimizer=opt)
        return Q

    # ===================
    # Update functions
    # ===================

    def update_Q(self, state, action, reward, state_new):
        gradients = np.vstack(self.gradients)
        rewards = np.vstack(self.rewards)
        #rewards = self.discount_rewards(rewards)
        rewards = rewards / np.std(rewards - np.mean(rewards))
        gradients *= rewards
        X = np.squeeze(np.vstack([self.states]))
        Y = self.probs + self.learning_rate * np.squeeze(np.vstack([gradients]))
        self.Q.train_on_batch(X, Y)
        # Reset episodic memory

    def clear_memory(self):
        self.states = []
        self.probs = []
        self.gradients = []
        self.rewards = []

    def get_action(self, state, env):
        # Reshape 2D state to 3D slice for NN input
        state = state.reshape([1, env.Ny, env.Nx])
        Q_actions = self.Q.predict(state, batch_size=1).flatten()  # forward pass
        prob_actions = Q_actions / np.sum(Q_actions)  # normalize to show probabilities
        action = np.random.choice(env.action_size, 1, p=prob_actions)[0]  # softmax (boltzmann) selection
        return action, prob_actions, Q_actions

    # ===================
    # I/O functions
    # ===================

    def load(self, filename):
        self.Q.load_weights(filename)

    def save(self, filename):
        self.Q.save_weights(filename)

    def update(self, state, state_new, action, reward):
        self.Q = 0

class Environment():
    def __init__(self):
        self.Ny = 100
        self.Nx = 100

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

    def get_reward(self, state, action):
        reward = -1
        idx = np.argwhere(state == 1)[0]
        if ((idx[0],idx[1]) == (self.Ny-2, self.Nx-1)) and (action == self.action_dict["right"]):
            reward = 100
        if ((idx[0],idx[1]) == (self.Ny-1, self.Nx-2)) and (action == self.action_dict["down"]):
            reward = 100
        return reward

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


    env = Environment()
    agent = Agent(env)

    N_episodes_train = 100

    for episode in range(N_episodes_train):
        print("episode {}".format(episode))

        iter = 0
        state = env.starting_state()
        while env.is_terminal_state(state) == False:
            print("iter {}".format(iter))

            action, prob_actions, Q_actions = agent.get_action(state, env)  # pick action using Q

            reward = env.get_reward(state, action)  # collect reward
            state_new = env.perform_action(state, action)  # observe next state

            agent.update_Q(state, action, reward, state_new)  # update Q

            #
            agent.rewards_history_episode.append(reward)
            agent.prob_actions_history_episode.append(prob_actions)

            state = state_new  # transition to next state

        # Clear memory for next episode
        agent.clear_memory()



# Driver
if __name__ == "__main__":
    main()