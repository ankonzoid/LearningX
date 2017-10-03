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
        # Parameters
        self.gamma = 0.99
        self.learning_rate = 0.001

        # Memory
        self.states = []
        self.gradients = []
        self.rewards = []
        self.probs = []

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
    # Training functions
    # ===================

    def get_action(self, state, env):
        # Reshape 2D state to 3D slice for NN input
        state = state.reshape([1, env.Ny, env.Nx])
        aprob = self.Q.predict(state, batch_size=1).flatten()  # forward pass
        self.probs.append(aprob)
        prob = aprob / np.sum(aprob)
        action = np.random.choice(env.action_size, 1, p=prob)[0]
        return action, prob

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
        self.N_actions = np.arange(len(self.action_coords))

        self.state_dim = (self.Ny, self.Nx)  # tuple of integers
        self.action_dim = (self.N_actions,)  # tuple of integers

        self.state_size = np.prod(np.array(list(self.state_dim), dtype=np.int))  # integer
        self.action_size = np.prod(np.array(list(self.action_dim), dtype=np.int))  # integer

        self.reward = self._define_reward()

        # Check
        if len(self.action_dict.keys()) != self.N_actions:
            raise IOError("Error: Inconsistent action dimensions!")


    def _define_reward(self):
        reward = np.zeros((self.Ny, self.Nx))
        reward[(self.Ny-10):, (self.Nx-10):] = 100
        return reward

    def starting_state(self):
        state = [0, 0]
        return state

    def get_reward(self, state, action):
        reward = -1
        if (state == (self.Ny-2, self.Nx-1)) and (action == self.action_dict["right"]):
            reward = 100
        if (state == (self.Ny-1, self.Nx-2)) and (action == self.action_dict["down"]):
            reward = 100
        return reward

    def perform_action(self, state, action, env):
        state_new = state + env.action_coords[action]
        return state_new


def main():

    env = Environment()
    agent = Agent(env)

    N_episodes_train = 100

    state = env.starting_state()
    for episode in range(N_episodes_train):
        print("episode {}".format(episode))

        action, prob = agent.get_action(state)  # pick action using Q
        reward = env.get_reward(state, action)  # collect reward
        state_new = env.perform_action(state, action)  # observe next state
        agent.update(state, state_new, action, reward)  # update Q
        state = state_new  # transition to next state



# Driver
if __name__ == "__main__":
    main()