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
        self.state_size = env.state_dim
        self.action_size = env.action_dim
        self.gamma = 0.99
        self.learning_rate = 0.001
        self.states = []
        self.gradients = []
        self.rewards = []
        self.probs = []


        self.Q = self._build_Q(env)

    def _build_Q(self, env):
        Q = Sequential()
        Q.add(Reshape((1, env.Ny, env.Nx), input_shape=(self.state_size,)))
        Q.add(Convolution2D(32, (6, 6), strides=(3, 3), padding="same",
                                activation="relu", kernel_initializer="he_uniform"))
        Q.add(Flatten())
        Q.add(Dense(64, activation="relu", kernel_initializer="he_uniform"))
        Q.add(Dense(32, activation="relu", kernel_initializer="he_uniform"))
        Q.add(Dense(self.action_size, activation="softmax"))
        opt = Adam(lr=self.learning_rate)
        Q.compile(loss="categorical_crossentropy", optimizer=opt)
        return Q

    # ===================
    # Training functions
    # ===================

    def get_action(self, state):
        state = state.reshape([1, state.shape[0]])
        aprob = self.Q.predict(state, batch_size=1).flatten()  # forward pass
        self.probs.append(aprob)
        prob = aprob / np.sum(aprob)
        action = np.random.choice(self.action_size, 1, p=prob)[0]
        return action, prob

    # ===================
    # I/O functions
    # ===================

    def load(self, name):
        self.model.load_weights(name)

    def save(self, name):
        self.model.save_weights(name)

    def update(self, state, state_new, action, reward):
        self.Q = 0

class Environment():
    def __init__(self):

        self.Ny = 100
        self.Nx = 100

        self.actions = np.arange(4)

        self.state_dim = self.Ny * self.Nx
        self.action_dim = len(self.actions)

        self.reward = self._define_reward()

    def _define_reward(self):
        reward = np.zeros((self.Ny, self.Nx))
        reward[(self.Ny-10):, (self.Nx-10):] = 100
        return reward

    def starting_state(self):
        state = [0, 0]
        return state

    def get_reward(self, state, action):
        reward = state + action
        return reward

    def perform_action(self, state, action):
        state_new = state
        return state_new


def main():

    env = Environment()
    agent = Agent(env)

    plt.figure(figsize=(8, 8))
    plt.imshow(env.reward)
    plt.show()
    
    exit()

    N_episodes_train = 100

    state = env.starting_state()
    for episode in range(N_episodes_train):

        print("episode {}".format(episode))

        action, prob = agent.get_action(state)  # pick action
        reward = env.get_reward(state, action)  # collect reward
        state_new = env.perform_action(state, action)  # observe next state

        agent.update(state, state_new, action, reward)  # update Q

        state = state_new  # transition to next state



# Driver
if __name__ == "__main__":
    main()