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
        self.model = self._build_model(env)

        self.Q = 0


    def _build_model(self, env):
        model = Sequential()
        model.add(Reshape((1, env.Ny, env.Nx), input_shape=(self.state_size,)))
        model.add(Convolution2D(32, (6, 6), strides=(3, 3), padding="same",
                                activation="relu", kernel_initializer="he_uniform"))
        model.add(Flatten())
        model.add(Dense(64, activation="relu", kernel_initializer="he_uniform"))
        model.add(Dense(32, activation="relu", kernel_initializer="he_uniform"))
        model.add(Dense(self.action_size, activation="softmax"))
        opt = Adam(lr=self.learning_rate)
        model.compile(loss="categorical_crossentropy", optimizer=opt)
        return model



    # ===================
    # Training functions
    # ===================

    def get_action(self, state):
        state = state.reshape([1, state.shape[0]])
        aprob = self.model.predict(state, batch_size=1).flatten()
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

        self.grid = np.zeros((100,100))
        self.grid[:10, :10] = 1

        self.actions = np.arange(4)

        self.Ny = 100
        self.Nx = 100
        self.state_dim = self.grid.shape[0] * self.grid.shape[1]
        self.action_dim = len(self.actions)

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


    plt.imshow(env.grid)
    plt.show()
    
    exit()

    N_episodes_train = 100

    state = env.starting_state()
    for episode in range(N_episodes_train):
        print("episode {}".format(episode))
        action, prob = agent.get_action(state)
        reward = env.get_reward(state, action)
        state_new = env.perform_action(state, action)

        agent.update(state, state_new, action, reward)

        state = state_new



# Driver
if __name__ == "__main__":
    main()