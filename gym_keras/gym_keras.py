"""

 pong.py  (author: Anson Wong / git: ankonzoid)

"""
import gym
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Reshape, Flatten
from keras.optimizers import Adam
from keras.layers.convolutional import Convolution2D


class Agent:

    def __init__(self, state_dim, action_dim):
        self.state_size = state_dim
        self.action_size = action_dim
        self.gamma = 0.99
        self.learning_rate = 0.001
        self.states = []
        self.gradients = []
        self.rewards = []
        self.probs = []
        self.model = None

    def _build_model(self):
        model = Sequential()
        model.add(Reshape((1, 80, 80), input_shape=(self.state_size,)))
        model.add(Convolution2D(32, 6, 6, subsample=(3, 3), border_mode='same',
                                activation='relu', init='he_uniform'))
        model.add(Flatten())
        model.add(Dense(64, activation='relu', init='he_uniform'))
        model.add(Dense(32, activation='relu', init='he_uniform'))
        model.add(Dense(self.action_size, activation='softmax'))
        opt = Adam(lr=self.learning_rate)
        model.compile(loss='categorical_crossentropy', optimizer=opt)
        return model

def main():
    env = gym.make("Pong-v0")
    #env = gym.make('Asteroids-v0')

    state = env.reset()

    print(state.shape)

    import matplotlib.pyplot as plt

    plt.imshow(state)
    plt.show()
    exit()

    # =============================
    # Environment
    # =============================
    if "n" in env.action_space.__dict__ or ('low' in env.action_space.__dict__ and 'high' in env.action_space.__dict__):
        print("action space = {}".format(env.action_space))

    if "n" in env.observation_space.__dict__ or ('low' in env.observation_space.__dict__ and 'high' in env.observation_space.__dict__):
        print("observation space = {}".format(env.observation_space))




    state_dim = None
    action_dim = None
    agent = Agent(state_dim, action_dim)

    episode = 0
    while True:

        env.render()

        action = 0
        env.step(action)

        state = env.reset()

        done = True

        episode += 1



    exit()

    state = env.reset()
    prev_x = None
    score = 0
    episode = 0

    state_size = 80 * 80
    action_size = env.action_space.n
    agent = PGAgent(state_size, action_size)
    #agent.load('pong.h5')
    while True:
        env.render()

        cur_x = preprocess(state)
        x = cur_x - prev_x if prev_x is not None else np.zeros(state_size)
        prev_x = cur_x

        action, prob = agent.act(x)
        state, reward, done, info = env.step(action)
        score += reward
        agent.remember(x, action, prob, reward)

        if done:
            episode += 1
            agent.train()
            print('Episode: %d - Score: %f.' % (episode, score))
            score = 0
            state = env.reset()
            prev_x = None
            if episode > 1 and episode % 10 == 0:
                agent.save('pong.h5')

# Driver
if __name__ == "__main__":
    main()