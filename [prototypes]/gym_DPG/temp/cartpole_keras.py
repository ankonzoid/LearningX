import gym
import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
from collections import deque

# Create the Cart-Pole game environment
env = gym.make('CartPole-v0')


class QNetwork:
    def __init__(self, learning_rate=0.01, state_size=4,
                 action_size=2, hidden_size=10):
        # state inputs to the Q-network
        self.model = Sequential()

        self.model.add(Dense(hidden_size, activation='relu',
                             input_dim=state_size))
        self.model.add(Dense(hidden_size, activation='relu'))
        self.model.add(Dense(action_size, activation='linear'))

        self.optimizer = Adam(lr=learning_rate)
        self.model.compile(loss='mse', optimizer=self.optimizer)


class Memory():
    def __init__(self, max_size=1000):
        self.buffer = deque(maxlen=max_size)

    def add(self, experience):
        self.buffer.append(experience)

    def sample(self, batch_size):
        idx = np.random.choice(np.arange(len(self.buffer)),
                               size=batch_size,
                               replace=False)
        return [self.buffer[ii] for ii in idx]


train_episodes = 200          # max number of episodes to learn from
max_steps = 200                # max steps in an episode
gamma = 0.99                   # future reward discount

# Exploration parameters
explore_start = 1.0            # exploration probability at start
explore_stop = 0.01            # minimum exploration probability
decay_rate = 0.0001            # exponential decay rate for exploration prob

# Network parameters
hidden_size = 16               # number of units in each Q-network hidden layer
learning_rate = 0.001         # Q-network learning rate

# Memory parameters
memory_size = 10000            # memory capacity
batch_size = 32                # experience mini-batch size
pretrain_length = batch_size   # number experiences to pretrain the memory

mainQN = QNetwork(hidden_size=hidden_size, learning_rate=learning_rate)

###################################
## Populate the experience memory
###################################

# Initialize the simulation
env.reset()
# Take one random step to get the pole and cart moving
state, reward, done, _ = env.step(env.action_space.sample())
state = np.reshape(state, [1, 4])

memory = Memory(max_size=memory_size)

# Make a bunch of random actions and store the experiences
for ii in range(pretrain_length):
    # Uncomment the line below to watch the simulation
    # env.render()

    # Make a random action
    action = env.action_space.sample()
    next_state, reward, done, _ = env.step(action)
    next_state = np.reshape(next_state, [1, 4])

    if done:
        # The simulation fails so no next state
        next_state = np.zeros(state.shape)
        # Add experience to memory
        memory.add((state, action, reward, next_state))

        # Start new episode
        env.reset()
        # Take one random step to get the pole and cart moving
        state, reward, done, _ = env.step(env.action_space.sample())
        state = np.reshape(state, [1, 4])
    else:
        # Add experience to memory
        memory.add((state, action, reward, next_state))
        state = next_state

#############
## Training
#############
step = 0
for ep in range(1, train_episodes):
    total_reward = 0
    t = 0
    while t < max_steps:
        step += 1
        # Uncomment this next line to watch the training
        # env.render()

        # Explore or Exploit
        explore_p = explore_stop + (explore_start - explore_stop)*np.exp(-decay_rate*step)
        if explore_p > np.random.rand():
            # Make a random action
            action = env.action_space.sample()
        else:
            # Get action from Q-network
            Qs = mainQN.model.predict(state)[0]
            action = np.argmax(Qs)

        # Take action, get new state and reward
        next_state, reward, done, _ = env.step(action)
        next_state = np.reshape(next_state, [1, 4])
        total_reward += reward

        if done:
            # the episode ends so no next state
            next_state = np.zeros(state.shape)
            t = max_steps

            print('Episode: {}'.format(ep),
                  'Total reward: {}'.format(total_reward),
                  'Explore P: {:.4f}'.format(explore_p))

            # Add experience to memory
            memory.add((state, action, reward, next_state))

            # Start new episode
            env.reset()
            # Take one random step to get the pole and cart moving
            state, reward, done, _ = env.step(env.action_space.sample())
            state = np.reshape(state, [1, 4])
        else:
            # Add experience to memory
            memory.add((state, action, reward, next_state))
            state = next_state
            t += 1

        # Replay
        inputs = np.zeros((batch_size, 4))
        targets = np.zeros((batch_size, 2))

        minibatch = memory.sample(batch_size)
        for i, (state_b, action_b, reward_b, next_state_b) in enumerate(minibatch):
            inputs[i:i+1] = state_b
            target = reward_b
            if not (next_state_b == np.zeros(state_b.shape)).all(axis=1):
                target_Q = mainQN.model.predict(next_state_b)[0]
                target = reward_b + gamma * np.amax(mainQN.model.predict(next_state_b)[0])
            targets[i] = mainQN.model.predict(state_b)
            targets[i][action_b] = target
        mainQN.model.fit(inputs, targets, epochs=1, verbose=0)