import os, random, csv
import numpy as np
from collections import deque
import gym
import tensorflow as tf
random.seed(0)
np.random.seed(0)
gpu_devices = tf.config.experimental.list_physical_devices('GPU')
for device in gpu_devices:
    tf.config.experimental.set_memory_growth(device, True)

class DQN():

    def __init__(self, state_size, action_size, gamma=0.95, learning_rate=0.1, hidden_layers=[6, 6]):
        self.state_size = state_size
        self.action_size = action_size
        self.gamma = gamma # discount factor
        self.learning_rate = learning_rate
        self.hidden_layers = hidden_layers
        self._build_model(state_size=state_size, action_size=action_size, hidden_layers=hidden_layers)

    def get_action(self, state, epsilon, allowed=None):
        if np.random.rand() < epsilon: # explore
            return random.choice(allowed) if (allowed is None) else random.randrange(self.action_size)
        else: # exploit
            state_reshaped = np.reshape(state, [1, self.state_size])
            Q = self.model.predict(state_reshaped)[0]
            return np.argmax(Q) if (allowed is None) else allowed[np.argmax(Q[allowed])]

    def train(self, memories, epochs):
        X, y = [], []
        for (state, action, reward, state_next, done) in memories:
            state_reshaped = np.reshape(state, [1, self.state_size])
            state_next_reshaped = np.reshape(state_next, [1, self.state_size])
            Qtarget = self.model.predict(state_reshaped)[0] # Qtarget[!action] = Q[!action]
            Qtarget[action] = reward
            if not done: # Qtarget[action] = reward + gamma * max[a'](Q_next(state_next))
                Qtarget[action] += self.gamma * np.amax(self.model.predict(state_next_reshaped)[0])
            X.append(state)
            y.append(Qtarget)
        self.model.fit(np.array(X), np.array(y), epochs=epochs, verbose=0)

    def _build_model(self, state_size, action_size, hidden_layers):
        if len(hidden_layers) < 1:
            raise Exception("Insert hidden layers!")
        layers = []
        layers.append(tf.keras.layers.Dense(hidden_layers[0], input_dim=state_size, activation='relu'))
        for i in range(1, len(hidden_layers)):
            layers.append(tf.keras.layers.Dense(hidden_layers[i], activation='relu'))
        layers.append(tf.keras.layers.Dense(action_size, activation='linear'))
        model = tf.keras.Sequential(layers)
        model.compile(loss='mse', optimizer=tf.keras.optimizers.Adam(self.learning_rate))
        self.model = model

# Settings
envname = "LunarLander-v2"
render = False

n_episodes = 5000  # number of rounds
n_iter_max = 1000
hidden_layers = [60, 40]
eps = 0.05  # initial exploration
eps_min = 0.01  # lower boundary on exploration
eps_decay = 0.99  # exploration decay factor per round
batch_size = 128  # batch size to train agent
epochs = 1  # number of epochs to train agent
rewards_recent = deque(maxlen=10)
replay_buffer = deque(maxlen=20000)

# Set up environment
env = gym.make(envname)
env.seed(0)
state_size = env.observation_space.shape[0]
action_size = env.action_space.n

# Set up agent
agent = DQN(state_size=state_size, action_size=action_size, gamma=1.0, learning_rate=0.1, hidden_layers=hidden_layers)

# Train agent
for episode in range(n_episodes):
    state = env.reset() # initial state
    reward_episode = 0
    for iter in range(n_iter_max):
        if render:
            env.render()
        action = agent.get_action(state=state, epsilon=eps) # sample policy
        state_next, reward, done, _ = env.step(action) # evolve state
        replay_buffer.append((state, action, reward, state_next, done)) # store to buffer
        reward_episode += reward # sum reward
        if done:
            break
        state = state_next # transition to next state
    rewards_recent.append(reward_episode)
    print("[episode={}] r_episode={:.2f}, r_mean={:.2f}, eps={:.3f}".format(episode + 1, reward_episode, np.mean(rewards_recent), eps))
    if batch_size < len(replay_buffer):
        replay_buffer_sample = random.sample(replay_buffer, batch_size)
        agent.train(replay_buffer_sample, epochs=epochs) # train
        eps = max(eps_decay * eps, eps_min) # decay exploration