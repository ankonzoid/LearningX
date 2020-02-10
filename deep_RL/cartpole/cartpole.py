"""

 cartpole.py

"""
import random, gym
import numpy as np
from collections import deque
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam

if __name__ == "__main__":

    class DQNAgent:

        def __init__(self, state_size, action_size):
            self.state_size = state_size
            self.action_size = action_size
            self.memory = deque(maxlen=10000)
            self.gamma = 0.90    # discount rate
            self.epsilon = 1.0   # exploration rate
            self.epsilon_min = 0.01
            self.epsilon_decay = 0.98
            self.learning_rate = 0.001
            self.model = self._build_model()

        def get_action(self, state):
            if np.random.rand() <= self.epsilon:
                # explore
                return random.randrange(self.action_size)
            else:
                # exploit
                return np.argmax(self.model.predict(state)[0])

        def train(self, batch_size=32):
            # Train using replay experience
            minibatch = random.sample(self.memory, batch_size)
            for memory in minibatch:
                state, action, reward, state_next, done = memory
                # Build Q target:
                # -> Qtarget[!action] = Q[!action]
                #    Qtarget[action] = reward + gamma * max[a'](Q_next(state_next))
                Qtarget = self.model.predict(state)
                dQ = reward
                if not done:
                    dQ += self.gamma * np.amax(self.model.predict(state_next)[0])
                Qtarget[0][action] = dQ
                self.model.fit(state, Qtarget, epochs=1, verbose=0)

            # Decary exploration after training
            if self.epsilon > self.epsilon_min:
                self.epsilon *= self.epsilon_decay

        def memorize(self, memory):
            self.memory.append(memory)

        def _build_model(self):
            model = Sequential()
            model.add(Dense(24, input_dim=self.state_size, activation='relu'))
            model.add(Dense(24, activation='relu'))
            model.add(Dense(self.action_size, activation='linear'))
            model.compile(loss='mse', optimizer=Adam(lr=self.learning_rate))
            return model

    # Settings
    n_episodes = 300
    render = False
    batch_size = 128

    # Initialize
    env = gym.make('CartPole-v1')
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n
    agent = DQNAgent(state_size, action_size)
   
    # Train agent
    for episode in range(n_episodes):

        # Generate episode
        state = np.reshape(env.reset(), [1, state_size])  # initial state
        for t in range(500):
            if render:
                env.render()
            action = agent.get_action(state)
            state_next, reward, done, _ = env.step(action)  # evolve state
            state_next = np.reshape(state_next, [1, state_size])
            agent.memorize((state, action, reward, state_next, done))  # store into memory
            if done:
                print("[episode {}/{}] total reward: {}, epsilon: {:.2}".format(episode, n_episodes, t, agent.epsilon))
                break
            state = state_next  # transition to next state

        # Train agent using replay experience
        if len(agent.memory) > batch_size:
            agent.train(batch_size=batch_size)
