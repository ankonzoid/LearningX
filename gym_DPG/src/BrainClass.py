"""

 BrainClass.py  (author: Anson Wong / git: ankonzoid)

"""
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Reshape, Flatten
from keras.layers.convolutional import Convolution2D
from keras.models import load_model

class Brain:

    def __init__(self, info):

        # Info
        self.env_info = info["env"]
        self.brain_info = info["brain"]

        # Learning parameters
        self.gamma = self.brain_info["discount"]
        self.learning_rate = self.brain_info["learning_rate"]

        # Policy network function
        if self.brain_info["arch"] == "2D":
            self.PN = self._build_PN_2D()  # 2D input
        elif self.brain_info["arch"] == "1D":
            self.PN = self._build_PN_1D()  # 1D input
        else:
            raise IOError("Invalid architecture selected!")

    def _build_PN_2D(self):
        # Input/output sizes
        input_dim_2D = self.env_info["state_dim"]
        input_dim_3D = (1,) + self.env_info["state_dim"]
        output_size = self.env_info["action_size"]
        # Build DQN architecture (outputs [Q(a_1), Q(a_2), ..., Q(a_n)])
        PN = Sequential()
        PN.add(Reshape(input_dim_3D, input_shape=input_dim_2D))  # Reshape 2D to 3D slice
        PN.add(Convolution2D(64, (2, 2), strides=(1, 1), padding="same", activation="relu", kernel_initializer="he_uniform"))
        PN.add(Flatten())
        PN.add(Dense(64, activation="relu", kernel_initializer="he_uniform"))
        PN.add(Dense(32, activation="relu", kernel_initializer="he_uniform"))
        PN.add(Dense(output_size, activation="softmax"))
        # Select optimizer and loss function
        PN.compile(loss="binary_crossentropy", optimizer="Adam")
        # Print QNN architecture summary
        PN.summary()
        return PN

    def _build_PN_1D(self):
        # Input/output sizes
        input_dim = self.env_info["state_dim"]
        output_size = self.env_info["action_size"]
        # Build DQN architecture (outputs [Q(a_1), Q(a_2), ..., Q(a_n)])
        PN = Sequential()
        PN.add(Dense(32, input_shape=input_dim))
        PN.add(Dense(64, activation="elu"))
        PN.add(Dense(32, activation="elu"))
        PN.add(Dense(output_size, activation="softmax"))
        # Select optimizer and loss function
        PN.compile(loss="mean_squared_error", optimizer="Adam")
        # Print QNN architecture summary
        PN.summary()
        return PN

    def update(self, memory):
        states = memory.state_memory
        actions = memory.action_memory
        rewards = memory.reward_memory
        PNprobs = memory.PNprob_memory
        probs = memory.prob_memory

        gamma = self.gamma
        learning_rate = self.learning_rate

        def compute_gradients(actions, probs):
            # The gradient is the correction in the action probabilities
            # gradient = onehotvec(a) - prob
            def gradient(action, prob):
                y = np.zeros([prob.shape[0]])
                y[action] = 1
                gradient = np.array(y).astype('float32') - prob
                return gradient

            gradients = []
            for (action, prob) in zip(actions, probs):
                gradients.append(gradient(action, prob))
            return gradients

        def compute_discounted_rewards_total(rewards, gamma):
            # If rewards = [r(0), r(1), r(2), ..., r(n-2), r(n-1), r(n)]
            # Then discounted rewards = [..., r(n-2) + gamma*r(n-1) + gamma^2*r(n), r(n-1) + gamma*r(n), r(n)]
            # We compute this by going through the list in reverse order
            rewards_total = [0.0] * len(rewards)
            rsum = 0.0
            for t in reversed(range(len(rewards))):
                rsum = rewards[t] + gamma * rsum
                rewards_total[t] = rsum
            return rewards_total

        # Compute discounted total rewards, and vertically stack as row vector matrix (2D)
        gradients = np.vstack(compute_gradients(actions, probs))
        discounted_rewards_total = np.vstack(compute_discounted_rewards_total(rewards, gamma))
        discounted_rewards_total /= np.std(discounted_rewards_total)

        # Compute loss scaled by discounted rewards
        loss = discounted_rewards_total * gradients

        # Construct training data of states
        X = np.squeeze(np.vstack([states]))

        # Construct labels by adding loss
        dPNprobs = learning_rate * np.squeeze(np.vstack([loss]))
        Y = PNprobs + dPNprobs

        # Train Q network
        self.PN.train_on_batch(X, Y)

    # ==================================
    # IO functions
    # ==================================

    def load_PN(self, filename):
        self.PN = load_model(filename)
        self.PN.compile(loss="binary_crossentropy", optimizer="Adam")
        self.PN.summary()

    def save_PN(self, filename):
        self.PN.save(filename)