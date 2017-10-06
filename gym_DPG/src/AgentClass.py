"""

 AgentClass.py  (author: Anson Wong / git: ankonzoid)

"""
import numpy as np
import random

class Agent:

    def __init__(self, info):

        # Info
        self.env_info = info["env"]
        self.agent_info = info["agent"]

        self.state_dim = self.env_info["state_dim"]
        self.policy_mode = self.agent_info["policy_mode"]

        # Keep track of agent episodes
        self.episode = 0

    # ==================
    # Policy
    # ==================

    def get_action(self, state, brain):

        # Reshape 2D state to 3D slice for NN input
        slice_dim_3D = (1,) + self.state_dim
        state_PN_input = state.reshape(list(slice_dim_3D))

        # Forward-pass state into PN network
        # PN(s) = [PN(a_1), ..., PN(a_n)]
        PNprob = brain.PN.predict(state_PN_input, batch_size=1).flatten()

        # Set zero to the states that are not physically allowed
        N_actions = len(PNprob)
        PNprob_allowed = []
        actions_allowed = []
        for action in range(N_actions):
            PNprob_allowed.append(PNprob[action])
            actions_allowed.append(action)
        PNprob_allowed = np.array(PNprob_allowed, dtype=np.float32)
        actions_allowed = np.array(actions_allowed, dtype=np.int)

        # Check that there exists at least 1 allowed action
        if np.sum(actions_allowed) == 0:
            raise IOError("Error: at state with no possible actions!")

        # Compute probabilities for each state
        prob = PNprob / np.sum(PNprob)  # action probabilities

        # Follow a policy method and select an action stochastically
        if (self.policy_mode == "epsilongreedy"):

            # Epsilon-greedy parameters
            self.epsilon = self.agent_info["epsilon"]
            self.epsilon_decay = self.agent_info["epsilon_decay"]
            self.epsilon_effective = self.epsilon * np.exp(-self.epsilon_decay*self.episode)

            if random.uniform(0, 1) < self.epsilon_effective:
                action = np.random.choice(actions_allowed)
            else:
                PN_max = max(PNprob_allowed)
                actions_PNmax_allowed = []
                for (action, PN) in zip(actions_allowed, PNprob_allowed):
                    if PN == PN_max:
                        actions_PNmax_allowed.append(action)
                action = np.random.choice(actions_PNmax_allowed)

        elif (self.policy_mode == "softmax"):

            # Sample action based on action probabilities
            prob_actions_allowed = PNprob_allowed / np.sum(PNprob_allowed)
            idx = np.random.choice(len(actions_allowed), 1, p=prob_actions_allowed)[0]
            action = actions_allowed[idx]

        else:
            raise IOError("Error: invalid policy mode!")

        return action, PNprob, prob