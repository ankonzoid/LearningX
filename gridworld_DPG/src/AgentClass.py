"""

 AgentClass.py  (author: Anson Wong / git: ankonzoid)

"""
import numpy as np
import random

class Agent:

    def __init__(self, env, agent_info):

        # Agent policy parameters
        self.policy_mode = agent_info["policy_mode"]

        # Epsilon-greedy parameters
        self.epsilon = agent_info["epsilon"]
        self.epsilon_decay = agent_info["epsilon_decay"]
        self.epsilon_effective = self.epsilon

        self.episode = 0

    # ==================
    # Policy
    # ==================

    def get_action(self, state, brain, env):
        # Reshape 2D state to 3D slice for NN input
        state_PN_input = state.reshape([1, env.Ny, env.Nx])

        # Forward-pass state into Q network
        # Q(s) = [Q(a_1), ..., Q(a_n)]
        PNprob = brain.PN.predict(state_PN_input, batch_size=1).flatten()

        # Set zero to the states that are not physically allowed
        N_actions = len(PNprob)
        PN_allowed = []
        actions_allowed = []
        for action in range(N_actions):
            if env.is_allowed_action(state, action):
                PN_allowed.append(PNprob[action])
                actions_allowed.append(action)

        # Check that there exists at least 1 allowed action
        if np.sum(actions_allowed) == 0:
            raise IOError("Error: at state with no possible actions!")

        # Compute probabilities for each state
        prob = PNprob / np.sum(PNprob)  # action probabilities

        # Follow a policy method and select an action stochastically
        if (self.policy_mode == "epsilongreedy"):

            # Epsilon-greedy selection
            self.epsilon_effective = self.epsilon * np.exp(-self.epsilon_decay*self.episode)

            if random.uniform(0, 1) < self.epsilon_effective:
                action = np.random.choice(actions_allowed)
            else:
                PN_max = max(PN_allowed)
                actions_PNmax_allowed = []
                for (action, Q) in zip(actions_allowed, PN_allowed):
                    if PN == PN_max:
                        actions_PNmax_allowed.append(action)
                action = np.random.choice(actions_PNmax_allowed)

        elif (self.policy_mode == "softmax"):

            # Sample action based on action probabilities
            action = np.random.choice(env.action_size, 1, p=prob)[0]

        else:
            raise IOError("Error: invalid policy mode!")

        return action, PNprob, prob