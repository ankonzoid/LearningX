"""

 AgentClass.py  (author: Anson Wong / git: ankonzoid)

"""
import numpy as np
import random

class Agent:

    def __init__(self, env, info):

        # Agent info
        self.agent_info = info["agent"]

        # Keep track of agent episodes
        self.episode = 0

    # ==================
    # Policy
    # ==================

    def get_action(self, state, brain, env):

        # Reshape 2D state to 3D slice for NN input
        state_PN_input = state.reshape([1, env.Ny, env.Nx])

        # Forward-pass state into PN network
        # PN(s) = [PN(a_1), ..., PN(a_n)]
        PNprob = brain.PN.predict(state_PN_input, batch_size=1).flatten()

        # Set zero to the states that are not physically allowed
        N_actions = len(PNprob)
        PNprob_allowed = []
        actions_allowed = []
        for action in range(N_actions):
            if env.is_allowed_action(state, action):
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
        policy_mode = self.agent_info["policy_mode"]
        if (policy_mode == "epsgreedy"):

            # Epsilon-greedy parameters
            self.eps = self.agent_info["eps"]
            self.eps_decay = self.agent_info["eps_decay"]
            self.eps_effective = self.eps * np.exp(-self.eps_decay*self.episode)

            if random.uniform(0, 1) < self.eps_effective:
                action = np.random.choice(actions_allowed)
            else:
                PN_max = max(PNprob_allowed)
                actions_PNmax_allowed = []
                for (action, PN) in zip(actions_allowed, PNprob_allowed):
                    if PN == PN_max:
                        actions_PNmax_allowed.append(action)
                action = np.random.choice(actions_PNmax_allowed)

        elif (policy_mode == "softmax"):

            # Sample action based on action probabilities
            prob_actions_allowed = PNprob_allowed / np.sum(PNprob_allowed)
            idx = np.random.choice(len(actions_allowed), 1, p=prob_actions_allowed)[0]
            action = actions_allowed[idx]

        else:
            raise IOError("Error: invalid policy mode!")

        return action, PNprob, prob