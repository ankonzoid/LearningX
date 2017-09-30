"""

 AgentClass.py  (author: Anson Wong / git: ankonzoid)

"""
import numpy as np
import random

class Agent:
    def __init__(self, agent_info):
        self.name = agent_info["name"]
        self.epsilon = agent_info["epsilon"]  # exploration probability

    def get_action(self, state, brain, env):
        # Explore actions
        def explore_actions_allowed(state, env):
            actions_explore_allowed = env.allowed_actions(state)
            return actions_explore_allowed
        # Choose highest value action
        def argmax_Q_actions_allowed(Q, state, env):
            actions_allowed = env.allowed_actions(state)
            Q_s = Q[state[0], state[1], actions_allowed]
            actions_Qmax_allowed = actions_allowed[np.flatnonzero(Q_s == np.max(Q_s))]
            return actions_Qmax_allowed

        # Perform epsilon-greedy selection
        if random.uniform(0, 1) < self.epsilon:
            actions_explore_allowed = explore_actions_allowed(state, env)
            return np.random.choice(actions_explore_allowed)
        else:
            actions_Qmax_allowed = argmax_Q_actions_allowed(brain.Q, state, env)
            return np.random.choice(actions_Qmax_allowed)
