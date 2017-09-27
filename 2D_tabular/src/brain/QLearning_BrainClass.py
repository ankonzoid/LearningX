"""

 BrainClass.py  (author: Anson Wong / git: ankonzoid)

"""
import numpy as np
import random, itertools

class Brain:
    def __init__(self, env, brain_info):
        self.Q = np.zeros(env.state_action_dim, dtype=np.float)  # Q state-action value
        self.state_action_dim = env.state_action_dim
        self.state_dim = env.state_dim

        self.learning_rate = brain_info["learning_rate"]  # beta
        self.discount = brain_info["discount"]  # gamma

    # ========================
    # Q(s,a) state-action values
    # ========================
    def update_Q(self, memory):
        state_action_episode_unique = list(set(memory.state_action_history_episode))
        dQsum = 0.0
        for sa in state_action_episode_unique:  # state-action history
            k_sa = memory.k_state_action_run[sa]
            reward_total_sa = memory.R_total_episode
            dQ = (reward_total_sa - self.Q[sa]) / (k_sa)  # assumes k counter already updated
            self.Q[sa] += dQ
            dQsum += np.abs(dQ)
        return dQsum

    # ========================
    # Policy
    # ========================
    def compute_policy(self, env):
        # Choose highest value action
        def argmax_Q_actions_allowed(Q, state, env):
            actions_allowed = env.allowed_actions(state)
            Q_s = Q[state[0], state[1], actions_allowed]
            actions_Qmax_allowed = actions_allowed[np.flatnonzero(Q_s == np.max(Q_s))]
            return actions_Qmax_allowed

        Ny = self.Q.shape[0]
        Nx = self.Q.shape[1]
        policy = np.zeros((Ny, Nx), dtype=int)
        for state in list(itertools.product(range(Ny), range(Nx))):
            actions_Qmax_allowed = argmax_Q_actions_allowed(self.Q, state, env)
            policy[state[0], state[1]] = random.choice(actions_Qmax_allowed)  # choose random allowed
        return policy