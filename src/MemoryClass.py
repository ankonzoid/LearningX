"""

 MemoryClass.py  (author: Anson Wong / git: ankonzoid)

"""
import numpy as np

class Memory:
    def __init__(self, env):
        self.state_action_dim = env.state_action_dim  # (s,a) dimension
        self.state_dim = env.state_dim  # (s) dimension

        # Run memory
        self.reset_run_counters()

        # Episode memory
        self.reset_episode_counters()

    # ========================
    # Run memory counters
    # ========================
    def reset_run_counters(self):
        self.k_state_action_run = np.zeros(self.state_action_dim, dtype=np.int)  # counts # of first occurrence-per-episodes (s,a) pairs
        self.R_state_action_run = np.zeros(self.state_action_dim, dtype=np.float)  # counts total reward of first occurrence-per-episodes (s,a) pairs

    def update_run_counters(self):
        state_action_episode_unique = list(set(self.state_action_history_episode))
        for sa in state_action_episode_unique:
            self.k_state_action_run[sa] += 1
            self.R_state_action_run[sa] += self.R_total_episode

    # ========================
    # Episode memory counters
    # ========================
    def reset_episode_counters(self):
        self.N_actions_episode = 0  # counts total # of actions in episode (scalar)
        self.R_total_episode = 0.0  # counts total reward collected in episode (scalar)
        self.N_state_action_episode = np.zeros(self.state_action_dim, dtype=np.int)  # counts total # of (s,a) pairs in episode
        self.N_states_episode = np.zeros(self.state_dim, dtype=np.int)  # counts total # of (s) in episode
        self.R_state_action_episode = np.zeros(self.state_action_dim, dtype=np.float)  # sum R(s,a) pairs in episode
        self.state_action_history_episode = []  # list of (s,a) tuples (not unique)
        self.state_history_episode = []  # list of tuples (s) tuples (not unique)

    def update_episode_counters(self, state, action, reward):
        sa = tuple(list(state) + [action])
        s = tuple(list(state))
        self.N_actions_episode += 1
        self.R_total_episode += reward
        self.N_state_action_episode[sa] += 1
        self.N_states_episode[s] += 1
        self.R_state_action_episode[sa] += reward
        self.state_action_history_episode.append(sa)
        self.state_history_episode.append(s)