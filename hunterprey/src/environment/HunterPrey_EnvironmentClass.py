"""

 EnvironmentClass.py  (author: Anson Wong / git: ankonzoid)

"""
import numpy as np
import random

class Environment:
    def __init__(self, env_info):
        self.name = "HunterPrey"

        # Read environment info
        self.Ny = env_info["Ny"]  # y-grid size
        self.Nx = env_info["Nx"]  # x-grid size
        self.N_agents = env_info["N_agents"]

        # State and Action space
        self.action_dict = {"up": 0, "right": 1, "down": 2, "left": 3}
        self.action_coords = np.array([[-1,0], [0,1], [1,0], [0,-1]], dtype=np.int)
        self.N_actions = len(self.action_dict.keys())

        self.state_dim = (self.Ny, self.Nx) * self.N_agents
        self.action_dim = (self.N_actions,) * self.N_agents
        self.state_action_dim = self.state_dim + self.action_dim

        # Terminal state
        self.state_terminal = np.array([self.Ny-1, self.Nx-1], dtype=np.int)

        # Rewards
        self.reward = self.define_rewards()

        # Make checks
        if self.N_actions != len(self.action_coords):
            raise IOError("Inconsistent actions given")

    # ========================
    # Rewards
    # ========================
    def _find_sa_to_terminal(self):
        state_terminal = self.state_terminal
        sa_candidate_list = []
        for action in range(self.N_actions):
            state_candidate = np.array(state_terminal, dtype=np.int) - self.action_coords[action]
            if self.is_allowed_state(state_candidate):
                sa = tuple(state_candidate) + (action,)
                sa_candidate_list.append(sa)
        return sa_candidate_list

    def define_rewards(self):
        R_goal = 100  # reward for arriving at a goal state
        R_nongoal = -0.1  # reward for arriving at a non-goal state
        reward = R_nongoal * np.ones(self.state_action_dim, dtype=np.float)
        # Set R_goal for all (s,a) that lead to terminal state
        sa_to_goal_list = self._find_sa_to_terminal()
        for sa in sa_to_goal_list:
            reward[sa] = R_goal
        return reward

    def get_reward(self, state, action):
        sa = tuple(list(state) + [action])
        return self.reward[sa]

    # ========================
    # Action restrictions
    # ========================
    def allowed_actions(self, state):
        actions = []
        for action in range(self.N_actions):
            state_query = self.perform_action(state, action)
            if self.is_allowed_state(state_query):
                actions.append(action)
        actions = np.array(actions, dtype=np.int)
        return actions

    def is_allowed_state(self, state):
        y = state[0]
        x = state[1]
        if (y >= 0) and (y < self.Ny) and (x >= 0) and (x < self.Nx):
            return True
        else:
            return False

    # ========================
    # Environment Details
    # ========================
    def starting_state(self):
        starting_state = np.array([0, 0], dtype=np.int)
        return starting_state

    def random_state(self):
        state_random = np.array([random.randint(0, self.Ny-1), random.randint(0, self.Nx-1)], dtype=np.int)
        while self.is_terminal(state_random):
            state_random = np.array([random.randint(0, self.Ny - 1), random.randint(0, self.Nx - 1)], dtype=np.int)
        return state_random

    def is_terminal(self, state):
        return np.array_equal(state, self.state_terminal)

    # ========================
    # Action utilities
    # ========================
    def perform_action(self, state, action):
        return np.add(state, self.action_coords[action])