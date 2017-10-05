"""

 EnvironmentClass.py  (author: Anson Wong / git: ankonzoid)

"""
import numpy as np

class Environment:

    def __init__(self, env_info):

        # Environment settings
        self.Ny = env_info["Ny"]
        self.Nx = env_info["Nx"]

        # State and action space
        self.action_dict = {"up": 0, "right": 1, "down": 2, "left": 3}
        self.action_coords = np.array([[-1,0], [0,1], [1,0], [0,-1]], dtype=np.int)
        self.N_actions = len(self.action_coords)

        self.state_dim = (self.Ny, self.Nx)  # tuple of integers
        self.action_dim = (self.N_actions,)  # tuple of integers

        self.state_size = np.prod(np.array(list(self.state_dim), dtype=np.int))  # integer
        self.action_size = np.prod(np.array(list(self.action_dim), dtype=np.int))  # integer

        # Check
        if len(self.action_dict.keys()) != self.N_actions:
            raise IOError("Error: Inconsistent action dimensions!")

    # ===================
    # Starting and terminal state
    # ===================

    def starting_state(self):
        # 2D zero grid with a 1 at top-left corner
        state = np.zeros((self.Ny, self.Nx), dtype=np.int)
        state[0, 0] = 1
        return state

    def is_terminal_state(self, state):
        idx = np.argwhere(np.array(state) == 1)[0]
        if (idx[0], idx[1]) == (self.Ny-1, self.Nx-1):
            return True
        else:
            return False

    # ======================
    # Rewards
    # ======================

    def get_reward(self, state, action):
        reward = -0.01
        idx = np.argwhere(np.array(state) == 1)[0]
        if (idx[0] == self.Ny-2) and (idx[1] == self.Nx-1) and (action == self.action_dict["down"]):
            reward = 100
        if (idx[0] == self.Ny-1) and (idx[1] == self.Nx-2) and (action == self.action_dict["right"]):
            reward = 100
        return reward

    # ======================
    # Apply action
    # ======================

    def is_allowed_action(self, state, action):
        # Find index of agent location in 2D image
        idx_list = np.argwhere(np.array(state) == 1)
        if len(idx_list) != 1:
            raise IOError("Error: Invalid state!")
        idx_new = idx_list[0] + self.action_coords[action]
        # Check
        if (idx_new[0] < 0) or (idx_new[0] >= self.Ny):
            return False
        if (idx_new[1] < 0) or (idx_new[1] >= self.Nx):
            return False
        # If it makes it here, then it is allowed
        return True

    def perform_action(self, state, action):
        # Find index of agent location in 2D image
        idx = np.argwhere(np.array(state) == 1)
        if len(idx) != 1:
            raise IOError("Error: Invalid state!")
        idx_new = idx[0] + self.action_coords[action]
        # Check
        if not self.is_allowed_action(state, action):
            raise IOError("Trying to perform unallowed action")
        # Create new state
        state_new = np.zeros(state.shape, dtype=np.int)
        state_new[tuple(idx_new)] = 1
        return state_new