"""

 EnvironmentClass.py  (author: Anson Wong / git: ankonzoid)

"""
import numpy as np
import random

class Environment:
    def __init__(self, env_info):
        self.name = "HunterPrey"

        # Read environment info
        self.Ny_global = env_info["Ny"]  # global y-grid size
        self.Nx_global = env_info["Nx"]  # global x-grid size
        self.Ny = 2*self.Ny_global - 1 # relative y-grid size
        self.Nx = 2*self.Nx_global - 1 # relative x-grid size

        # Set up grid
        self.ygrid_global = np.array(list(range(0, self.Ny_global)), dtype=np.int)
        self.ygrid = np.array(list(range(-(self.Ny_global-1), self.Ny_global)), dtype=np.int)
        self.xgrid_global = np.array(list(range(0, self.Nx_global)), dtype=np.int)
        self.xgrid = np.array(list(range(-(self.Nx_global-1), self.Nx_global)), dtype=np.int)

        # State and Action space
        self.action_dict = {"up": 0, "right": 1, "down": 2, "left": 3}
        self.action_coords = np.array([[-1,0], [0,1], [1,0], [0,-1]], dtype=np.int)
        self.N_actions = len(self.action_dict.keys())

        self.state_dim = (self.Ny, self.Nx)
        self.action_dim = (self.N_actions,)
        self.state_action_dim = self.state_dim + self.action_dim

        # Terminal state (local)
        self.state_terminal = np.array([self.Ny_global-1, self.Nx_global-1], dtype=np.int)  # local grid coord [0,0]
        self.state_terminal_global = None

        # Rewards
        self.reward = self.define_rewards()

        # Make checks
        if (len(self.ygrid_global) != self.Ny_global) or (len(self.xgrid_global) != self.Nx_global) or \
            (len(self.ygrid) != self.Ny) or (len(self.ygrid) != self.Ny):
            raise IOError("Error: Inconsistent grid given!")

        if self.N_actions != len(self.action_coords):
            raise IOError("Error: Inconsistent actions given!")

        if self.ygrid[self.state_terminal[0]]!=0 or self.xgrid[self.state_terminal[1]]!=0:
            raise IOError("Error: Unexpected terminal state")


    # ========================
    # Rewards
    # ========================

    def define_rewards(self):

        def find_sa_to_terminal(env):
            state_terminal = env.state_terminal
            sa_candidate_list = []
            #for action in range(env.N_actions):
            #    state_candidate = np.array(state_terminal, dtype=np.int) - env.action_coords[action]
            #    if env.is_allowed_state(state_candidate):
            #        sa = tuple(state_candidate) + (action,)
            #        sa_candidate_list.append(sa)
            for action in range(env.N_actions):
                state_candidate = np.array(state_terminal, dtype=np.int) - env.action_coords[action]
                sa = tuple(state_candidate) + (action,)
                sa_candidate_list.append(sa)
            return sa_candidate_list

        R_goal = 100  # reward for arriving at a goal state
        R_nongoal = -0.1  # reward for arriving at a non-goal state
        reward = R_nongoal * np.ones(self.state_action_dim, dtype=np.float)
        # Set R_goal for all (s,a) that lead to terminal state
        for sa in find_sa_to_terminal(self):
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
        if len(actions) == 0:
            raise IOError("Error: agent is in a state where no actions are allowed")
        return actions

    def is_allowed_state(self, state):  # does not use state (but should include it)

        state_y_range = np.array(
            [- self.state_terminal_global[0],
             - self.state_terminal_global[0] + (self.Ny_global - 1)], dtype=np.int)
        state_y_range += (self.Ny_global - 1)
        state_x_range = np.array(
            [- self.state_terminal_global[1],
             - self.state_terminal_global[1] + (self.Nx_global - 1)], dtype=np.int)
        state_x_range += (self.Nx_global - 1)

        #print("{}: yrange={} xrange={}".format(state, state_y_range, state_x_range))

        if (state[0] >= state_y_range[0]) and (state[0] <= state_y_range[1]) and \
            (state[1] >= state_x_range[0]) and (state[1] <= state_x_range[1]):
            return True
        else:
            return False

    # ========================
    # Environment Details
    # ========================

    def set_state_terminal_global(self, state_terminal_global):
        self.state_terminal_global = state_terminal_global

    def get_random_state(self):

        def random_state_global():
            state_random_global = np.array([random.randint(0, self.Ny_global-1),
                                            random.randint(0, self.Nx_global-1)], dtype=np.int)
            return state_random_global

        def random_state_global_prey(state_global):
            state_target_random_global = random_state_global()
            while np.array_equal(state_target_random_global, state_global):
                state_target_random_global = random_state_global()
            return state_target_random_global

        # Choose random state and target state, the compute relative state
        state_global = random_state_global()
        state_target_global = random_state_global_prey(state_global)

        dy = state_global[0] - state_target_global[0]
        dx = state_global[1] - state_target_global[1]

        # Retrieve state based on this grid state
        state_y = dy + (self.Ny_global-1)
        state_x = dx + (self.Nx_global-1)
        if state_y < 0 or state_y >= self.Ny:
            raise IOError("state_y is not in proper range")
        if state_x < 0 or state_x >= self.Nx:
            raise IOError("state_x is not in proper range")
        state = np.array([state_y, state_x], dtype=np.int)

        return (state, state_global, state_target_global)

    def is_terminal(self, state):
        return np.array_equal(state, self.state_terminal)

    # ========================
    # Action utilities
    # ========================
    def perform_action(self, state, action):
        state_next = np.add(state, self.action_coords[action])
        return state_next

    def perform_action_global(self, state_global, action):
        state_next_global = np.add(state_global, self.action_coords[action])
        return state_next_global