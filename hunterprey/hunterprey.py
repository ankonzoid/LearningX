"""

 hunterprey.py  (author: Anson Wong / git: ankonzoid)

 Trains a hunter agent to capture a prey agent on a (Ny, Nx) grid.

"""
import numpy as np
import random
import itertools
import operator

def main():
    # =========================
    # Settings
    # =========================
    N_episodes = 50000  # specify number of training episodes
    env_info = {"Ny": 7, "Nx": 7}
    agent_info = {"epsilon": 0.5}

    # =========================
    # Train agent
    # =========================
    env = Environment(env_info)  # set up environment
    agent = Agent(env, agent_info)  # set up agent

    agent.reset_run_counters()  # reset run counters once only
    print("\nTraining epsilon-greedy agent on GridWorld for {} episodes (epsilon = {})...\n".format(N_episodes, agent.epsilon))
    for episode in range(N_episodes):
        agent.reset_episode_counters()  # reset episodic counters

        state = env.starting_state()  # starting state
        while not env.is_terminal(state):
            # Get action from policy, and collect reward from environment
            action = agent.get_action(state, env)  # get action from policy
            reward = env.get_reward(state, action)  # get reward
            # Update episode counters, and transition to next state
            agent.update_episode_counters(state, action, reward)  # update our episodic counters
            state = env.perform_action(state, action)  # observe next state

        # Update run counters first (before updating Q)
        agent.update_run_counters()
        # Update Q
        dQsum = agent.update_Q()

        # Print
        if (episode+1) % (N_episodes/20) == 0:
            print(" episode = {}/{}, reward = {:.1F}, n_actions = {}, dQsum = {:.2E}".format(
                episode + 1, N_episodes, agent.R_total_episode, agent.N_actions_episode, dQsum))

    # =======================
    # Print results
    # =======================
    print("\nFinal policy:\n")
    print(agent.compute_policy(env))
    print("")
    for (key, val) in sorted(env.action_dict.items(), key=operator.itemgetter(1)):
        print(" action['{}'] = {}".format(key, val))


# ======================
#
# Environment Class
#
# ======================
class Environment:
    def __init__(self, agent_info):
        # State space
        self.Ny = agent_info["Ny"]  # y-grid size
        self.Nx = agent_info["Nx"]  # x-grid size

        # Action space
        self.action_dict = {"up": 0, "right": 1, "down": 2, "left": 3}
        self.action_coords = np.array([[-1,0], [0,1], [1,0], [0,-1]], dtype=np.int)
        self.Nactions = len(self.action_dict.keys())
        self.state_action_dim = (self.Ny, self.Nx, self.Nactions)
        self.state_dim = (self.Ny, self.Nx)

        # Rewards
        self.reward = self.define_rewards()

        # Make checks
        if self.Nactions != len(self.action_coords):
            raise IOError("Inconsistent actions given")

    # ========================
    # Rewards
    # ========================
    def define_rewards(self):
        R_goal = 100  # reward for arriving at a goal state
        R_nongoal = -0.1  # reward for arriving at a non-goal state
        reward = R_nongoal * np.ones(self.state_action_dim, dtype=np.float)
        reward[self.Ny-2, self.Nx-1, self.action_dict["down"]] = R_goal
        reward[self.Ny-1, self.Nx-2, self.action_dict["right"]] = R_goal
        return reward

    def get_reward(self, state, action):
        sa = tuple(list(state) + [action])
        return self.reward[sa]

    # ========================
    # Action restrictions
    # ========================
    def allowed_actions(self, state):
        actions = []
        y = state[0]
        x = state[1]
        if (y > 0): actions.append(self.action_dict["up"])
        if (y < self.Ny-1): actions.append(self.action_dict["down"])
        if (x > 0): actions.append(self.action_dict["left"])
        if (x < self.Nx-1): actions.append(self.action_dict["right"])
        actions = np.array(actions, dtype=np.int)
        return actions

    # ========================
    # Environment Details
    # ========================
    def starting_state(self):
        return np.array([0, 0], dtype=np.int)

    def is_terminal(self, state):
        return np.array_equal(state, np.array([self.Ny-1, self.Nx-1], dtype=np.int))

    # ========================
    # Action utilities
    # ========================
    def perform_action(self, state, action):
        return np.add(state, self.action_coords[action])


# ======================
#
# Agent Class
#
# ======================
class Agent:
    def __init__(self, env, env_info):
        # Agent settings
        self.epsilon = env_info["epsilon"]  # exploration probability

        # Q state-action value
        self.Q = np.zeros(env.state_action_dim, dtype=np.float)

        # Helper variables
        self.state_action_dim = env.state_action_dim
        self.state_dim = env.state_dim

    # ========================
    # Counters
    # ========================
    def reset_run_counters(self):
        self.k_state_action_run = np.zeros(self.state_action_dim, dtype=np.int)
        self.R_state_action_run = np.zeros(self.state_action_dim, dtype=np.float)

    def update_run_counters(self):
        state_action_episode_unique = list(set(self.state_action_history_episode))
        for sa in state_action_episode_unique:
            self.k_state_action_run[sa] += 1
            self.R_state_action_run[sa] += self.R_total_episode

    def reset_episode_counters(self):
        self.N_actions_episode = 0
        self.R_total_episode = 0.0
        self.N_state_action_episode = np.zeros(self.state_action_dim, dtype=np.int)
        self.N_state_episode = np.zeros(self.state_dim, dtype=np.int)
        self.R_state_action_episode = np.zeros(self.state_action_dim, dtype=np.float)
        self.state_action_history_episode = []  # list of tuples
        self.state_history_episode = []  # list of tuples

    def update_episode_counters(self, state, action, reward):
        sa = tuple(list(state) + [action])
        s = tuple(list(state))
        self.N_actions_episode += 1
        self.R_total_episode += reward
        self.N_state_action_episode[sa] += 1
        self.N_state_episode[s] += 1
        self.R_state_action_episode[sa] += reward
        self.state_action_history_episode.append(sa)  # list of tuples
        self.state_history_episode.append(s)  # list of tuples

    # ========================
    # Q(s,a) state-action values
    # ========================
    def update_Q(self):
        state_action_episode_unique = list(set(self.state_action_history_episode))
        dQsum = 0.0
        for sa in state_action_episode_unique:  # state_action_history
            k_sa = self.k_state_action_run[sa]
            reward_total_sa = self.R_total_episode
            dQ = (reward_total_sa - self.Q[sa]) / (k_sa)  # assumes k counter already updated
            self.Q[sa] += dQ
            dQsum += np.abs(dQ)
        return dQsum

    def argmax_Q_actions_allowed(self, state, env):
        actions_allowed = env.allowed_actions(state)
        Q_s = self.Q[state[0], state[1], actions_allowed]
        actions_Qmax_allowed = actions_allowed[np.flatnonzero(Q_s == np.max(Q_s))]
        return actions_Qmax_allowed

    def explore_actions_allowed(self, state, env):
        actions_explore_allowed = env.allowed_actions(state)
        return actions_explore_allowed

    # Pick up action using epsilon-greedy agent
    def get_action(self, state, env):
        if random.uniform(0, 1) < self.epsilon:
            actions_explore_allowed = self.explore_actions_allowed(state, env)
            return np.random.choice(actions_explore_allowed)
        else:
            actions_Qmax_allowed = self.argmax_Q_actions_allowed(state, env)
            return np.random.choice(actions_Qmax_allowed)

    # ========================
    # Policy
    # ========================
    def compute_policy(self, env):
        Ny = self.Q.shape[0]
        Nx = self.Q.shape[1]
        policy = np.zeros((Ny, Nx), dtype=int)
        for state in list(itertools.product(range(Ny), range(Nx))):
            actions_Qmax_allowed = self.argmax_Q_actions_allowed(state, env)
            policy[state[0], state[1]] = random.choice(actions_Qmax_allowed)  # choose random allowed
        return policy

# Driver 
if __name__ == '__main__':
    main()