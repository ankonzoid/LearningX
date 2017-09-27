"""

 gridworld.py  (author: Anson Wong / git: ankonzoid)

 Trains an agent to move from (0, 0) to (Ny-1, Nx-1) on a rectangular grid
 in the least number of grid steps. The approach taken here is an on-policy
 Monte Carlo reward-average sampling on an epsilon-greedy agent. Also,
 tabular version of the rewards R(s,a), state-action value Q(s,a), and
 policy policy(s) are used.

 Note: the optimal policy exists but is a highly degenerate solution because
 of the multitude of ways one can traverse down the grid in the minimum
 number of steps. Therefore, what is more important is that the policy
 at every non-terminal state is moving in the direction of the goal
 i.e. every action is either 1 (move right) or 2 (move down).

 Here is an example output of this code

 Final policy:

  [[2 1 1 2 2 2 2]
   [2 2 1 1 2 1 2]
   [1 2 1 1 2 2 2]
   [2 1 1 2 1 1 2]
   [1 1 2 2 2 1 2]
   [1 1 1 1 1 2 2]
   [1 1 1 1 1 1 3]]

  action['up'] = 0
  action['right'] = 1
  action['down'] = 2
  action['left'] = 3

"""
import numpy as np
import random
import itertools
import operator

def main():
    # =========================
    # Settings
    # =========================
    N_episodes = 100000  # specify number of training episodes
    env_info = {"Ny": 7, "Nx": 7}
    agent_info = {"epsilon": 0.5}

    # =========================
    # Set up environment, agent, memory and brain
    # =========================
    env = Environment(env_info)  # set up environment rewards and state-transition rules
    agent = Agent(agent_info)  # set up epsilon-greedy agent
    brain = Brain(env)  # stores and updates Q(s,a) and policy(s)
    memory = Memory(env)  # keeps track of run and episode (s,a) histories

    # =========================
    # Train agent
    # =========================
    print("\nTraining {} agent on {} environment for {} episodes (epsilon = {})...\n".format(agent.name, env.name, N_episodes, agent.epsilon))

    memory.reset_run_counters()  # reset run counters once only
    for episode in range(N_episodes):
        memory.reset_episode_counters()  # reset episodic counters
        state = env.starting_state()  # starting state
        while not env.is_terminal(state):
            # Get action from policy, and collect reward from environment
            action = agent.get_action(state, brain, env)  # get action from policy
            reward = env.get_reward(state, action)  # get reward
            # Update episode counters, and transition to next state
            memory.update_episode_counters(state, action, reward)  # update our episodic counters
            state = env.perform_action(state, action)  # observe next state

        # Update run counters first (before updating Q)
        memory.update_run_counters()
        # Update Q
        dQsum = brain.update_Q(memory)

        # Print
        if (episode+1) % (N_episodes/20) == 0:
            print(" episode = {}/{}, reward = {:.1F}, n_actions = {}, dQsum = {:.2E}".format(episode + 1, N_episodes, memory.R_total_episode, memory.N_actions_episode, dQsum))

    # =======================
    # Print results
    # =======================
    print("\nFinal policy:\n")
    print(brain.compute_policy(env))
    print("")
    for (key, val) in sorted(env.action_dict.items(), key=operator.itemgetter(1)):
        print(" action['{}'] = {}".format(key, val))


# ======================
#
# Environment Class
#
# ======================
class Environment:
    def __init__(self, env_info):
        self.name = "GridWorld"

        # State space
        self.Ny = env_info["Ny"]  # y-grid size
        self.Nx = env_info["Nx"]  # x-grid size

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
    def __init__(self, agent_info):
        self.name = "epsilon-greedy"
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

        # Perform epsilone-greedy selection
        if random.uniform(0, 1) < self.epsilon:
            actions_explore_allowed = explore_actions_allowed(state, env)
            return np.random.choice(actions_explore_allowed)
        else:
            actions_Qmax_allowed = argmax_Q_actions_allowed(brain.Q, state, env)
            return np.random.choice(actions_Qmax_allowed)


# ======================
#
# Brain Class
#
# ======================
class Brain:
    def __init__(self, env):
        self.Q = np.zeros(env.state_action_dim, dtype=np.float)  # Q state-action value
        self.state_action_dim = env.state_action_dim
        self.state_dim = env.state_dim

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


# ======================
#
# Memory Class
#
# ======================
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


# Driver
if __name__ == '__main__':
    main()