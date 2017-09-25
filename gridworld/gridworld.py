"""

 gridworld.py  (author: Anson Wong / git: ankonzoid)

 Teaches an agent how to move from the top-left corner (0,0) to the 
 bottom-right corner (Ly,Lx) of a rectangular grid in the least amount 
 of steps.

 Our method uses:
  - on-policy Monte Carlo (epsilon-greedy)
  - tabular R(s,a) rewards
  - tabular Q(s,a) state-action value
  - tabular policy(s) policy

 Note that there are multiple degenerate optimal policies for this problem,
 so the specific policy found here is not unique (what is more important
 is that the greedy action chosen by the optimal policy at every non-terminal 
 state moves towards the end goal).

 Example output after a single run (Ly=6, Lx=6):

 Final policy (action indices 0, 1, 2, 3):
  [[2 1 1 2 2 2 2]
   [2 2 1 1 2 1 2]
   [1 2 1 1 2 2 2]
   [2 1 1 2 1 1 2]
   [1 1 2 2 2 1 2]
   [1 1 1 1 1 2 2]
   [1 1 1 1 1 1 3]]

 Action[action indices]:
  action[0] = [-1 0]
  action[1] = [0 1]
  action[2] = [1 0]
  action[3] = [0 -1]

"""
import numpy as np
import random
import itertools
from scipy.linalg import norm

def main():
    # ====================================================
    # Set up run parameters
    # ====================================================
    epsilon = 0.5  # exploration probability
    N_episodes = 20000  # number of episodes to generate
    R_nongoal = -1
    R_goal = 10000  # reward for finding goal (make this sufficiently large)

    # Set up convenient variables for parsing
    class State():
        def __init__(self, index, dx):
            self.index = index
            self.dx = dx

    class Action():
        def __init__(self, index, dx):
            self.index = index
            self.dx = dx

    action_space = {
        "up": Action(0, [-1, 0]),
        "right": Action(1, [0, 1]),
        "down": Action(2, [1, 0]),
        "left": Action(3, [0, -1])
    }

    env_info = {
        "Ny": 7,
        "Nx": 7,
        "action_space": action_space,
        "R_nongoal": R_nongoal,
        "R_goal": R_goal
    }

    # =====================================
    # Set up agent and environment
    # =====================================
    env = Environment(env_info)
    agent = Agent(env)

    # =====================================
    # Run many episodes to train agent
    # ======================================
    for episode in range(N_episodes):
        state = env.starting_state()  # set starting state
        trial = 0  # iterations within episode
        reward_total = 0  # total accumulated reward of episode
        while not env.is_terminal(state):
            action = agent.get_action(state, env)  # get action
            reward = env.get_reward(state, action)  # get reward
            agent.update_N(state, action)  # update our counts

            state = env.perform_action(state, action)  # update state
            reward_total += reward
            trial += 1

        # =============================
        # Update action-value Q and policy
        # =============================
        dQ = agent.update_Q()  # update Q
        dpolicy = agent.update_policy()  # update policy

        print("episode = {0}/{1}, reward_tot = {2}, iter = {3}, dQ = {4}, dpolicy = {5}".format(episode + 1, N_episodes, reward, trial, dQ, dpolicy))

    # Print final policy
    print("\nFinal policy (action indices 0, 1, 2, 3):\n")
    print(agent.policy)
    print("\nActions[action indices]:\n")
    for i, a in enumerate(env.Nactions):
        print(" action[{0}] = {1}".format(i, a))


# =========================================
#
# Side functions
#
# =========================================

class Environment:
    def __init__(self, info):
        self.Ny = info["Ny"]
        self.Nx = info["Nx"]
        self.Nactions = info["Nactions"]

        self.reward = self.define_rewards(info)
        self.R_goal = info["R_goal"]
        self.R_nongoal = info["R_nongoal"]

    # Manually hard-code rewards for our environment
    def define_rewards(self, info):
        reward = self.R_nongoal * np.ones((self.Ny, self.Nx, self.Nactions), dtype=float)
        reward[self.Ny-1, self.Nx-1, 2] = self.R_goal
        reward[self.Ny-1, self.Nx-2, 1] = self.R_goal
        return reward

    def get_reward(self, state, action):
        return self.reward[state[0], state[1], action]

    def allowed_actions(self, state):
        # Allowed actions (these are global indices)
        index_actions_allowed = []
        if state[0] > 0:  # y_state > 0
            index_actions_allowed.append(0)  # can move up
        if state[0] < self.Ny-1:  # y_state < Ly
            index_actions_allowed.append(2)  # can move down
        if state[1] > 0:  # x_state > 0
            index_actions_allowed.append(3)  # can move left
        if state[1] < self.Nx-1:  # x_state < Lx
            index_actions_allowed.append(1)  # can move right
        index_actions_allowed = np.array(index_actions_allowed, dtype=np.int)
        return index_actions_allowed

    def perform_action(self, state, action):
        state_new = state + action
        return state_new

    def starting_state(self):
        return np.array([0, 0], dtype=np.int)

    def is_terminal(self, state):
        terminal_state = np.array([self.Ny - 1, self.Nx - 1], dtype=np.int
        return np.equal(state, terminal_state)_

class Agent:
    def __init__(self, env, agent_info):
        self.epsilon = agent_info["epsilon"]
        self.k =
        self.Q = np.zeros((env.n_states_y, env.n_states_x, env.n_actions), dtype=float)
        self.policy = np.zeros((env.n_states_y, env.n_states_x), dtype=np.int)

        # Cumulative stats to help with Q updates
        self.reset_N(env)

    def reset_N(self, env):
        # Cumulative stats to help with Q updates
        self.N_sa = np.zeros((env.Ny, env.Nx, env.Nactions), dtype=np.int)
        self.N_s = np.zeros((env.Ny, env.Nx), dtype=np.int)
        self.R_total = np.zeros((env.Ny, env.Nx, env.Nactions), dtype=np.float)
        self.R_avg = np.zeros((env.Ny, env.Nx, env.Nactions), dtype=np.float)

        # Find non-zero visits of the most recent episode
        self.sa_visited = np.nonzero(self.N_sa)  # find visited (state, action)
        self.sa_zipped = zip(self.sa_visited[0], self.sa_visited[1], self.sa_visited[2])
        self.s_visited = np.nonzero(self.N_s)  # find visited (state)
        self.s_zipped = zip(self.s_visited[0], self.s_visited[1])

    def update_Q(self, state_history, reward):
        Q_old = self.Q.copy()
        for i, sa in enumerate(sa_zipped):
            self.N_sa[sa] += 1
            self.R_total[sa] += reward
            self.R_avg[sa] = self.R_total[sa] / self.N_sa[sa]
            self.Q[sa] = self.R_avg[sa]
        dQ = norm(self.Q - Q_old)
        return dQ

    def update_policy(self, env):
        policy_old = self.policy.copy()
        for i, s in enumerate(s_zipped):
            # greedy policy on the subset of available actions
            index_actions_usable = env.allowed_actions(s, info)
            j = np.argmax(self.Q[s[0], s[1], index_actions_usable])
            self.policy[s[0], s[1]] = index_actions_usable[j]
        dpolicy = norm(self.policy - policy_old)
        return dpolicy

    def update_N(self, state, action):
        self.N_sa[state[0], state[1], action] += 1
        self.N_s[state[0], state[1]] = 1

    def randomize_policy(self, env):
        for state in list(itertools.product(range(env.n_states_y), range(env.n_states_x))):
            index_actions_allowed = env.allowed_actions(state)
            self.policy[state[0], state[1]] = \
                random.choice(index_actions_allowed)  # choose random allowed

    # Under a epsilon-greedy policy, we find the action to be take be a state
    def get_action(self, state, env):
        idx_actions = env.allowed_actions(state)
        rand = random.uniform(0, 1)
        if rand < self.epsilon:
            idx_action = random.choice(idx_actions.tolist())
        else:
            idx_action = np.argmax(self.Q[state[0], state[1], :])
        return idx_action


# Driver 
if __name__ == '__main__':
    main()