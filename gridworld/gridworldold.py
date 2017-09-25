"""

 gridworld.py  (author: Anson Wong / git: ankonzoid)

 Teaches an agent how to move from the top-left corner (0,0) to the 
 bottom-right corner (Ly,Lx) of a rectangular grid in the least amount 
 of steps.

 Our method uses:
  - on-policy Monte Carlo (epsilon-greedy)
  - tabular R(s,a) rewards
  - tabular Q(s,a) state-action value
  - tabular P(s) policy

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
    Ly = 6  # grid: state_y = 0, 1, 2, ..., Ly
    Lx = 6  # grid: state_x = 0, 1, 2, ..., Lx
    epsilon = 0.5  # exploration probability
    n_episodes = 20000  # number of episodes to generate
    R_goal = 10000  # reward for finding goal (make this sufficiently large)

    # Set actions
    # 0: up = [-1,0]
	# 1: right = [0,1]
	# 2: down = [1,0]
	# 3: left = [0,-1]
    up_move = [-1, 0]; right_move = [0, 1]; down_move = [1, 0]; left_move = [0, -1]
    actions = np.array([up_move, right_move, down_move, left_move], dtype=int)
    n_actions = len(actions)

    # Set up convenient variables for parsing
    n_states_y = Ly + 1
    n_states_x = Lx + 1
    info = {
        'Ly': Ly,
        'Lx': Lx,
        'epsilon': epsilon,
        'n_states_y': n_states_y,
        'n_states_x': n_states_x,
        'n_actions': n_actions,
        'R_goal': R_goal
    }

    # ====================================================
    # Method:
    # - on-policy Monte Carlo (epsilon-greedy)
    # - tabular R(s,a) rewards
    # - tabular Q(s,a) state-action value
    # - tabular P(s) policy
    # ====================================================
    R = set_R_rewards(info)  # rewards: (n_states_y, n_states_x, n_actions)
    Q = initialize_Q_values(info)  # value: (n_states_y, n_states_x, n_actions)
    P = initialize_P_policy(info)  # policy: (n_states_y, n_states_x)

    # Cumulative stats to help with Q update
    nvisits_sa = np.zeros((n_states_y, n_states_x, n_actions), dtype=int)
    rewards_total = np.zeros((n_states_y, n_states_x, n_actions), dtype=float)
    rewards_avg = np.zeros((n_states_y, n_states_x, n_actions), dtype=float)

    # Use policy iteration for epsilon-hard policies
    for episode in range(n_episodes):

        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        #
        # Generate a full episode following policy P
        #
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

        # Set initial and termination state
        state = np.array([0, 0], dtype=int)
        state_terminal = np.array([Ly, Lx], dtype=int)

        nvisits_sa_episode = np.zeros((n_states_y, n_states_x, n_actions), dtype=int)
        nvisits_s_episode = np.zeros((n_states_y, n_states_x), dtype=int)

        i_episode = 0  # iterations within episode
        terminate = False  # termination flag
        reward_episode = 0  # total accumulated reward of episode
        while not terminate:

            # Find action to take from epsilon policy
            index_action_take, action_str = find_action_from_policy(state, P, info)

            # Evolve state based on action chosen by policy
            action_take = actions[index_action_take]
            reward_take = R[state[0], state[1], index_action_take]
            nvisits_sa_episode[state[0], state[1], index_action_take] += 1
            nvisits_s_episode[state[0], state[1]] = 1

            # Update state and episode reward
            state += action_take  # update state with action
            reward_episode += reward_take  # update total reward of episode

            # Set termination flag
            terminate = np.array_equal(state, state_terminal)
            i_episode += 1

        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        #
        # Update action-value Q and policy P using the episode result
        #
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

        # Find non-zero visits of the most recent episode
        sa_visited_episode = np.nonzero(nvisits_sa_episode)  # find visited (state, action)
        sa_zipped = zip(sa_visited_episode[0],
                        sa_visited_episode[1],
                        sa_visited_episode[2])
        s_visited_episode = np.nonzero(nvisits_s_episode)  # find visited (state)
        s_zipped = zip(s_visited_episode[0], s_visited_episode[1])

        # Updating Q table, and compute dQ
        Q_old = Q.copy()
        for i, sa in enumerate(sa_zipped):
            nvisits_sa[sa] += 1
            rewards_total[sa] += reward_episode
            rewards_avg[sa] = rewards_total[sa] / nvisits_sa[sa]
            Q[sa] = rewards_avg[sa]
        dQ = norm(Q - Q_old)

        # Updating policy P table (using Q table), and compute dP
        P_old = P.copy()
        for i, s in enumerate(s_zipped):
            # greedy policy on the subset of available actions
            index_actions_usable = find_allowed_actions(s, info)
            j = np.argmax(Q[s[0], s[1], index_actions_usable])
            P[s[0], s[1]] = index_actions_usable[j]
        dP = norm(P - P_old)

        print("episode = {0}/{1}, reward_tot = {2}, iter = {3}, dQ = {4}, dP = {5}".format(episode + 1, n_episodes,
                                                                                           reward_episode, i_episode,
                                                                                           dQ, dP))

    # Print final policy
    print("\nFinal policy (action indices 0, 1, 2, 3):\n")
    print(P)
    print("\nActions[action indices]:\n")
    for i, a in enumerate(actions):
        print(" action[{0}] = {1}".format(i, a))


# =========================================
#
# Side functions
#
# =========================================

"""
 Finds allowed actions for a given state 
"""
def find_allowed_actions(state, info):
    # Allowed actions (these are global indices)
    Ly = info['Ly']
    Lx = info['Lx']
    index_actions_allowed = []
    if state[0] > 0:  # y_state > 0
        index_actions_allowed.append(0)  # can move up
    if state[0] < Ly:  # y_state < Ly
        index_actions_allowed.append(2)  # can move down
    if state[1] > 0:  # x_state > 0
        index_actions_allowed.append(3)  # can move left
    if state[1] < Lx:  # x_state < Lx
        index_actions_allowed.append(1)  # can move right
    index_actions_allowed = np.array(index_actions_allowed, dtype=int)
    return index_actions_allowed

"""
 Manually hard-code rewards for our environment
"""
def set_R_rewards(info):
    n_states_y = info['n_states_y']
    n_states_x = info['n_states_x']
    n_actions = info['n_actions']
    Ly = info['Ly']
    Lx = info['Lx']
    R_goal = info['R_goal']

    # Note we have no reward starting from the actual terminal goal state
    R = -1 * np.ones((n_states_y, n_states_x, n_actions), dtype=float)  # set -1 grid
    R[Ly - 1, Lx, 2] = R_goal  # strong reward to move down to goal
    R[Ly, Lx - 1, 1] = R_goal  # strong reward to move right to goal
    return R

"""
 Initialize Q-table to be zero
"""
def initialize_Q_values(info):
    n_states_y = info['n_states_y']
    n_states_x = info['n_states_x']
    n_actions = info['n_actions']
    Q = np.zeros((n_states_y, n_states_x, n_actions), dtype=float)
    return Q

"""
 Initialize policy to be be random allowed actions for each state
"""
def initialize_P_policy(info):
    n_states_y = info['n_states_y']
    n_states_x = info['n_states_x']
    P = np.zeros((n_states_y, n_states_x), dtype=int)
    for state in list(itertools.product(range(n_states_y), range(n_states_x))):
        index_actions_allowed = find_allowed_actions(state, info)
        P[state[0], state[1]] = random.choice(index_actions_allowed)  # choose random allowed
    return P

"""
 Under a epsilon-greedy policy, we find the action to be take be a state
"""
def find_action_from_policy(state, P, info):
    epsilon = info['epsilon']

    # Allowed actions to be taken by this state
    index_actions_allowed = find_allowed_actions(state, info)

    # Find the action of highest reward, or find action according to the policy
    rand = random.uniform(0, 1)
    action_str = None
    if 1:
        index_action_greedy = P[state[0], state[1]]
        if index_action_greedy not in index_actions_allowed:
            raise Exception("Invalid action taken by greedy policy!")
        # Follow policy
        if rand < epsilon:
            action_str = 'policy-explore'
            index_actions_nongreedy = index_actions_allowed.tolist()
            index_actions_nongreedy.remove(index_action_greedy)
            index_action_take = random.choice(index_actions_nongreedy)
        else:
            action_str = 'policy-greedy'
            index_action_take = index_action_greedy
    return index_action_take, action_str


main()

# Driver 
if __name__ == 'main':
    main()
