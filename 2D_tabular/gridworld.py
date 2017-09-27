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
import operator, sys

sys.path.append("./src")
sys.path.append("./src/agent")
sys.path.append("./src/brain")
sys.path.append("./src/environment")
sys.path.append("./src/memory")
from EpsilonGreedy_AgentClass import Agent
from SampleAveraging_BrainClass import Brain
from GridWorld_EnvironmentClass import Environment
from MemoryClass import Memory
import utils

def main():
    # =========================
    # Settings
    # =========================
    N_episodes = 100000  # specify number of training episodes
    env_info = {"Ny": 7, "Nx": 7}
    agent_info = {"name": "epsilon-greedy", "epsilon": 0.5}
    brain_info = {"learning_rate": 1.0, "discount": 1.0}  # only relevant for Q-learning

    # =========================
    # Set up environment, agent, memory and brain
    # =========================
    env = Environment(env_info)  # set up environment rewards and state-transition rules
    agent = Agent(agent_info)  # set up epsilon-greedy agent
    brain = Brain(env, brain_info)  # stores and updates Q(s,a) and policy(s)
    memory = Memory(env)  # keeps track of run and episode (s,a) histories

    # =========================
    # Train agent
    # =========================
    print("\nTraining '{}' agent on '{}' environment for {} episodes (epsilon = {})...\n".format(agent.name, env.name, N_episodes, agent.epsilon))

    memory.reset_run_counters()  # reset run counters once only
    for episode in range(N_episodes):
        memory.reset_episode_counters()  # reset episodic counters
        state = env.starting_state()  # starting state
        while not env.is_terminal(state):
            # Get action from policy, and collect reward from environment
            action = agent.get_action(state, brain, env)  # get action from policy
            reward = env.get_reward(state, action)  # get reward
            # Update episode counters
            memory.update_episode_counters(state, action, reward)  # update our episodic counters
            # Compute next state
            state_next = env.perform_action(state, action)  # observe next state
            # Update Q during episode (if needed)
            if "update_Q_during_episode" in utils.method_list(Brain):
                brain.update_Q_during_episode(memory)
            # Transition to next state
            state = state_next

        # Update run counters first (before updating Q)
        memory.update_run_counters()  # use episode counters to update run counters

        # Update Q after episode (if needed)
        dQsum = -1
        if "update_Q_after_episode" in utils.method_list(Brain):
            dQsum = brain.update_Q_after_episode(memory)

        # Print
        if (episode+1) % (N_episodes/20) == 0:
            print(" episode = {}/{}, reward = {:.1F}, n_actions = {}, dQsum = {:.2E}".format(episode + 1, N_episodes, memory.R_total_episode, memory.N_actions_episode, dQsum))

    # =======================
    # Print final policy
    # =======================
    print("\nFinal policy:\n")
    print(brain.compute_policy(env))
    print("")
    for (key, val) in sorted(env.action_dict.items(), key=operator.itemgetter(1)):
        print(" action['{}'] = {}".format(key, val))


# Driver
if __name__ == '__main__':
    main()