"""

 hunterprey.py  (author: Anson Wong / git: ankonzoid)

 Trains a hunter agent to capture a prey agent on an (Ny, Nx) grid.

"""
import numpy as np
import random, itertools, operator, sys, os

sys.path.append("./src")
sys.path.append("./src/agent")
sys.path.append("./src/brain")
sys.path.append("./src/environment")
sys.path.append("./src/memory")
from EpsilonGreedy_AgentClass import Agent
from SampleAveraging_BrainClass import Brain
#from QLearning_BrainClass import Brain
from HunterPrey_EnvironmentClass import Environment
from MemoryClass import Memory
import utils

def main():
    # =========================
    # Settings
    # =========================
    N_episodes = 1000000
    agent_info = {"name": "hunter", "epsilon": 0.5}
    env_info = {"Ny_global": 7, "Nx_global": 7}
    brain_info = {"learning_rate": 0.8, "discount": 0.9}  # only relevant for Q-learning

    # =========================
    # Set up environment, agent, memory and brain
    # =========================
    agent = Agent(agent_info)
    env = Environment(env_info)
    brain = Brain(env, brain_info)
    memory = Memory(env)

    # =========================
    # Train agent
    # =========================
    print("\nTraining '{}' agent on '{}' environment for {} episodes (epsilon = {})...\n".format(agent.name, env.name, N_episodes, agent.epsilon))

    memory.reset_run_counters()  # reset run counters once only
    for episode in range(N_episodes):
        memory.reset_episode_counters()  # reset episodic counters

        # state = position of hunter relative to prey (want to get to [0,0])
        # state_global = global position of hunter
        # state_target_global = global position of prey
        (state, state_global, state_target_global) = env.get_random_state()
        env.set_state_terminal_global(state_target_global)
        state_start_global = state_global

        state_global_history = [state_global]
        while not env.is_terminal(state):  # NOTE: terminates when local prey coordinates hit (0,0)
            # Get action from policy
            action = agent.get_action(state, brain, env)  # get action from policy
            # Collect reward from environment
            reward = env.get_reward(state, action)  # get reward
            # Update episode counters
            memory.update_episode_counters(state, action, reward)  # update our episodic counters
            # Compute and observe next state
            state_next = env.perform_action(state, action)
            state_global_next = env.perform_action_global(state_global, action)
            # Update Q after episode (if needed)
            if "update_Q_during_episode" in utils.method_list(Brain):
                brain.update_Q_during_episode(state, action, state_next, reward)
            # Transition to next state
            state = state_next
            state_global = state_global_next
            state_global_history.append(state_global)

        # Update run counters first (before updating Q)
        memory.update_run_counters()  # use episode counters to update run counters

        # Update Q after episode (if needed)
        if "update_Q_after_episode" in utils.method_list(Brain):
            brain.update_Q_after_episode(memory)

        # Print
        if (episode + 1) % (N_episodes / 20) == 0:
            n_optimal = np.abs(env.ygrid_global[state_start_global[0]] - env.ygrid_global[state_target_global[0]]) + np.abs(env.xgrid_global[state_start_global[1]] - env.xgrid_global[state_target_global[1]])

            print(" [train episode = {}/{}] total reward = {:.1F}, n_actions = {}, n_optimal = {}".format(episode + 1, N_episodes, memory.R_total_episode, memory.N_actions_episode, n_optimal))
            print(" grid goal: [{},{}] -> [{},{}]".format(env.ygrid_global[state_start_global[0]], env.xgrid_global[state_start_global[1]], env.ygrid_global[state_target_global[0]], env.xgrid_global[state_target_global[1]]))
            grid_path_str = " grid path: "
            for i, s in enumerate(state_global_history):
                grid_path_str += "[{},{}]".format(env.ygrid_global[s[0]], env.xgrid_global[s[1]])
                if i < len(state_global_history) - 1:
                    grid_path_str += " -> "
            print("{}\n".format(grid_path_str))


    # Test our policy

    # =======================
    # Print final policy
    # =======================
    #print("\nFinal policy (hunter):\n")
    #print(brain.compute_policy(env))
    #print("")
    #for (key, val) in sorted(env.action_dict.items(), key=operator.itemgetter(1)):
    #    print(" action['{}'] = {}".format(key, val))


# Driver
if __name__ == '__main__':
    main()