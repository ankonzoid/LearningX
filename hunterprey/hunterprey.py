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
    N_episodes = 100000
    agent_info = {"name": "hunter", "epsilon": 0.5}
    env_info = {"Ny": 7, "Nx": 7}
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

        # state = position of hunter relative to prey
        # state_global = global position of hunter
        # -> global position of prey = state_global - state
        (state, state_global) = env.get_random_state()

        while not env.is_terminal(state):  # NOTE: terminates when local prey coordinates hit (0,0)
            # Get action from policy
            action = agent.get_action(state, state_global, brain, env)  # get action from policy
            # Collect reward from environment
            reward = env.get_reward(state, action)  # get reward
            # Update episode counters
            memory.update_episode_counters(state, action, reward)  # update our episodic counters
            # Compute and observe next state
            (state_next, state_global_next) = env.perform_action(state, state_global, action)
            # Update Q after episode (if needed)
            if "update_Q_during_episode" in utils.method_list(Brain):
                brain.update_Q_during_episode(state, action, state_next, reward)
            # Transition to next state
            state = state_next
            state_global = state_global_next

        # Update run counters first (before updating Q)
        memory.update_run_counters()  # use episode counters to update run counters

        # Update Q after episode (if needed)
        if "update_Q_after_episode" in utils.method_list(Brain):
            brain.update_Q_after_episode(memory)

        # Print
        if (episode + 1) % (N_episodes / 20) == 0:
            print(" hunter: episode = {}/{}, reward = {:.1F}, n_actions = {}".format(episode + 1, N_episodes, memory.R_total_episode, memory.N_actions_episode))


    # =======================
    # Print final policy
    # =======================
    print("\nFinal policy (hunter):\n")
    print(brain.compute_policy(env))
    print("")
    for (key, val) in sorted(env.action_dict.items(), key=operator.itemgetter(1)):
        print(" action['{}'] = {}".format(key, val))


# Driver
if __name__ == '__main__':
    main()