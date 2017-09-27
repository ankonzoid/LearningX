"""

 hunterprey.py  (author: Anson Wong / git: ankonzoid)

 Trains a hunter agent to capture a prey agent on a (Ny, Nx) grid.

"""
import numpy as np
import random, itertools, operator, sys, os

sys.path.append("./src/agent")
sys.path.append("./src/brain")
sys.path.append("./src/environment")
sys.path.append("./src/memory")
from EpsilonGreedy_AgentClass import Agent
from SampleAveraging_BrainClass import Brain
from HunterPrey_EnvironmentClass import Environment
from MemoryClass import Memory

def main():
    # =========================
    # Settings
    # =========================
    N_episodes = 100000
    hunter_agent_info = {"name": "hunter", "epsilon": 0.2}
    prey_agent_info = {"name": "prey", "epsilon": 1.0}
    env_info = {"Ny": 7, "Nx": 7}

    # =========================
    # Set up environment, agent, memory and brain
    # =========================
    agent_hunter = Agent(hunter_agent_info)
    agent_prey = Agent(prey_agent_info)
    env = Environment(env_info)
    brain_hunter = Brain(env)
    brain_prey = Brain(env)
    memory_hunter = Memory(env)
    memory_prey = Memory(env)

    # =========================
    # Train agent
    # =========================
    print("\nTraining '{}' agent to hunt '{}' agent on '{}' environment for {} episodes (hunter_epsilon = {}, prey_epsilon = {})...\n".format(agent_hunter.name, agent_prey.name, env.name, N_episodes, agent_hunter.epsilon, agent_prey.epsilon))

    memory_hunter.reset_run_counters()  # reset run counters once only
    memory_prey.reset_run_counters()  # reset run counters once only
    for episode in range(N_episodes):
        memory_hunter.reset_episode_counters()  # reset episodic counters
        memory_prey.reset_episode_counters()  # reset episodic counters

        state_hunter = env.starting_state()  # starting hunter state
        state_prey = env.random_state()  # starting prey state

        while not env.is_terminal(state_hunter):
            # Get action from policy, and collect reward from environment
            action_hunter = agent_hunter.get_action(state_hunter, brain_hunter, env)  # get action from policy
            action_prey = agent_prey.get_action(state_prey, brain_prey, env)  # get action from policy

            reward_hunter = env.get_reward(state_hunter, action_hunter)  # get reward
            reward_prey = env.get_reward(state_prey, action_prey)  # get reward

            # Update episode counters
            memory_hunter.update_episode_counters(state_hunter, action_hunter, reward_hunter)  # update our episodic counters
            memory_prey.update_episode_counters(state_prey, action_prey, reward_prey)  # update our episodic counters

            # Transition to next state
            state_hunter = env.perform_action(state_hunter, action_hunter)  # observe next state
            state_prey = env.perform_action(state_prey, action_prey)  # observe next state

        # Update run counters first (before updating Q)
        memory_hunter.update_run_counters()
        memory_prey.update_run_counters()

        # Update Q
        dQsum_hunter = brain_hunter.update_Q(memory_hunter)
        dQsum_prey = brain_prey.update_Q(memory_prey)

        # Print
        if (episode + 1) % (N_episodes / 20) == 0:
            print(" hunter: episode = {}/{}, reward = {:.1F}, n_actions = {}, dQsum = {:.2E}".format(episode + 1, N_episodes, memory_hunter.R_total_episode, memory_hunter.N_actions_episode, dQsum_hunter))
            print(" prey: episode = {}/{}, reward = {:.1F}, n_actions = {}, dQsum = {:.2E}".format(episode + 1, N_episodes, memory_prey.R_total_episode, memory_prey.N_actions_episode, dQsum_prey))


    # =======================
    # Print final policy
    # =======================
    print("\nFinal policy (hunter):\n")
    print(brain_hunter.compute_policy(env))
    print("")
    for (key, val) in sorted(env.action_dict.items(), key=operator.itemgetter(1)):
        print(" action['{}'] = {}".format(key, val))

    print("\nFinal policy (prey):\n")
    print(brain_prey.compute_policy(env))
    print("")
    for (key, val) in sorted(env.action_dict.items(), key=operator.itemgetter(1)):
        print(" action['{}'] = {}".format(key, val))


# Driver
if __name__ == '__main__':
    main()