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
from Plain2D_EnvironmentClass import Environment
from MemoryClass import Memory

def main():
    print("Hunter-Prey world")

    # Settings
    hunter_agent_info = {"name": "hunter", "epsilon": 0.2}
    prey_agent_info = {"name": "prey", "epsilon": 1.0}
    env_info = {"Ny": 7, "Nx": 7}

    # Set up agent and environment
    hunter_agent = Agent(hunter_agent_info)
    prey_agent = Agent(prey_agent_info)
    env = Environment(env_info)
    brain = Brain(env)
    memory = Memory(env)

    # Train agent on environment
    N_episodes = 10000
    print("\nTraining '{}' agent to hunt '{}' agent on {} environment for {} episodes (hunter_epsilon = {}, prey_epsilon = {})...\n".format(hunter_agent.name, prey_agent.name, env.name, N_episodes, hunter_agent.epsilon, prey_agent.epsilon))


    exit()

    memory.reset_run_counters()  # reset run counters once only
    for episode in range(N_episodes):
        memory.reset_episode_counters()  # reset episodic counters

        hunter_state = env.starting_state()  # starting state
        prey_state = env.starting_state()

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
        if (episode + 1) % (N_episodes / 20) == 0:
            print(" episode = {}/{}, reward = {:.1F}, n_actions = {}, dQsum = {:.2E}".format(episode + 1, N_episodes,
                                                                                             memory.R_total_episode,
                                                                                             memory.N_actions_episode,
                                                                                             dQsum))




# Driver 
if __name__ == '__main__':
    main()