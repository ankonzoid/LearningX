"""

 hunterprey.py  (author: Anson Wong / git: ankonzoid)

 Trains a hunter agent to capture a prey agent on a (Ny, Nx) grid.

"""
import numpy as np
import random, itertools, operator, sys, os

sys.path.append("../src")
from AgentClass import Agent
from BrainClass import Brain
from EnvironmentClass import Environment
from MemoryClass import Memory


def main():
    print("Hunter-Prey world")

    # Settings
    hunter_agent_info = {"epsilon": 0.2}
    prey_agent_info = {"epsilon": 1.0}
    env_info = {"Ny": 7, "Nx": 7}

    # Set up agent and environment
    hunter_agent = Agent(hunter_agent_info)
    prey_agent = Agent(prey_agent_info)
    env = Environment(env_info)
    brain = Brain(env)
    memory = Memory(env)





# Driver 
if __name__ == '__main__':
    main()