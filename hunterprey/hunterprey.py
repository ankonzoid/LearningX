"""

 hunterprey.py  (author: Anson Wong / git: ankonzoid)

 Trains a hunter agent to capture a prey agent on a (Ny, Nx) grid.

"""
import numpy as np
import random
import itertools
import operator

def main():
    print("Hunter-Prey world")
    hunter_agent_info = {"epsilon": 0.2}
    prey_agent_info = {"epsilon": 1.0}


    hunter_agent = Agent(hunter_agent_info)
    prey_agent = Agent(prey_agent_info)






# Driver 
if __name__ == '__main__':
    main()