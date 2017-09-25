"""

 gym_env_descrip.py (author: Anson Wong / git: ankonzoid)

 Prints descriptions of all available gym environments.

"""
from gym import envs

class NullE:
    def __init__(self):
        self.observation_space = "N/A"
        self.action_space = "N/A"
        self.reward_range = "N/A"

for e in envs.registry.all():
    try:
        env = e.make()
    except:
        env = NullE()
        continue  #  Skip these for now

    print("")
    print("e.id = {}".format(e.id))
    print("env.observation_space = {}".format(env.observation_space))
    print("env.action_space = {}".format(env.action_space))
    print("env.reward_range = {}".format(env.reward_range))
    print("e.timestep_limit = {}".format(e.timestep_limit))
    print("e.trials = {}".format(e.trials))
    print("e.reward_threshold = {}".format(e.reward_threshold))
