"""

 MemoryClass.py  (author: Anson Wong / git: ankonzoid)

"""
class Memory:

    def __init__(self, info):

        # Memory info
        self.memory_info = info["memory"]

        self.clear_memory()

    # ===================
    # Memory
    # ===================

    def append_to_memory(self, state, action, PNprob, prob, reward):
        self.state_memory.append(state)
        self.action_memory.append(action)
        self.PNprob_memory.append(PNprob)
        self.prob_memory.append(prob)
        self.reward_memory.append(reward)

    def clear_memory(self):
        self.state_memory = []
        self.action_memory = []
        self.PNprob_memory = []
        self.prob_memory = []
        self.reward_memory = []