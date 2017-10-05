"""

 MemoryClass.py  (author: Anson Wong / git: ankonzoid)

"""
class Memory():

    def __init__(self, memory_info):
        self.clear_memory()

    # ===================
    # Memory
    # ===================

    def append_to_memory(self, state, action, Qprob, prob, reward):
        self.state_memory.append(state)
        self.action_memory.append(action)
        self.Qprob_memory.append(Qprob)
        self.prob_memory.append(prob)
        self.reward_memory.append(reward)

    def clear_memory(self):
        self.state_memory = []
        self.action_memory = []
        self.Qprob_memory = []
        self.prob_memory = []
        self.reward_memory = []