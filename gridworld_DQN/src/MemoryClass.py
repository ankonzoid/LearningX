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

    def append_to_memory(self, state, state_next, action, MN_output, prob, reward):
        self.state_memory.append(state)
        self.state_next_memory.append(state_next)
        self.action_memory.append(action)
        self.MN_output_memory.append(MN_output)
        self.prob_memory.append(prob)
        self.reward_memory.append(reward)

    def clear_memory(self):
        self.state_memory = []
        self.state_next_memory = []
        self.action_memory = []
        self.MN_output_memory = []
        self.prob_memory = []
        self.reward_memory = []