"""

 gridworld_DPG.py  (author: Anson Wong / git: ankonzoid)

 Teach an agent to move optimally in GridWorld using deep policy gradients.

"""
import numpy as np
import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), "src"))

from EnvironmentClass import Environment
from AgentClass import Agent
from BrainClass import Brain
from MemoryClass import Memory

def main():
    # ==============================
    # Settings
    # ==============================
    N_episodes = 1000
    load_PN = False
    save_PN = True
    save_PN_filename = os.path.join("model", "PN_model.h5")

    info = {
        "env": {"Ny": 20,
                "Nx": 20},
        "agent": {"policy_mode": "epsilongreedy", # "epsilongreedy", "softmax"
                  "epsilon": 1.0,
                  "epsilon_decay": 2.0*np.log(10.0)/N_episodes},
        "brain": {"discount": 0.9,
                  "learning_rate": 0.4},
        "memory": {}
    }

    # ==============================
    # Setup environment and agent
    # ==============================
    env = Environment(info)
    agent = Agent(env, info)
    brain = Brain(env, info)
    memory = Memory(info)

    if load_PN:
        brain.load_PN(save_PN_filename)

    # ==============================
    # Train agent
    # ==============================
    for episode in range(N_episodes):

        iter = 0
        state = env.starting_state()
        while env.is_terminal_state(state) == False:
            # Pick an action by sampling PN(state) probabilities
            action, PNprob, prob = agent.get_action(state, brain, env)
            # Collect reward and observe next state
            reward = env.get_reward(state, action)
            state_new = env.perform_action(state, action)
            # Append quantities to memory
            memory.append_to_memory(state, action, PNprob, prob, reward)
            # Transition to next state
            state = state_new
            iter += 1

        # Print
        policy_mode = agent.agent_info["policy_mode"]
        if (policy_mode == "epsilongreedy"):

            print("[episode {}] mode = {}, iter = {}, epsilon = {:.4F}, reward = {:.2F}".format(episode, policy_mode, iter, agent.epsilon_effective, sum(memory.reward_memory)))

        elif (policy_mode == "softmax"):

            print("[episode {}] mode = {}, iter = {}, reward = {:.2F}".format(episode, policy_mode, iter, sum(memory.reward_memory)))

        # Update PN when episode finishes
        brain.update(memory)
        agent.episode += 1

        # Save PN
        if save_PN:
            brain.save_PN(save_PN_filename)

        # Clear memory for next episode
        memory.clear_memory()

    # ==============================
    # Results
    # ==============================


# Driver
if __name__ == "__main__":
    main()