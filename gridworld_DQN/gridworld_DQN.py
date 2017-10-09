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
    N_episodes = 200
    load_MN = False  # load model
    save_MN = True  # save model on last episode
    save_MN_filename = os.path.join("model", "model.h5")

    info = {
        "env": {"Ny": 20,
                "Nx": 20},
        "agent": {"policy_mode": "epsgreedy", # "epsgreedy", "softmax"
                  "eps": 1.0,
                  "eps_decay": 2.0*np.log(10.0)/N_episodes},
        "brain": {"discount": 0.99,
                  "learning_rate": 0.9},
        "memory": {}
    }

    # ==============================
    # Setup environment and agent
    # ==============================
    env = Environment(info)
    agent = Agent(env, info)
    brain = Brain(env, info)
    memory = Memory(info)

    if load_MN:
        brain.load_MN(save_MN_filename)

    # ==============================
    # Train agent
    # ==============================
    for episode in range(N_episodes):

        iter = 0
        state = env.starting_state()
        while env.is_terminal_state(state) == False:
            # Pick an action by sampling action probabilities
            action, MN_output, prob = agent.get_action(state, brain, env)
            # Collect reward and observe next state
            reward = env.get_reward(state, action)
            state_next = env.perform_action(state, action)
            # Append quantities to memory
            memory.append_to_memory(state, state_next, action, MN_output, prob, reward)
            # Transition to next state
            state = state_next
            iter += 1

        # Print
        policy_mode = agent.agent_info["policy_mode"]
        if (policy_mode == "epsgreedy"):

            print("[episode {}] mode = {}, iter = {}, eps = {:.4F}, reward = {:.2F}".format(episode, policy_mode, iter, agent.eps_effective, sum(memory.reward_memory)))

        elif (policy_mode == "softmax"):

            print("[episode {}] mode = {}, iter = {}, reward = {:.2F}".format(episode, policy_mode, iter, sum(memory.reward_memory)))

        # Update MN when episode finishes
        brain.update(memory, env)
        agent.episode += 1

        # Save MN
        if save_MN and (episode == N_episodes-1):
            brain.save_MN(save_MN_filename)

        # Clear memory for next episode
        memory.clear_memory()

    # ==============================
    # Results
    # ==============================


# Driver
if __name__ == "__main__":
    main()