"""

 gym_DPG_1D.py  (author: Anson Wong / git: ankonzoid)

 Teach an agent to play in gym environments where the state space is a 2D grid of pixels.
 We search for optimal policy using deep policy gradients.

"""
import numpy as np
import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), "src"))

from AgentClass import Agent
from BrainClass import Brain
from MemoryClass import Memory
import utils

def main():
    # ==============================
    # Settings
    # ==============================
    N_episodes = 1000
    load_PN = False
    save_PN = False
    save_PN_filename = os.path.join("model", "PN_model.h5")
    render = False

    # ==============================
    # Import gym environment
    # ==============================
    import gym

    env_str = "LunarLander-v2"  # CartPole-v0, LunarLander-v2
    save_folder = os.path.join("results", env_str)
    env = gym.make(env_str)
    #env = gym.wrappers.Monitor(env, save_folder, force=True)

    utils.check_env(env)  # check environment has Discrete() action and Box() observation space

    # =====================================
    # Collect state and action dimensions (in the processed state space)
    # =====================================
    state_dim = env.observation_space.shape
    action_dim = (env.action_space.n,)
    state_size = np.prod(np.array(list(state_dim), dtype=np.int))
    action_size = np.prod(np.array(list(action_dim), dtype=np.int))

    # ==============================
    # Setup environment and agent
    # ==============================
    env_info = {"state_dim": state_dim, "action_dim": action_dim, "state_size": state_size, "action_size": action_size}
    agent_info = {"policy_mode": "epsilongreedy", "epsilon": 1.0, "epsilon_decay": 2.0 * np.log(10.0) / N_episodes}
    #agent_info = {"policy_mode": "softmax"}
    brain_info = {"discount": 0.99, "learning_rate": 0.1, "arch": "1D"}
    memory_info = {}

    info = {"env": env_info, "agent": agent_info, "brain": brain_info, "memory": memory_info}

    agent = Agent(info)
    brain = Brain(info)
    memory = Memory(info)

    if load_PN:
        brain.load_PN(save_PN_filename)

    # ==============================
    # Train agent
    # ==============================
    for episode in range(N_episodes):

        # Reset agent state
        observation = env.reset()
        #state_processed_previous = np.zeros_like(process_img(observation))

        # Run episode
        iter = 0
        done = False
        while not done:
            # Render
            if render:
                env.render(mode='rgb_array')
            # Current state
            state_processed = observation
            # Let agent pick an action
            action, PNprob, prob = agent.get_action(state_processed, brain)
            # Transition to next state and collect reward
            state_new, reward, done, info = env.step(action)
            # Append quantities to memory
            memory.append_to_memory(state_processed, action, PNprob, prob, reward)
            # Update iteration parameter
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