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
    N_episodes = 10000
    load_PN = False
    save_PN = False
    save_PN_filename = os.path.join("model", "PN_model.h5")
    render = True

    # ==============================
    # Import gym environment
    # ==============================
    import gym

    env_str = "CartPole-v0"  # CartPole-v0, LunarLander-v2
    save_folder = os.path.join("results", env_str)
    env = gym.make(env_str)
    env = gym.wrappers.Monitor(env, save_folder, force=True)

    utils.check_env(env)  # check environment has Discrete() action and Box() observation space

    # ==============================
    # Define our pre-processing for our environment
    # ==============================

    # Design pre-processing for Pong-v0
    def process_img(img_raw):
        img = img_raw
        # Extract sub image
        img = img[32:196, :, :]  # for Pong-v0
        # Reduce 3D rgb image to 2D image by taking only r channel
        img = img[::2, ::2, 0]
        # Convert 2D image to 0's and 1's
        img[img == 144] = 0  # convert enemy to zero
        img[img == 109] = 0  # convert background to zero
        img[img != 0] = 1  # ball and our agent
        return img
    
    # Our second layer of pre-processing to include velocity
    def subtract_img(img_now, img_previous):
        img_subtract = img_now - img_previous
        return img_subtract

    # Turn this on if you want to adjust our process_img function
    if 0:
        state_1 = env.reset()
        state_2 = process_img(state_1)
        utils.compare_imgs(state_1, state_2)
        exit()

    # =====================================
    # Collect state and action dimensions (in the processed state space)
    # =====================================
    state_dim = process_img(env.reset()).shape  # find shape of our processed image
    action_dim = (env.action_space.n,)
    state_size = np.prod(np.array(list(state_dim), dtype=np.int))
    action_size = np.prod(np.array(list(action_dim), dtype=np.int))

    # ==============================
    # Setup environment and agent
    # ==============================
    env_info = {"state_dim": state_dim, "action_dim": action_dim, "state_size": state_size, "action_size": action_size}
    #agent_info = {"policy_mode": "epsilongreedy", "epsilon": 1.0, "epsilon_decay": 2.0 * np.log(10.0) / N_episodes}
    agent_info = {"policy_mode": "softmax"}
    brain_info = {"discount": 0.9, "learning_rate": 0.4, "arch": "2D"}
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
        state_processed_previous = np.zeros_like(process_img(observation))

        # Run episode
        iter = 0
        done = False
        while not done:
            # Render
            if render:
                env.render(mode='rgb_array')
            # Current state
            state_processed = process_img(observation)  # extract subimage and 3D->2D
            state_processed = subtract_img(state_processed, state_processed_previous)  # include velocity
            state_processed_previous = state_processed  # keep a copy of previous state for next velocity calculation
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