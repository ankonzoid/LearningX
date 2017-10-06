"""

 gym_DPG.py  (author: Anson Wong / git: ankonzoid)

 Teach an agent to play in gym environments using deep policy gradients.

"""
import numpy as np
import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), "src"))

from AgentClass import Agent
from BrainClass import Brain
from MemoryClass import Memory

def main():
    # ==============================
    # Settings
    # ==============================
    N_episodes = 1000

    # ==============================
    # Import gym environment
    # ==============================
    import gym

    env_str = "Pong-v0"  # CartPole-v0, LunarLander-v2, Pong-v0
    save_folder = os.path.join("results", env_str)
    env = gym.make(env_str)
    env = gym.wrappers.Monitor(env, save_folder, force=True)
    check_env(env)

    state_dim = env.observation_space.shape[:2]  # omit color channels (we will greyscale)
    action_dim = (env.action_space.n,)
    state_size = np.prod(np.array(list(state_dim), dtype=np.int))
    action_size = np.prod(np.array(list(action_dim), dtype=np.int))
    action_names = env.unwrapped.get_action_meanings()

    # ==============================
    # Setup environment and agent
    # ==============================
    env_info = {"state_dim": state_dim, "action_dim": action_dim, "state_size": state_size, "action_size": action_size,
                "action_names": action_names}
    agent_info = {"policy_mode": "epsilongreedy", "epsilon": 1.0, "epsilon_decay": 2.0 * np.log(10.0) / N_episodes}
    #agent_info = {"policy_mode": "softmax"}
    brain_info = {"discount": 0.9, "learning_rate": 0.4}
    memory_info = {}

    info = {"env": env_info, "agent": agent_info, "brain": brain_info, "memory": memory_info}

    agent = Agent(info)
    brain = Brain(info)
    memory = Memory(info)

    # ==============================
    # Train agent
    # ==============================
    for episode in range(N_episodes):

        # Reset agent state
        observation = env.reset()

        # Run episode
        iter = 0
        done = False
        while not done:
            # Render
            env.render()
            # Current state
            state = observation
            state_grey = convert_rgb2grey(state)
            # Let agent pick an action
            action, PNprob, prob = agent.get_action(state_grey, brain)
            # Transition to next state and collect reward
            state_new, reward, done, info = env.step(action)
            # Append quantities to memory
            memory.append_to_memory(state_grey, action, PNprob, prob, reward)
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

        # Clear memory for next episode
        memory.clear_memory()

    # ==============================
    # Results
    # ==============================


# ================================
# Side functions
# ================================

def check_env(env):
    # Make sure action space is discrete (Discrete)
    if 'n' in env.action_space.__dict__:
        pass
    elif 'low' in env.action_space.__dict__ and 'high' in env.action_space.__dict__:
        raise IOError("env.action_space is Box. Stop.")
    else:
        raise IOError("Invalid action space")

    # Make sure observation is continuous (Box)
    if 'n' in env.observation_space.__dict__:
        raise IOError("env.observation_space is Discrete. Stop.")
    elif 'low' in env.observation_space.__dict__ and 'high' in env.observation_space.__dict__:
        pass
    else:
        raise IOError("Invalid observation space")

def convert_rgb2grey(img_rgb):
    img_grey = np.dot(img_rgb[...,:3], [0.299, 0.587, 0.114])
    return img_grey


# Driver
if __name__ == "__main__":
    main()