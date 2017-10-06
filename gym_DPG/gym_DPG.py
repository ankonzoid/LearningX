"""

 gridworld_DPG.py  (author: Anson Wong / git: ankonzoid)

 Teach an agent to move optimally in GridWorld using deep policy gradients.

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
    agent_info = {"policy_mode": "epsilongreedy", "epsilon": 1.0, "epsilon_decay": 2.0*np.log(10.0)/N_episodes}
    #agent_info = {"policy_mode": "softmax"}
    brain_info = {"discount": 0.9, "learning_rate": 0.4}
    memory_info = {}

    # ==============================
    # Setup environment and agent
    # ==============================
    import gym

    env_str = "Pong-v0"  # CartPole-v0, LunarLander-v2, Pong-v0
    save_folder = os.path.join("results", env_str)
    env = gym.make(env_str)
    env = gym.wrappers.Monitor(env, save_folder, force=True)



    action_space= env.action_space
    observation_space = env.observation_space
    action_names = env.unwrapped.get_action_meanings()


    print(action_space)
    print(observation_space)
    print(action_names)


    if 1:
        # Make sure action space is discrete (Discrete)
        if 'n' in env.action_space.__dict__:
            print(env.action_space)
        elif 'low' in env.action_space.__dict__ and 'high' in env.action_space.__dict__:
            raise IOError("env.action_space is Box. Stop.")
        else:
            raise IOError("Invalid action space")

        # Make sure observation is continuous (Box)
        if 'n' in env.observation_space.__dict__:
            raise IOError("env.observation_space is Discrete. Stop.")
        elif 'low' in env.observation_space.__dict__ and 'high' in env.observation_space.__dict__:
            print(env.observation_space)
        else:
            raise IOError("Invalid observation space")

    exit()


    agent = Agent(env, agent_info)
    brain = Brain(env, brain_info)
    memory = Memory(memory_info)

    # ==============================
    # Train agent
    # ==============================
    for episode in range(N_episodes):

        iter = 0
        state = env.reset()  # reset state
        state = state  # convert to greyscale
        done = 0  # variable for when we have reached terminal state

        while done:
            # Render current state
            env.render()

            # Let agent pick an action
            action, PNprob, prob = agent.get_action(state, brain, env)

            # Perform action on current state
            state, reward, done, info = env.step(action)


            # Collect reward and observe next state
            reward = env.get_reward(state, action)
            state_new = env.get_action(state, action)
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

        # Clear memory for next episode
        memory.clear_memory()

    # ==============================
    # Results
    # ==============================


# Driver
if __name__ == "__main__":
    main()