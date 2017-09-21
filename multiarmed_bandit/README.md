# Solving the Multi-armed Bandit Problem

The multi-armed bandit problem is as follows: we have _n_ rigged slot machines bandits each with their own probability distribution of success (which are inaccessible to you). Pulling the lever of any machine gives us either a reward of +1 (success) or -1 (failure). Our goal is to play many episodes of this game with the goal of *maximizing* our total accumulated rewards. 

In this code, we deploy an epsilon-greedy agent to play the multi-armed bandit game for a fixed number of episodes using a policy-based optimization method of policy gradient updates on a feed-forward neural network approximation of the policy. As a result of playing the game, we learn the optimal policy which in this case is the best bandit (slot machine with the highest probability of success).

We also provide appendices for the: 
* Epsilon-greedy agent (Appendix A)
* Policy gradients (Appendix B)

### Usage:

> python MAB_tabular.py

> python MAB_group_tabular.py

> python MAB_PG.py

### Example output:

In the example provided, we train on 40,000 episodes with `epsilon = 0.6` and 5 bandits with success probabilities of `{0.6, 0.1, 0.9, 0.3, 0.7}` respectively. As can be read, the optimal policy found should select bandit #3 as the global minima, but it is possible on the off-chances for the optimization algorithm to get stuck in local minima of J(p) and select bandits #1 or #5 (we have not run in cases where it selects #2 or #4).  

### Libraries required:

* tensorflow

## Appendix A: The epsilon-greedy agent

The *epsilon-greedy agent* is an agent which at decision time either selects a greedy action with _1-epsilon_ probability, or explores the entire action space with the remaining _epsilon_ probability. Taking a greedy action means selecting the action with the highest expected reward.

Note that a mixture of exploration and greed is a quintessential aspect of reinforcement learning. Exploration is necessary for finding the optimal policy as the agent needs to understand the environment it is in, and to learn routes leaing to high long-term rewards. Although greed is not necessarily needed for finding the optimal policy, a purely exploratory agent does not utilize its experience to strategize for its future moves -- therefore no actual learning is involved without a greed mechanism in play. 


## Appendix B: Policy gradients

Given a policy objective function J(p) and policy p, the method of policy gradients is to iteratively search for local minima of J(p) by updating our current estimation of it by a step size proportional to the direction of the gradient dJ(p)/dp (basically gradient descent).

In this code, we use an objective function of J(p) = -w_action*reward_action, where w_action is the weight associated with the action chosen, and reward_action is the reward sampled from the bandit. This is designed such that in order to keep J(p) as negative as possible, w_action tends to become positively large for action rewards of +1, and negatively large for action rewards of -1. 