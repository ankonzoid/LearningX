# Solving the Multi-armed Bandit Problem

The multi-armed bandit problem is a classic reinforcement learning example where we are given a slot machine with n arms (bandits) with each arm having its own rigged probability distribution of success. Pulling any one of the arms gives you a stochastic reward of either R=+1 for success, or R=0 for failure. Our objective is to pull the arms one-by-one in sequence such that we maximize our total reward collected in the long run.

<p align="center">
<img src="https://github.com/ankonzoid/Deep-Reinforcement-Learning-Tutorials/blob/master/multiarmed_bandit/images/coverart.png"  width="50%">
</p>

In this code, we deploy an epsilon-greedy agent to play the multi-armed bandit game for a fixed number of episodes using a well-established classical reinforcement learning method of an epsilon-greedy agent and reward-average sampling to compute the action-values Q(a).


<p align="center">
<img src="https://github.com/ankonzoid/Deep-Reinforcement-Learning-Tutorials/blob/master/multiarmed_bandit/results/MAB_actions.png"  width="75%">
</p>

### Usage:

In the example provided, we train on 2,000 experiments with 10,000 episodes per experiment. The default exploring parameter is `epsilon = 0.1` and 10 bandits are intialized with success probabilities of `{0.10, 0.50, 0.60, 0.80, 0.10, 0.25, 0.60, 0.45, 0.75, 0.65}`. To run the code, use 

> python multiarmed_bandit.py

Bandit #4 should be selected as the "best" bandit on average, with bandit #9 running second, and bandit #10 as a far third.

### Libraries required:

* numpy
