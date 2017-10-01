# Grid-based Hunter-Prey agent (`hunterprey.py`)

We train an agent to chase a non-moving prey that spawns repeatedly every time it is caught on a 2D square N x N grid (integer N). 

Our implementation uses an epsilon-greedy agent and a Monte Carlo on-policy reward-averaging learning algorithm, similar to what was done in our previous example of [Training an Agent to beat GridWorld](https://github.com/ankonzoid/Deep-Reinforcement-Learning-Tutorials/blob/master/gridworld).

<p align="center">
<img src="https://github.com/ankonzoid/Deep-Reinforcement-Learning-Tutorials/blob/master/hunterprey/results/hunterprey.gif" width="70%">
</p>

### Usage:

> python hunterprey.py

### Libraries required:

* numpy
* ffmpy (`pip3 install MoviePy`)
