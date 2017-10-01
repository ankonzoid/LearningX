# Grid-based Hunter-Prey agent (`hunterprey.py`)

For a 2D square (N, N) for integer N, we train an agent chase repeatedly spawning prey that spawn  every time it is caught. The agent and learning is similar to what we used in Grid World here:

[Training an Agent to beat GridWorld](https://github.com/ankonzoid/Deep-Reinforcement-Learning-Tutorials/blob/master/gridworld)

We use a Monte Carlo on-policy learning algorithm of reward-averaging on an epsilon-greedy agent. *Tabular forms* of the action-value *Q(s,a)*, reward *R(s,a)*, and policy *P(s)* functions are used. The agent is restricted to only actions of displacing itself up/down/left/right by 1 grid square. 


![alt text](https://github.com/ankonzoid/Deep-Reinforcement-Learning-Tutorials/blob/master/hunterprey/results/hunterprey.gif "Logo Title Text 1")


### Usage:

> python hunterprey.py

### Libraries required:

* numpy
* ffmpy (`pip3 install MoviePy`)
