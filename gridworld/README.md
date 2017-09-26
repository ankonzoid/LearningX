# Training an agent to beat Grid World (`gridworld.py`)

Given a 2D rectangular grid with opposing corners at (0, 0) and (Ny-1, Nx-1) for Ny, Nx integers, we train an agent standing at (0,0) to find an optimized path to (Ny-1, Nx-1) in the least number of grid steps possible. 

In this code, we implement a *Monte Carlo on-policy* learning algorithm of reward-averaging on an epsilon-greedy agent. *Tabular forms* of the action-value *Q(s,a)*, reward *R(s,a)*, and policy *P(s)* functions are used. The agent is restricted to only actions of displacing itself up/down/left/right by 1 grid square. 

<p align="center">
<img src="https://github.com/ankonzoid/Deep-Reinforcement-Learning-Tutorials/blob/master/gridworld/images/policy.png" width="40%">
</p>

There are multiple degenerate optimal policy solutions that take the optimized (Ny + Nx) actions to reach the goal, which means the _specific_ optimal policy found is **not** a unique solution. Rather, what is of higher importance is that the optimal policy found should be physical intuitive *i.e. the optimal policy only suggests actions bringing the agent directly closer to the (Ny-1, Nx-1) goal*. 

Although tabular forms are not feasible in reinforcement learning for large state/action spaces, it is reasonable here for small rectangular grid sizes for demonstration when Ny < 10 and Nx < 10.

### Example output:

Our provided example has been hard-coded with parameters

```swift
Ny = 7
Nx = 7
epsilon = 0.5
n_episodes = 50000
R_goal = 100
R_nongoal = -0.1
```

with run output of

```swift
 Final policy:

  [[2 1 1 2 2 2 2]
   [2 2 1 1 2 1 2]
   [1 2 1 1 2 2 2]
   [2 1 1 2 1 1 2]
   [1 1 2 2 2 1 2]
   [1 1 1 1 1 2 2]
   [1 1 1 1 1 1 3]]

  action['up'] = 0
  action['right'] = 1
  action['down'] = 2
  action['left'] = 3
```

### Usage:

> python gridworld.py

### Libraries required:

* numpy
