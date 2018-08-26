# GridWorld using Q-Learning (`gridworld.py`)

We use Q-learning to train an epsilon-greedy agent to find the shortest path between position (0, 0) to opposing corner (Ny-1, Nx-1) of a 2D rectangular grid in the 2D GridWorld environment of size (Ny, Nx).

The agent is restricted to displacing itself up/down/left/right by 1 grid square per action. The agent receives a `-0.1` penalty for every action not reaching the terminal state (to incentivize shortest path search), and a `100` reward upon reaching the terminal state (not necessary but helps improve the value signal). The agent exploration parameter `epsilon` also decays by a multiplicative constant after every training episode. *Tabular forms* of the action-value *Q(s,a)*, reward *R(s,a)*, and policy *P(s)* functions are used. 

<p align="center">
<img src="https://github.com/ankonzoid/Deep-Reinforcement-Learning-Tutorials/blob/master/gridworld/images/coverart.png" width="40%">
</p>

Note: the optimal policy exists but is a highly degenerate solution because of the multitude of ways one can traverse down the grid in the minimum number of steps. Therefore a greedy policy that always moves the agent closer towards the goal can be considered an optimal policy (can get to the goal in `Ny + Nx - 2` actions). In our example, this corresponds to actions of moving right or down to the bottom-right corner.

### Usage

> python3 gridworld.py

### Example Output

 ```
 Training agent...

[episode = 100/10000] reward = 94.3, n_actions = 58, epsilon = 0.951

Greedy policy(y, x):
[[1 1 1 1 1 2]
 [1 1 1 1 1 2]
 [1 1 1 1 1 2]
 [1 1 1 1 1 2]
 [1 1 1 1 1 2]
 [1 1 1 1 1 0]]

 action['up'] = 0
 action['right'] = 1
 action['down'] = 2
 action['left'] = 3

 ...
 ...
 ...

[episode = 10000/10000] reward = 99.1, n_actions = 10, epsilon = 0.010

Greedy policy(y, x):
[[1 1 1 1 1 2]
 [1 1 1 1 1 2]
 [1 1 1 1 1 2]
 [1 1 1 1 1 2]
 [1 1 1 1 1 2]
 [1 1 1 1 1 0]]

 action['up'] = 0
 action['right'] = 1
 action['down'] = 2
 action['left'] = 3
 ```

And is also the case for `learning_mode = "QLearning"` if trained long enough (the current hard coded parameters usually give almost optimal policy).

### Libraries

* numpy
