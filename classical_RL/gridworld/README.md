# GridWorld using Q-Learning (`gridworld.py`)

We use Q-learning to train an epsilon-greedy agent to find the shortest path between position `(0, 0)` to opposing corner `(Ny-1, Nx-1)` of a 2D rectangular grid in the 2D GridWorld environment of size (Ny, Nx).

The agent is restricted to displacing itself up/down/left/right by 1 grid square per action. The agent receives a `-0.1` penalty for every action not reaching the terminal state (to incentivize shortest path search), and a `100` reward upon reaching the terminal state (not necessary but helps improve the value signal). The agent exploration parameter `epsilon` also decays by a multiplicative constant after every training episode. *Tabular forms* of the action-value *Q(s,a)*, reward *R(s,a)*, and policy *P(s)* functions are used. 

<p align="center">
<img src="images/coverart.png" width="30%">
</p>

Note: the optimal policy exists but is a highly degenerate solution because of the multitude of ways one can traverse down the grid in the minimum number of steps. Therefore a greedy policy that always moves the agent closer towards the goal can be considered an optimal policy (can get to the goal in `Ny + Nx - 2` actions). In our example, this corresponds to actions of moving right or down to the bottom-right corner.

### Usage

> python3 gridworld.py

### Example Output

 ```
Training agent...

[episode 1/500] eps = 0.990 -> iter = 406, rew = 59.5
[episode 10/500] eps = 0.904 -> iter = 26, rew = 97.5
[episode 20/500] eps = 0.818 -> iter = 64, rew = 93.7
[episode 30/500] eps = 0.740 -> iter = 62, rew = 93.9
[episode 40/500] eps = 0.669 -> iter = 46, rew = 95.5
[episode 50/500] eps = 0.605 -> iter = 32, rew = 96.9
[episode 60/500] eps = 0.547 -> iter = 46, rew = 95.5
[episode 70/500] eps = 0.495 -> iter = 16, rew = 98.5
[episode 80/500] eps = 0.448 -> iter = 22, rew = 97.9
[episode 90/500] eps = 0.405 -> iter = 26, rew = 97.5
[episode 100/500] eps = 0.366 -> iter = 36, rew = 96.5
[episode 110/500] eps = 0.331 -> iter = 18, rew = 98.3
[episode 120/500] eps = 0.299 -> iter = 18, rew = 98.3
[episode 130/500] eps = 0.271 -> iter = 20, rew = 98.1
[episode 140/500] eps = 0.245 -> iter = 22, rew = 97.9
[episode 150/500] eps = 0.221 -> iter = 16, rew = 98.5
[episode 160/500] eps = 0.200 -> iter = 20, rew = 98.1
[episode 170/500] eps = 0.181 -> iter = 20, rew = 98.1
[episode 180/500] eps = 0.164 -> iter = 14, rew = 98.7
[episode 190/500] eps = 0.148 -> iter = 16, rew = 98.5
[episode 200/500] eps = 0.134 -> iter = 14, rew = 98.7
[episode 210/500] eps = 0.121 -> iter = 14, rew = 98.7
[episode 220/500] eps = 0.110 -> iter = 18, rew = 98.3
[episode 230/500] eps = 0.099 -> iter = 14, rew = 98.7
[episode 240/500] eps = 0.090 -> iter = 14, rew = 98.7
[episode 250/500] eps = 0.081 -> iter = 14, rew = 98.7
[episode 260/500] eps = 0.073 -> iter = 16, rew = 98.5
[episode 270/500] eps = 0.066 -> iter = 16, rew = 98.5
[episode 280/500] eps = 0.060 -> iter = 14, rew = 98.7
[episode 290/500] eps = 0.054 -> iter = 14, rew = 98.7
[episode 300/500] eps = 0.049 -> iter = 14, rew = 98.7
[episode 310/500] eps = 0.044 -> iter = 14, rew = 98.7
[episode 320/500] eps = 0.040 -> iter = 14, rew = 98.7
[episode 330/500] eps = 0.036 -> iter = 14, rew = 98.7
[episode 340/500] eps = 0.033 -> iter = 16, rew = 98.5
[episode 350/500] eps = 0.030 -> iter = 14, rew = 98.7
[episode 360/500] eps = 0.027 -> iter = 16, rew = 98.5
[episode 370/500] eps = 0.024 -> iter = 14, rew = 98.7
[episode 380/500] eps = 0.022 -> iter = 14, rew = 98.7
[episode 390/500] eps = 0.020 -> iter = 14, rew = 98.7
[episode 400/500] eps = 0.018 -> iter = 14, rew = 98.7
[episode 410/500] eps = 0.016 -> iter = 14, rew = 98.7
[episode 420/500] eps = 0.015 -> iter = 14, rew = 98.7
[episode 430/500] eps = 0.013 -> iter = 14, rew = 98.7
[episode 440/500] eps = 0.012 -> iter = 14, rew = 98.7
[episode 450/500] eps = 0.011 -> iter = 14, rew = 98.7
[episode 460/500] eps = 0.010 -> iter = 14, rew = 98.7
[episode 470/500] eps = 0.010 -> iter = 14, rew = 98.7
[episode 480/500] eps = 0.010 -> iter = 14, rew = 98.7
[episode 490/500] eps = 0.010 -> iter = 14, rew = 98.7
[episode 500/500] eps = 0.010 -> iter = 14, rew = 98.7

Greedy policy(y, x):
[[1 1 2 1 1 2 3 2]
 [1 1 1 2 2 2 1 2]
 [1 1 1 2 1 2 1 2]
 [1 1 1 1 1 1 2 2]
 [1 1 1 1 2 1 1 2]
 [1 1 1 1 1 1 1 2]
 [1 1 1 1 1 1 1 2]
 [0 1 1 1 1 1 1 0]]

 action['up'] = 0
 action['right'] = 1
 action['down'] = 2
 action['left'] = 3
 ```

### Libraries

* numpy

### Author

Anson Wong
