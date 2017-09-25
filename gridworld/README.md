# Training an agent to move from corner to corner of a 2D rectangular grid (`gridworld.py`)

Given a 2D rectangular grid with opposing corners at (0,0) and (Ly,Lx) where Ly, Lx are integers, we train an agent standing at (0,0) to find its way to (Ly, Lx) in the least number of steps possible. 

In this code, we implement a *Monte Carlo on-policy* learning algorithm (hard epsilon-greedy selection) and use *tabular forms* for our action-value *Q(s,a)*, reward *R(s,a)*, and policy *P(s)* functions. The agent is only allowed actions of displacing itself exactly up/down/left/right by 1 grid square. 

We also provide appendices for the: 
* Degenerate optimal policies (Appendix A)
* Computational limitations tabular forms (Appendix B)
* Action-value function *Q(s,a)* (Appendix C)

### Usage:

> python corner2corner.py

### Example output:

Our provided example has been hard-coded with parameters

```swift
Ly = 6
Lx = 6
epsilon = 0.5
n_episodes = 20000
R_goal = 10000
```

with run output of

```swift
 Final policy (action indices 0, 1, 2, 3):
 
  [[2 1 1 2 2 2 2]
   [2 2 1 1 2 1 2]
   [1 2 1 1 2 2 2]
   [2 1 1 2 1 1 2]
   [1 1 2 2 2 1 2]
   [1 1 1 1 1 2 2]
   [1 1 1 1 1 1 3]]

 Action[indices]:
 
  action[0] = [-1 0]
  action[1] = [0 1]
  action[2] = [1 0]
  action[3] = [0 -1]
```

## Appendix A: Degenerate optimal policies

There are multiple degenerate optimal policy solutions that take Lx + Ly - 2 actions to reach the goal, which means the _specific_ optimal policy found is **not** a unique solution. Rather, what is of higher importance is that the optimal policy found should be physical intuitive *i.e. the optimal policy only suggests actions bringing the agent directly closer to the (Ly, Lx) goal*. 

## Appendix B: Computational limitations of tabular forms

Although tabular forms are not feasible in reinforcement learning for large state/action spaces, it is reasonable here for small rectangular grid sizes for demonstration when Ly < 10 and Lx < 10.

## Appendix C: The action-value Q(s,a)

Being 'greedy' is a delicate and philosophical issue, but in classical reinforcement learning an agent is considered to be greedily taking an action if the action it picks has the largest action-value *Q(s,a)* amongst the rest. Note that *Q(s,a)* is not known a priori (or else there would be no purpose of training the agent) so it is initialized and computed iteratively from the experience of the agent interacting with the environment. One common way is to use the Bellman optimality constraint equations in recursive form.