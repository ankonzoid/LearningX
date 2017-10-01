# Grid-based Hunter-Prey agent (`hunterprey.py`)

We train a hunter agent to chase a non-moving prey agent that spawns repeatedly every time it is caught on a 2D square N x N grid (integer N). The hunter has a choice of 4 actions at each time step of moving up / down / left / right by 1 grid square.

Our implementation uses an epsilon-greedy agent and a Monte Carlo on-policy reward-averaging learning algorithm, similar to what was done in our previous example of [Training an Agent to beat GridWorld](https://github.com/ankonzoid/Deep-Reinforcement-Learning-Tutorials/blob/master/gridworld). 

<p align="center">
<img src="https://github.com/ankonzoid/Deep-Reinforcement-Learning-Tutorials/blob/master/hunterprey/results/hunterprey.gif" width="70%">
</p>

Although the problem seems like we are introducing a second agent, or a dynamic environment with time-dependence, it can really can be simplified and solved in the framework of a single agent findings its way to the origin (0, 0) in *relative grid coordinates of the hunter with respect to the prey*. The hunter's arrival at the relative coordinates of (0,0) corresponds to the situation of the hunter and agent coinciding on the same grid square (also know as the prey capturing the prey!).

By using relative coordinates of the hunter (with respect to the prey), this means:

* The earth is no longer the working frame of reference, there is no need to consider the direct product state space of S_{hunter} x S_{prey} holding all possible global coordinates pairs for the hunter and the prey. Instead you just need S_{hunter} which represents all relative coordinates away from the prey. Doing so not only shrinks the state space, but it also eliminates the need to redundantly train the agent to on situations where the global coordinates are different but the relative coordinates are the same *i.e. teaching a hunter at (0,0) to catch a prey at (2,3) should be transfer to the situation of instructing the hunter at (1,1) to catch a prey (3,4), as both the relative positions of the hunter from the prey is (-2,-3)*.

* If the global grid is of size (N,N), then the relative position state space will be of size (2N-1, 2N-1) to account for all the possible relative positions the hunter can have from the prey on the global (N,N) grid as the maximum separation is (N-1) grid-squares in both x and y directions. With relative coordinates in place, our problem is reduced to the statoc Grid World problem of searching for (0,0) in a [-N+1,N-1] x [-N+1,N-1].

* Note that generalization to more complex problems of multi-agent systems or added obstacles will require a drastic change in the framework *i.e. introduct neural networks, add more agents*.

### Usage:

> python hunterprey.py

### Libraries required:

* numpy
* ffmpy (`pip3 install MoviePy`)
