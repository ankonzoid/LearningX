# Solving the Hunter-Prey problem using relative coordinates (`hunterprey.py`)

We train a hunter agent to chase a non-moving prey agent that spawns repeatedly every time it is caught on a 2D square N x N grid (integer N). The hunter has a choice of 4 actions at each time step of moving up / down / left / right by 1 grid square.

Our implementation uses a *single* epsilon-greedy agent and a Monte Carlo on-policy reward-averaging learning algorithm, similar to what was done in our previous example of [Training an Agent to beat GridWorld](https://github.com/ankonzoid/Deep-Reinforcement-Learning-Tutorials/blob/master/gridworld). We will explain below why mapping the hunter and prey global coordinates to a single relative coordinate allows us to reduce the problem to one with a *single* agent in a fixed environment of searching for the (0,0) grid square.

<p align="center">
<img src="https://github.com/ankonzoid/Deep-Reinforcement-Learning-Tutorials/blob/master/hunterprey/results/hunterprey.gif" width="70%">
</p>

Although the Hunter-Prey problem seems like we are introducing a second agent or a dynamic environment with time-dependence, it can really can be simplified and solved in the framework of a single agent finding its way to the origin (0, 0) in relative grid coordinates of the hunter with respect to the prey. For example, the relative coordinates between a hunter at global coordinates (1,1) and a prey at global coordinates (2,4), then the relative coordinates of the hunter with respect to the prey is (1,1) - (2,4) = (-1,-3). Therefore a hunter’s arrival at (0,0) in relative coordinates corresponds to the hunter coinciding with the prey at the same grid square (also know as the prey capturing the prey!).

By using our physics-motivated relative coordinates of the hunter with respect to the prey, we get that the earth is no longer our primary working frame of reference. We exchange the need to keep track of the direct product S_{hunter} x S_{prey} global state space to to just a S_{hunter, relative} state space holding relative positions of the hunter from the prey. Doing so not only shrinks the state search space, but it also eliminates much of the redundancy in training an agent on situations where the global coordinates are different but the relative coordinates are the same. What this means is that teaching a hunter at (0,0) to catch a prey at (2,3) should be transfer to the instructing of a hunter at (1,1) to catch a prey at (3,4), as both the relative positions of the hunter from the prey is (-2,-3).

If the global grid is of size (N,N), then the relative position state space we will be working with is an enlarged size of (2N-1, 2N-1) to account for all the possible relative positions the hunter can have from the prey on the global (N,N) grid. Intuitively you can see this by noticing that the maximum separation between 2 points on this global (N,N) grid is N grid squares in both x and y directions, so the dimensionality in either directions is 2*N-1 (the -1 comes from the shared zero distance). This procedure is what allows us to solve the problem of finding the (0,0) in optimal steps in a [-N+1,N-1] x [-N+1,N-1] Grid World.

### Usage:

> python hunterprey.py

### Libraries required:

* numpy
* ffmpy (`pip3 install MoviePy`)
