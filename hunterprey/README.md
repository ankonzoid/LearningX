# Solving the Hunter-Prey problem as a single-agent problem using relative coordinates (`hunterprey.py`)

We train a hunter agent to chase a stationary prey agent that spawns repeatedly every time it is caught on a 2D square N x N grid (integer N). The hunter has a choice of 4 actions at each time step of moving up / down / left / right by 1 grid square.

Our implementation uses a *single* epsilon-greedy agent and a Monte Carlo on-policy reward-averaging learning algorithm, similar to what was done in our previous example of [Training an Agent to beat GridWorld](https://github.com/ankonzoid/Deep-Reinforcement-Learning-Tutorials/blob/master/gridworld). We will explain below why mapping the hunter and prey global coordinates to a single relative coordinate allows us to reduce the problem to one with a *single* agent in a fixed environment of searching for the (0,0) grid square.

<p align="center">
<img src="https://github.com/ankonzoid/Deep-Reinforcement-Learning-Tutorials/blob/master/hunterprey/results/hunterprey.gif" width="70%">
</p>

Although the generalized Hunter-Prey problem does require a dynamic second agent, the simplest version of spawning a stationary prey around the grid after it is caught can be reduced to a stationary environment of a single-agent finding its way to the origin (0, 0) in relative grid coordinates of the hunter with respect to the prey. This method of fixing the reference frame to a moving body is very common in obtaining simpler equations of motion in 2-body classical mechanics physics problems.

To show how relative coordinates work, consider a hunter located at global cartesian coordinates (2, 4) and a prey at global cartesian coordinates (1, 1). With basic subtraction of the hunter coordinates by the prey coordinates, we get (2, 4) - (1, 1) = (+1, +3) as the relative position of the hunter with respect to the prey in cartesian coordinates. If the hunter can somehow learn to navigate around and arrive at (0,0) in this coordinate system, then we will have the hunter coinciding with the prey on the same grid square (also known as the hunter capturing the prey!).

<p align="center">
<img src="https://github.com/ankonzoid/Deep-Reinforcement-Learning-Tutorials/blob/master/hunterprey/images/coverart.png" width="50%">
</p>

By fixing the reference frame to the prey, we no longer need to concern ourselves with the global earth reference frame — we can work solely in the prey reference frame. This means we can focused on the relative state space of S_{hunter, relative} instead of what would be a more daunting direct product S_{hunter} x S_{prey} global state space. Doing so not only shrinks our state search space to train our agent faster, but it also eliminates much redundancy in training an agent on physical situations that are translationally equivalent i.e. teaching a hunter at (2,4) to catch a prey at (1,1) now also teaches the hunter at (3,5) to catch a prey at (2,2) because of the equivalence in relative positions of (+1,+3) of the hunter from the prey.

To see exactly how we set up the single-agent framework, let us start with a global grid of size (N, N). In exchange for the luxury of working with a single-agent, we will have to expand our relative position state space grid to hold all possible relative positions. Specifically this number works out to be an enlarged state space grid of size (2N-1, 2N-1) with integer values of [-N+1, N-1] x [-N+1, N-1]. The reasoning behind this is given a particular cartesian direction of x or y, the maximum displacement between any 2 grid points on the global grid is N grid squares. So by accounting for both positive and negative displacements we have the full range of 2*N-1 possible grid square displacements (the -1 here comes from the shared zero grid square distance). Once we have figured this enlarged (but fixed) grid state space, we can simply solve our seemingly double-agent Hunter-Prey problem as a single-agent Grid World problem of searching for (0, 0) on a [-N+1, N-1] x [-N+1, N-1] grid!

### Usage:

> python hunterprey.py

### Libraries required:

* numpy
* ffmpy (`pip3 install MoviePy`)
