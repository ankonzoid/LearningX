# Solving the Hunter-Prey problem as a single-agent problem using relative coordinates (`hunterprey.py`)

We train a hunter agent to chase a non-moving prey agent that spawns repeatedly every time it is caught on a 2D square N x N grid (integer N). The hunter has a choice of 4 actions at each time step of moving up / down / left / right by 1 grid square.

Our implementation uses a *single* epsilon-greedy agent and a Monte Carlo on-policy reward-averaging learning algorithm, similar to what was done in our previous example of [Training an Agent to beat GridWorld](https://github.com/ankonzoid/Deep-Reinforcement-Learning-Tutorials/blob/master/gridworld). We will explain below why mapping the hunter and prey global coordinates to a single relative coordinate allows us to reduce the problem to one with a *single* agent in a fixed environment of searching for the (0,0) grid square.

<p align="center">
<img src="https://github.com/ankonzoid/Deep-Reinforcement-Learning-Tutorials/blob/master/hunterprey/results/hunterprey.gif" width="70%">
</p>

Although the Hunter-Prey problem seems like we are introducing a second agent or a dynamic environment with time-dependence, it can really can be simplified to a single agent finding its way to the origin (0, 0) in fixed relative grid coordinates of the hunter with respect to the prey. This type of method is used very often in classical 2-body physics problems to make the equations of motion easier to solve by setting the frame of reference to on of the bodies.

As a concrete example of how to set up relative coordinates for a hunter at global coordinates (2,4) and a prey at global coordinates (1,1), all we do in the classical picture is subtract the hunter global coordinates by that of the prey to get (2,4) - (1,1) = (+1,+3) as the single relative positional measure of the hunter with respect to the prey as in relative units. This means that the hunter needs to somehow arrive at (0,0) in relative coordinates to coincide with the prey at the same grid square (also know as the prey capturing the prey!).

By introducing reference frames (motivated by physics) of the hunter with respect to the prey, we no longer explicitly care are concerned with the global earth frame of reference for any simulation dynamics. We essentially exchanged the need to keep track of the direct product S_{hunter} x S_{prey} global state space to to just a S_{hunter, relative} state space holding relative positions of the hunter from the prey. Doing so not only shrunk our state search space, but it also eliminated much redundancy in training an agent on situations where the global coordinates of the hunter and prey were different, but the relative coordinates the same. This means that in global coordinates, teaching a hunter at (0,0) to catch a prey at (2,3) also teaches the hunter at (1,1) to catch a prey at (3,4) because of the equivalence in relative positions of the hunter from the prey.

Implementation-wise, if the global grid is of size (N,N), then the relative position state space we need to work with is an enlarged size of (2N-1, 2N-1) to account for all the possible relative positions the hunter can have from the prey on the global (N,N) grid. This is naturally the case as the maximum separation between any 2 grid points on the global (N,N) grid is N grid squares in any direction, so the full dimensionality is 2*N-1 grid squares in both x and y directions (the -1 comes from the shared zero distance). This type of reasoning is our bread-and-butter here and allows us to simplify our seemingly double-agent system to single-agent system of finding a fixed (0,0) in a [-N+1,N-1] x [-N+1,N-1] Grid World problem optimally!

### Usage:

> python hunterprey.py

### Libraries required:

* numpy
* ffmpy (`pip3 install MoviePy`)
