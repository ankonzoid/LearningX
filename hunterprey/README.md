# Solving the Hunter-Prey problem as a single-agent problem using relative coordinates (`hunterprey.py`)

We train a hunter agent to chase a non-moving prey agent that spawns repeatedly every time it is caught on a 2D square N x N grid (integer N). The hunter has a choice of 4 actions at each time step of moving up / down / left / right by 1 grid square.

Our implementation uses a *single* epsilon-greedy agent and a Monte Carlo on-policy reward-averaging learning algorithm, similar to what was done in our previous example of [Training an Agent to beat GridWorld](https://github.com/ankonzoid/Deep-Reinforcement-Learning-Tutorials/blob/master/gridworld). We will explain below why mapping the hunter and prey global coordinates to a single relative coordinate allows us to reduce the problem to one with a *single* agent in a fixed environment of searching for the (0,0) grid square.

<p align="center">
<img src="https://github.com/ankonzoid/Deep-Reinforcement-Learning-Tutorials/blob/master/hunterprey/results/hunterprey.gif" width="70%">
</p>

Although the Hunter-Prey problem seems like we are introducing a second agent or creating a dynamic environment, the simple version of the problem can really be simplified to the case of a single-agent finding its way to the origin (0, 0) in fixed relative grid coordinates of the hunter with respect to the prey. This method of fixing the reference frame to a body is common in 2-body classical mechanics problems in physics for arriving at simpler equations of motion forms.

As a concrete example to we relative coordinates work, consider a hunter located at global cartesian coordinates (2,4) and a prey at global cartesian coordinates (1,1). With a basic subtraction of the hunter coordinates by the prey coordinates, we get (2,4) - (1,1) = (+1,+3) the position of the hunter with respect to the prey in relative cartesian coordinates. If the hunter can somehow navigate around and arrive at (0,0) in this relative coordinate system then we have the hunter coinciding with the prey on the same grid square (also known as the hunter capturing the prey!).

<p align="center">
<img src="https://github.com/ankonzoid/Deep-Reinforcement-Learning-Tutorials/blob/master/hunterprey/images/coverart.png" width="50%">
</p>

By introducing fixing the reference frame to the prey in our reinforcement learning, we no longer concern ourselves with the global earth coordinates simulation when computing the optimal policy — we not work solely in the relative reference frame which essentially exchanges the need to keep track of the direct product S_{hunter} x S_{prey} global state space to a job of just keeping track of the S_{hunter, relative} state space which holds relative positions of the hunter from the prey. Doing so not only shrinks our state search space, but it also eliminates much redundancy in training an agent on physical situations that are translationally equivalent i.e. teaching a hunter at (2,4) to catch a prey at (1,1) now also teaches the hunter at (3,5) to catch a prey at (2,2) because of the equivalence in relative positions of (+1,+3) of the hunter from the prey.

Implementation-wise if we start with a global grid of size (N,N), then the corresponding relative position state space we need to work in is an enlarged (2N-1, 2N-1) sized grid with values of [-N+1,N-1] x [-N+1,N-1] to account for all the possible relative positions the hunter can have from the prey. This is naturally the case as the maximum separation between any 2 grid points on the global grid is N grid squares in any direction, so the full dimensionality is 2*N-1 grid squares in both x and y directions (the -1 here comes from the shared zero distance). This reasoning is our bread-and-butter and allows us to simplify our seemingly double-agent system to single-agent system Grid World problem of searching for (0,0) on a [-N+1,N-1] x [-N+1,N-1] grid!

### Usage:

> python hunterprey.py

### Libraries required:

* numpy
* ffmpy (`pip3 install MoviePy`)
