# Training an Agent to beat GridWorld using Deep Q-Networks (`gridworld_DQN.py`)

We train an agent to beat Grid World using deep Q-networks (keras). This is scalable neural network upgrade to classical tabular methods as used in our previous Grid World examples:

* [Training an Agent to beat GridWorld (`gridworld.py`)](https://github.com/ankonzoid/Deep-Reinforcement-Learning-Tutorials/blob/master/gridworld)

* [Solving the Hunter-Prey problem as a single-agent problem using relative coordinates (`hunterprey.py`)](https://github.com/ankonzoid/Deep-Reinforcement-Learning-Tutorials/blob/master/hunterprey)

In deep Q-network methods, the Q-network is usually constructed in one of 2 ways:
 
 * A neural network that takes in state *s* and action *a*, and outputs a single scalar value that is Q(*s*,*a*)
 
 * A neural network that takes in a state *s*, and outputs a vector with values Q(*s*,*a*) for each action *a* (this is the implmentation that DeepMind used for their beating Atari games) 
 
 We will use the 2nd method of feed-forwarding a state *s* to the deep Q-network (DQN) to get a vector of Q(*s*,:) values (very similar in how policy networks work where Q(s) values replaces pi(s) probabilities here). The DQN weights are the generalization of the Q-tabular matrix values with the scalability of the policy network coming from treating the problem as a regression problem and using the pattern learning abilities of neural networks. The downsides of deep policy networks however are the ones that typically come with training a neural network *i.e. you have to choose the right architectures, losses, optimizers, training samples, etc.*

The available agents in this code are:

* *Epsilon-greedy selection* which chooses a random action with epsilon probability, or acts greedily otherwise by choosing action with highest Q (`"policy_mode": "epsilongreedy"`) 

### Example output:

For grid size of (20, 20) set by `env_info = {"Ny": 20, "Nx": 20}`, we train the agent to find move from the top-left corner to the bottom-right corner in 1000 training episodes with a exponentially-decaying epsilon exploration parameter. 

For epsilon-greedy selection, we get (more episodes might be needed for convergence):

```
[episode 0] mode = epsgreedy, iter = 1594, eps = 1.0000, reward = 84.07
[episode 1] mode = epsgreedy, iter = 666, eps = 0.9772, reward = 93.35
[episode 2] mode = epsgreedy, iter = 4514, eps = 0.9550, reward = 54.87
[episode 3] mode = epsgreedy, iter = 2220, eps = 0.9333, reward = 77.81
...
...
...
[episode 196] mode = epsgreedy, iter = 38, eps = 0.0110, reward = 99.63
[episode 197] mode = epsgreedy, iter = 40, eps = 0.0107, reward = 99.61
[episode 198] mode = epsgreedy, iter = 38, eps = 0.0105, reward = 99.63
[episode 199] mode = epsgreedy, iter = 38, eps = 0.0102, reward = 99.63
```

Both policy modes end up giving the optimal number of actions (`iter`) in an episode of Ny + Nx - 1 = 38 for our (20, 20) grid square.

### Usage:

> python gridworld_DQN.py

### Libraries required:

* keras

* numpy
