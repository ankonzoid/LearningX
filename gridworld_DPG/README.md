# Training an Agent to beat GridWorld using Deep Policy Gradients (`gridworld_DPG.py`)

We train an agent to beat Grid World using deep policy gradients with the keras library. This is scalable neural network upgrade to classical tabular methods as used in our previous Grid World examples:

* [Training an Agent to beat GridWorld (`gridworld.py`)](https://github.com/ankonzoid/Deep-Reinforcement-Learning-Tutorials/blob/master/gridworld)

* [Solving the Hunter-Prey problem as a single-agent problem using relative coordinates (`hunterprey.py`)](https://github.com/ankonzoid/Deep-Reinforcement-Learning-Tutorials/blob/master/hunterprey)

In Q-tabular methods, the estimated Q values are stored in a 2D matrix of size (N_state, N_actions) with each Q(*s*,*a*) entry representing the expected discounted total reward to be obtained in the future by following action *a* from state *s* then following optimal policy. One of the advantages of treating Q as a table is that it is simple to understand and update. You can plug your state and action indices to retrieve your Q(*s*,*a*) value in O(1) time, and greedy action selections can come from calculating argmax(Q(*s*,:)). Unfortunately tabular methods suffer the curse of dimensionality as higher-dimensional state spaces easily make the number of states, N_states, too large to hold in memory. For example a particle with 6-dimensions (x, y, z, vx, vy, vz) and 100 discrete values in each dimension gives for N_state = 10^12 unique states without considering the action space -- this is too large for any practical application and is too sparse to explore. Moreover when the state space is continuous then we definitely need an alternative to tabular methods.

In deep policy network methods, the estimated action probabilities values come out as output from a neural network (NN) via the feed-forwarding of a state. The weights of the NN hold information of how to convert the input state into a list of softmax values corresponding to how likely an agent should take an action. Why policy gradients to tabular methods makes for a scalable option is because of the interpolational and pattern learning aspects of computing policies for "nearby" but unseen state-spaces. However the downsides of deep policy networks are the ones that come with training a neural network *i.e. you have to choose appropriate architectures, losses, optimizers, etc.*

We provide two alternative agent policy selection modes of *epsilon-greedy selection* (`"policy_mode": "epsilongreedy"`) and *softmax selection* (`"policy_mode": "softmax"`). For a given state *s*, the:

* Epsilon-greedy agent chooses a random action at epsilon probability, otherwise it acts greedily by choosing action argmax{a} Q(s,a).

* Boltzmann (softmax) agent samples for an action *a* from the probabilities proportional to the values of Q(s,a). 

### Example output:

For grid size of (20, 20) set by `env_info = {"Ny": 20, "Nx": 20}`, we train the agent to find move from the top-left corner to the bottom-right corner in 1000 training episodes with a exponentially-decaying epsilon exploration parameter. 

For epsilon-greedy selection:

```
[episode 0] mode = epsilongreedy, iter = 6710, epsilon = 1.0000, reward = 32.91
[episode 1] mode = epsilongreedy, iter = 2812, epsilon = 0.9954, reward = 71.89
[episode 2] mode = epsilongreedy, iter = 2178, epsilon = 0.9908, reward = 78.23
[episode 3] mode = epsilongreedy, iter = 932, epsilon = 0.9863, reward = 90.69
...
...
...
[episode 997] mode = epsilongreedy, iter = 38, epsilon = 0.0101, reward = 99.63
[episode 998] mode = epsilongreedy, iter = 38, epsilon = 0.0101, reward = 99.63
[episode 999] mode = epsilongreedy, iter = 38, epsilon = 0.0100, reward = 99.63
```

For softmax selection (it converges to optimal early because it is adaptively greedy):

```
[episode 0] mode = softmax, iter = 9868, reward = 1.33
[episode 1] mode = softmax, iter = 4208, reward = 57.93
[episode 2] mode = softmax, iter = 2488, reward = 75.13
[episode 3] mode = softmax, iter = 572, reward = 94.29
...
...
...
[episode 129] mode = softmax, iter = 38, reward = 99.63
[episode 130] mode = softmax, iter = 38, reward = 99.63
[episode 131] mode = softmax, iter = 38, reward = 99.63
...
...
...
```

Both policy modes end up giving the optimal number of actions (`iter`) in an episode of Ny + Nx - 1 = 38 for our (20, 20) grid square.

### Usage:

> python gridworld_DPG.py

### Libraries required:

* keras

* numpy
