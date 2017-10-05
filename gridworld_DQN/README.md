# Training an Agent to beat GridWorld using Deep Q-Networks (`gridworld_DQN.py`)

We train an agent to beat Grid World using a deep Q-network (DQN) with the keras library. This is neural network Q-approximation is a scalable upgrade of classical tabular methods as used in our previous Grid World examples:

* [Training an Agent to beat GridWorld (`gridworld.py`)](https://github.com/ankonzoid/Deep-Reinforcement-Learning-Tutorials/blob/master/gridworld)

* [Solving the Hunter-Prey problem as a single-agent problem using relative coordinates (`hunterprey.py`)](https://github.com/ankonzoid/Deep-Reinforcement-Learning-Tutorials/blob/master/hunterprey)

The difference between DQNs and Q-tabules is in the framework. 

In tabular methods, the estimated Q values are stored in a 2D matrix of size (N_state, N_actions) with each Q(*s*,*a*) entry representing the expected discounted total reward to be obtained in the future by following action *a* from state *s* then following optimal policy. One of the advantages of treating Q as a table is that it is simple to understand and update. You can plug your state and action indices to retrieve your Q(*s*,*a*) value in O(1) time, and greedy action selections can come from calculating argmax(Q(*s*,:)). Unfortunately tabular methods suffer the curse of dimensionality as higher-dimensional state spaces easily make the number of states, N_states, too large to hold in memory. For example a particle with 6-dimensions (x, y, z, vx, vy, vz) and 100 discrete values in each dimension gives for N_state = 10^12 unique states without considering the action space -- this is too large for any practical application and is too sparse to explore. Moreover when the state space is continuous then we definitely need an alternative to tabular methods.

In DQN methods, the estimated Q values come out as output from a neural network (NN) via the feed-forwarding of a state and action. The weights are now what hold information about how to convert the inpute state and action into a Q value, with the added functionality of treating Q as regression problem and being able to compute Q(*s*,*a*) for states *s* and actions *a* not visited before (being able generalized to unseen data is one of the greatest utilities of neural networks). Also we can easily scale DQN methods to much bigger problems because of how the neural network can learn patterns. However, the downsides of DQN methods are often the downsides of training any neural network: you have to choose appropriate architectures, losses, optimizers, etc.

We provide two alternative agent policies of *epsilon-greedy selection* (`"policy_mode": "epsilongreedy"`) and *softmax selection* (`"policy_mode": "softmax"`). For a given state *s*, the:

* Epsilon-greedy agent chooses a random action at epsilon probability, otherwise it acts greedily by choosing action argmax{a} Q(s,a).

* Boltzmann (softmax) agent samples for an action *a* from the probabilities proportional to the values of Q(s,a). 

### Example output:

For grid size of (20, 20) set by `env_info = {"Ny": 20, "Nx": 20}`, we train the agent to find move from the top-left corner to the bottom-right corner in 1000 training episodes with a exponentially-decaying epsilon exploration parameter. The initial training was:
```
[episode 0] iter = 6710, epsilon = 1.0000, reward = 32.91
[episode 1] iter = 2812, epsilon = 0.9954, reward = 71.89
[episode 2] iter = 2178, epsilon = 0.9908, reward = 78.23
[episode 3] iter = 932, epsilon = 0.9863, reward = 90.69
[episode 4] iter = 572, epsilon = 0.9817, reward = 94.29
[episode 5] iter = 1300, epsilon = 0.9772, reward = 87.01
[episode 6] iter = 772, epsilon = 0.9727, reward = 92.29
...
```

and the final output was:

```
[episode 994] iter = 38, epsilon = 0.0103, reward = 99.63
[episode 995] iter = 38, epsilon = 0.0102, reward = 99.63
[episode 996] iter = 38, epsilon = 0.0102, reward = 99.63
[episode 997] iter = 38, epsilon = 0.0101, reward = 99.63
[episode 998] iter = 38, epsilon = 0.0101, reward = 99.61
[episode 999] iter = 38, epsilon = 0.0100, reward = 99.61
```

Here you can see a optimal policy found as the optimal number of actions (`iter`) in an episode is expected to be Ny + Nx - 1 = 38 for our (20, 20) grid square.

### Usage:

> python gridworld_DQN.py

### Libraries required:

* keras

* numpy
