# Q-learning on OpenGym environments (`qlearning_opengym.py`)

OpenAI Gym is a toolkit filled with environments for you to train your agents on (Atari games, Pacman, simulated lunar landings, cartpole, etc). Each environment provides their own states, actions, and rewards.
 
 In this code, we provide an implementation of the Q-learning algorithm (in both tabular form, and as a neural network approximation) applied to `gym` environments for *discrete* states and *discrete* actions.

As will be described specifically in the usage section, we tested on the the following environments:

* [Taxi pick-up and drop-off (`Taxi-v2`)](https://gym.openai.com/envs/Taxi-v1)

* [Walking across a frozen lake (`FrozenLake-v0`)](https://gym.openai.com/envs/FrozenLake-v0)


### Usage:

For Q(s,a) in tabular form:

> python Qtabular_gym.py

For Q(s,a) as an NN approximation:

> python QNN_gym.py

For print text descriptions of available gym environments:

> python gym_env_descrip.py

To switch environments from our default `Taxi-v2` environment, modify the string variable `env_str` in `Qtabular_gym.py` and `QNN_gym.py` to any environment name listed above.


### Example output:

#### `Qtabular_gym.py`

For `Qtabular_gym.py`, we train our agent on the `Taxi-v2` environment for `n_episodes = 2000` with a learning rate `beta = 0.8` and discount factor `gamma = 0.95` to get an output of

```swift
Average iterations:  [81.835, 17.085, 13.54, 13.38, 13.005, 12.965, 12.76, 12.95, 12.885, 13.16]
Average rewards:  [-99.235, 2.205, 6.83, 7.26, 7.815, 8.035, 8.24, 8.05, 8.115, 7.84]
Lowest/highest episodic reward: -686 / 15
```

with plots of actions and rewards per training episode

<img src="https://github.com/ankonzoid/Deep-NN-Python-Tutorials/blob/master/RL/qlearning_opengym/plots/Taxi-v2_Qtab_nactions.png" width="75%" align="center">

<img src="https://github.com/ankonzoid/Deep-NN-Python-Tutorials/blob/master/RL/qlearning_opengym/plots/Taxi-v2_Qtab_totreward.png" width="75%" align="center">

#### `QNN_gym.py`

For `QNN_gym.py`, we train our agent on the `Taxi-v2` environment for `n_episodes = 2000` with a discount factor `gamma = 0.99` and a decreasing exploration noise per episode *i* and policy of

```swift
exploration_noise = (1./(i+1))
a = np.argmax(Q[s,:] + np.random.randn(1,n_actions)*exploration_noise)
```

to get an output of

```swift
Average iterations:  [140.86, 50.34, 20.845, 14.77, 13.58, 13.52, 13.405, 13.235, 12.79, 12.82]
Average rewards:  [-213.355, -48.9, -4.12, 3.8, 5.755, 6.4, 6.155, 6.415, 7.67, 7.64]
Lowest/highest episodic reward: -677 / 15
```

with plots of actions and rewards per training episode

<img src="https://github.com/ankonzoid/Deep-NN-Python-Tutorials/blob/master/RL/qlearning_opengym/plots/Taxi-v2_QNN_nactions.png" width="75%" align="center">

<img src="https://github.com/ankonzoid/Deep-NN-Python-Tutorials/blob/master/RL/qlearning_opengym/plots/Taxi-v2_QNN_totreward.png" width="75%" align="center">

#### `gym_env_descrip.py`

A list of gym environment descriptions are given

```swift
...

e.id = FrozenLake-v0
env.observation_space = Discrete(16)
env.action_space = Discrete(4)
env.reward_range = (-inf, inf)
e.timestep_limit = 100
e.trials = 100
e.reward_threshold = 0.78

...

e.id = Taxi-v2
env.observation_space = Discrete(500)
env.action_space = Discrete(6)
env.reward_range = (-inf, inf)
e.timestep_limit = 200
e.trials = 100
e.reward_threshold = 8

...
```

