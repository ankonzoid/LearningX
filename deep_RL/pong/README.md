# Pong

Trains a Pong agent using policy gradients on OpenAI's gym. This code was copied from Andrej Karpathy's [Deep Reinforcement Learning: Pong from Pixels](http://karpathy.github.io/2016/05/31/rl/), and almost all changes to the code were for cosmetic purposes. Please refere to Karpathy's walkthrough to learn more about the implementation!

<p align="center">
<img src="images/pong.gif" width="45%">
</p>

### Usage

> python3 pong.py

Set `resume = True` in `pong.py` if you want to continue training the agent where it was left off in `model.p`, otherwise set `resume = False` to start the agent training from scratch.

### Output

```
Resuming model 'model.p'...
ep 0: game finished, reward: 1.000000
ep 0: game finished, reward: 1.000000
ep 0: game finished, reward: 1.000000
ep 0: game finished, reward: -1.000000
ep 0: game finished, reward: 1.000000
ep 0: game finished, reward: -1.000000
...
...
...
```
