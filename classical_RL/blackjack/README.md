# Blackjack

 Finds optimal blackjack policy for a 52-count card deck under house rules. The learning algorithm is Monte Carlo exploring starts with reward sample averaging. Refer to Chapter 5 (Monte Carlo Methods) of Richard Sutton's "Reinforcement Learning: An Introduction" textbook for more details about this problem.

<p align="center">
<img src="images/coverart.png" width="60%">
</p>

The algorithm used in the code:

```
 Monte Carlo ES (Exploring Starts), for estimating π ≈ π∗
 
 Initialize, for all s ∈ S, a ∈ A(s): 
   Q(s, a) ← arbitrary
   π(s) ← arbitrary
   Returns(s, a) ← empty list
 
 Repeat forever:
   Choose S0 ∈ S and A0 ∈ A(S0) s.t. all pairs have probability > 0 
   Generate an episode starting from S0, A0, following π
   For each pair s, a appearing in the episode:
     G ← the return that follows the first occurrence of s, a 
     Append G to Returns(s, a)
     Q(s, a) ← average(Returns(s, a))
   
   For each s in the episode: 
     π(s) ← argmax[a'] Q(s, a')
```

### Usage

Run

> python3 blackjack.py

### Output

Displays the greedy policy where `0` represents stick, and `1` represents hit. The y-axis corresponds to the player sum starting from 12 -> 21 (index 0 -> 9), and the x-axis corresponds to the value of card the dealer is revealing from 1 -> 10 (index 0 -> 9; ace is treated as non-usable, and Jack, Queen, King count as value 10).

```
Training blackjack agent...
[episode = 20000/2000000]

Displaying greedy policy:
[[1 0 0 0 1 0 1 1 1 1]
 [0 0 0 0 0 0 1 1 1 1]
 [0 0 0 0 0 0 0 1 1 1]
 [0 0 0 0 0 0 0 1 0 1]
 [0 0 0 0 0 0 0 0 1 0]
 [0 0 0 0 0 0 0 0 0 0]
 [0 0 0 0 0 0 0 0 0 0]
 [0 0 0 0 0 0 0 0 0 0]
 [0 0 0 0 0 0 0 0 0 0]
 [0 0 0 0 0 0 0 0 0 0]]

 ...
 ...
 ...

[episode = 2000000/2000000]

Displaying greedy policy:
[[1 1 0 0 0 0 1 1 1 1]
 [1 0 0 0 0 0 1 1 1 1]
 [0 0 0 0 0 0 1 1 1 1]
 [0 0 0 0 0 0 1 1 1 1]
 [0 0 0 0 0 0 1 1 1 0]
 [0 0 0 0 0 0 0 0 0 0]
 [0 0 0 0 0 0 0 0 0 0]
 [0 0 0 0 0 0 0 0 0 0]
 [0 0 0 0 0 0 0 0 0 0]
 [0 0 0 0 0 0 0 0 0 0]]
```

### Author

Anson Wong
