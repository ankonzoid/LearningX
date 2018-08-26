# Blackjack

 Finds optimal blackjack policy for a 52-count card deck under house rules. The learning algorithm is Monte Carlo exploring starts with reward sample averaging. Refer to Chapter 5 of Sutton on Monte Carlo methods for more details about this problem.

<p align="center">
<img src="images/coverart.png" width="60%">
</p>

### Usage

Run:

> python3 blackjack.py

### Output

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
