"""

 blackjack.py  (author: Anson Wong / git: ankonzoid)

 Solving the game of blackjack (no-usable ace, dealing with replacement).
 The learning method is Monte Carlo Exploring Starts with reward sample averaging. 
 Refer to Chapter 5 (Monte Carlo Methods) of Richard Sutton's "Reinforcement Learning: 
 An Introducion".

 ============================================================
 
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

 ============================================================

"""
import random, pickle
import numpy as np
from collections import deque

class Environment:
    
    def __init__(self):
        # Starts are parametrized efficiently with (s_player, s_dealer) where:
        #  s_player = player card sum in range [12, 21] inclusive
        #             (lower sums can be hit consecutively until it fits into this range)
        #  s_dealer = dealer showing card value in the range of [1, 10] for non-usable ace
        self.state_dim = (10, 10)  # (s_player, s_dealer)
        self.action_dim = (2,)  # 0=stick, 1=hit

    def reset(self):
        # Create a fresh deck of 52 cards (we will randomly sample this deck with replacement)
        SUITS = ['diamond', 'club', 'heart', 'spade']
        RANKS = ['A', '2', '3', '4', '5', '6', '7', '8', '9', '10', 'J', 'Q', 'K']
        self.DECK = list()
        for suit in SUITS:
            for rank in RANKS:
                self.DECK.append({"rank": rank, "suit": suit})

        # Note: it is important that we sample all possible (state,action) pairs 
        # or you might run into the problem of not reaching convergence in Q.
        self.state = (np.random.randint(self.state_dim[0]), 
                      np.random.randint(self.state_dim[1]))
        return self.state

    def step(self, action):

        def deal_card(DECK):
            card = random.choice(DECK)  # deal card with replacement
            if card["rank"] in ["A"]:
                return 1  # non-usable ace
            elif card["rank"] in ["J", "Q", "K"]:
                return 10  # face card
            else:
                return int(card["rank"])

        # Perform player action (0 = stick, 1 = hit)
        if action == 1:
            self.state = (self.state[0] + deal_card(self.DECK), self.state[1])

        # Compute card sums
        player_card_sum = self.state[0] + 12
        dealer_card_sum = self.state[1] + 1

        # Receive (stochastic) reward and decide if we hit terminal state
        # Terminate if player bust
        if action == 1 and player_card_sum <= 21:
            # player hit and not bust
            reward, done = 0, False
        elif action == 1 and player_card_sum > 21:
            # player hit and bust
            reward, done = -1, True
        else:
            # player stick (terminal state) -> now let dealer follow house rules
            reward, done = 0, True
            while True:
                if dealer_card_sum > 21:
                    # dealer bust
                    reward = 1
                    break
                elif dealer_card_sum >= 17:
                    # if dealer card sum >= 17, compare hands with player
                    if player_card_sum > dealer_card_sum:
                        # player wins
                        reward = 1
                    elif player_card_sum < dealer_card_sum:
                        # dealer wins
                        reward = -1
                    else:
                        # tie
                        reward = 0
                    break
                else:
                    # dealer must deal a card (via house rules) if sum < 17
                    dealer_card_sum += deal_card(self.DECK)

        return self.state, reward, done

class Agent:
    
    def __init__(self, state_dim, action_dim):
        self.state_dim = state_dim  # state dimension
        self.action_dim = action_dim  # action dimension
        self.epsilon = 0.05  # epsilon exploration probability
        self.reset_learning_memory()

    def reset_episodic_memory(self):
        # Reset episodic memories
        self.memories = list()

    def reset_learning_memory(self):
        # Reset Q[s,a], r_visits[s,a], n_visits[s,a] to zero
        self.Q = np.zeros(self.state_dim + self.action_dim, dtype=float)  # Q(s,a) value
        self.returns_visits = np.zeros(self.state_dim + self.action_dim, dtype=float)  # 3-dim matrix
        self.n_visits = np.zeros(self.state_dim + self.action_dim, dtype=int)  # 3-dim matrix

    def get_action(self, state, force_random=False):
        if random.uniform(0, 1) < self.epsilon or force_random:
            # explore
            action = np.random.randint(self.action_dim[0])
        else:
            # greedy
            action = np.argmax(self.Q[state[0], state[1], :])
        return action

    def train(self):
        # List all uniquely visited (state, action) pairs from episodic memory.
        # Also compute the total reward collected from the episode.
        # Returns after first-occurence of (s,a)
        returns_episode = {}
        for i, memory_i in enumerate(self.memories):
            state, action, state_next, reward, done = memory_i
            sa = state + (action,)
            if sa not in returns_episode.keys():
                returns_episode[sa] = 0.0
                for memory_j in self.memories[i:]:
                    _, _, _, reward, _ = memory_j
                    returns_episode[sa] += reward

        # Update Q[s,a] (using reward sample averaging) on all UNIQUE (s,a) pairs visited
        for memory in self.memories:
            state, action, state_next, reward, done = memory
            sa = state + (action,)  # (s,a) pair = (s_player, s_dealer, action)
            self.returns_visits[sa] += returns_episode[sa]  # add to reward total for (s,a)
            self.n_visits[sa] += 1  # iterate visit count for (s,a)
            self.Q[sa] = self.returns_visits[sa] / self.n_visits[sa]  # update Q(s,a)

    def memorize(self, memory):
        self.memories.append(memory)
    
    def display_greedy_policy(self):
        # Display greedy policy:
        #  - rows are s_player
        #  - columns are s_dealer
        print("\nDisplaying greedy policy:")
        np.set_printoptions(precision=3)
        greedy_policy = np.zeros(self.state_dim, dtype=int)
        for s_player in range(self.state_dim[0]):
            for s_dealer in range(self.state_dim[1]):
                greedy_policy[s_player, s_dealer] = np.argmax(self.Q[s_player, s_dealer, :])
        print(greedy_policy)
        print()

# ===================
# Driver
# ===================
if __name__ == "__main__":
    env = Environment()  # initialize discrete state, discrete action blackjack environment
    agent = Agent(env.state_dim, env.action_dim)  # initialize reward-averaging agent

    print("Training blackjack agent...")
    N_episodes = 5000000  # number of blackjack games to play
    for episode in range(N_episodes):
        # Generate an episode
        agent.reset_episodic_memory()  # reset agent episodic memory
        state = env.reset()  # initialize random (s_player, s_dealer) state
        i = 0
        while True:
            action = agent.get_action(state, force_random=(i==0))  # get action from policy
            state_next, reward, done = env.step(action)  # evolve state
            agent.memorize((state, action, state_next, reward, done))  # memorize evolution result
            if done:
                break  # finish episode if we hit a terminal state
            state = state_next  # otherwise transition to next state
            i += 1  # iter

        # Train agent using reward averaging on its visited (state, action) pairs 
        agent.train()

        # Print training progress
        if (episode + 1) % int(N_episodes / 100) == 0:
            print("[episode = {}/{}]".format(episode + 1, N_episodes))
            agent.display_greedy_policy()
