import gym
import numpy as np
from gym import spaces
from gym.utils import seeding

#construction of blackjack env.
class BlackjackEnv(gym.Env):
    def __init__(self):
        #actions: hit or stick
        self.action_space = spaces.Discrete(2)
        #observation: player's points,
        #   shown dealer point, usable ace
        self.observation_space = spaces.Tuple((
            spaces.Discrete(32),
            spaces.Discrete(11),
            spaces.Discrete(2)))
        #initialize random seed
        self._seed()
        self._reset()
        self.nA = 2

    #set seed
    #draw card with np_random, which is related to seed
    def _seed(self, seed = None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def reset(self):
        return self._reset()

    #initialize for player and dealer
    #return initial state
    def _reset(self):
        self.dealer = draw_hand(self.np_random)
        self.player = draw_hand(self.np_random)
        while sum_hand(self.player) < 12:
            self.player.append(draw_card(self.np_random))
        return self._get_obs()

    def step(self, action):
        return self._step(action)

    #take action
    def _step(self, action):
        #if action_space has no action
        #   rasie error and stop next actions
        assert self.action_space.contains(action)
        if action:
            self.player.append(draw_card(self.np_random))
            if is_bust(self.player):
                done = True
                reward = -1
            else:
                done = False
                reward = 0
        else:
            done = True
            while sum_hand(self.dealer) < 17:
                self.dealer.append(draw_card(self.np_random))
            reward = cmp(score(self.player), score(self.dealer))
        #return: state, reward, done, info
        return self._get_obs(), reward, done, {}

    #observation(state)
    def _get_obs(self):
        return (sum_hand(self.player), self.dealer[0], usable_ace(self.player))

#basic functions for Env.
def draw_card(np_random):
    deck = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 10, 10, 10]
    return np_random.choice(deck)

def draw_hand(np_random):
    return [draw_card(np_random), draw_card(np_random)]

def usable_ace(hand):
    return 1 in hand and (sum(hand) + 10) <= 21

def sum_hand(hand):
    return sum(hand) + 10 if usable_ace(hand) else sum(hand)

def is_bust(hand):
    return sum_hand(hand) > 21

def score(hand):
    return 0 if is_bust(hand) else sum_hand(hand)

def cmp(a, b):
    return int(a > b) - int(a < b)

#test for blackjack env
def blackjack_env_test():
    env = BlackjackEnv()
    def print_observation(observation):
        score, dealer_score, usable_ace = observation
        print("Player Score: {} (Usable Ace: {}), Dealer Score: {}".format(
              score, usable_ace, dealer_score))

    def strategy(observation):
        score, dealer_score, usable_ace = observation
        # Stick (action 0) if the score is > 20, hit (action 1) otherwise
        return 0 if score >= 20 else 1

    for i_episode in range(20):
        observation = env.reset()
        for t in range(100):
            print_observation(observation)
            action = strategy(observation)
            print("Taking action: {}".format( ["Stick", "Hit"][action]))
            observation, reward, done, _ = env.step(action)
            if done:
                print_observation(observation)
                print("Game end. Reward: {}\n".format(float(reward)))
                break

def main():
    blackjack_env_test()

if __name__ == '__main__':
    main()