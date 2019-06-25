import numpy as np
import random

from collections import defaultdict

class mclearner:
    def __init__(self, e):
        self.env = e

    def learn(self):
        
        state_size = self.env.state_size
        action_size = self.env.action_size

        self.qtable = np.zeros((state_size, action_size))

        state_space = self.env.state_space
        action_space = self.env.action_space

        for s in state_space:
            for a in action_space:
                out_of_bounds = self.env.bound_qtable(s, a)
                if out_of_bounds:
                    self.qtable[s, a] = -10000

        returns_sum = defaultdict(float)
        returns_count = defaultdict(float)

        total_episodes = 100
        max_steps = 10
        gamma = 0.9

        epsilon = 1.0
        max_epsilon = 1.0
        min_epsilon = 0.1
        decay_rate = 0.0001

        for episode in range(total_episodes):

            print(episode, end='\r')
            state = self.env.reset()
            episode_results = []
            step = 0

            for step in range(max_steps):

                exp_exp_tradeoff = random.uniform(0,1)

                if exp_exp_tradeoff > epsilon:
                    action = np.argmax(self.qtable[state,:])
                else:
                    action = self.env.sample_action()

                new_state, reward, done = self.env.step(action)

                episode_results.append((state, action, reward))
                state = new_state

                if done:
                    break

            sa_in_episode = set([(s, a) for s, a, _ in episode_results])
            
            for state, action in sa_in_episode:
                sa_pair = (state, action)

                first_idx = next(i for i, x in enumerate(episode_results) if x[0] == state and x[1] == action)
                G = sum([x[2] * (gamma ** i) for i, x in enumerate(episode_results[first_idx:])])

                returns_sum[sa_pair] += G
                returns_count[sa_pair] += 1.0
                self.qtable[state][action] = returns_sum[sa_pair] / returns_count[sa_pair]

            epsilon = min_epsilon + (max_epsilon - min_epsilon) * np.exp(-decay_rate * episode)

        self.env.reset()
        print()

    def act(self, state):
        return np.argmax(self.qtable[state,:])