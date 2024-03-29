import numpy as np
import random

from collections import defaultdict

class mc_on:
    def __init__(self, e, params):
        self.env = e

    def train(self):

        episode_rewards = []
        opt_rewards = []
        
        state_size = self.env.state_size
        action_size = self.env.action_size

        self.qtable = np.zeros((state_size, action_size))

        state_space = self.env.state_space
        action_space = self.env.action_space

        for s, v in state_space.items():
            for a in action_space:
                out_of_bounds = self.env.bound_qtable(v, a)
                if out_of_bounds:
                    self.qtable[v, a] = -1000000

        returns_sum = defaultdict(float)
        returns_count = defaultdict(float)

        total_episodes = 200
        max_steps = 10
        gamma = 0.9

        epsilon = 1.0
        max_epsilon = 1.0
        min_epsilon = 0.05
        decay_rate = 0.1

        for episode in range(total_episodes):

            print(episode, end='\r')
            # print(epsilon, end='\r')
            state = self.env.reset()
            episode_results = []
            step = 0
            total_reward = 0

            for step in range(max_steps):

                exp_exp_tradeoff = random.uniform(0,1)

                if exp_exp_tradeoff > epsilon:
                    action = np.argmax(self.qtable[state,:])
                else:
                    action = self.env.sample_action()

                new_state, reward, done = self.env.step(action)

                episode_results.append((state, action, reward))
                state = new_state
                total_reward += reward

                if done:
                    break

            episode_rewards.append(total_reward)
            sa_in_episode = set([(s, a) for s, a, _ in episode_results])
            
            for state, action in sa_in_episode:
                sa_pair = (state, action)

                first_idx = next(i for i, x in enumerate(episode_results) if x[0] == state and x[1] == action)
                G = sum([x[2] * (gamma ** i) for i, x in enumerate(episode_results[first_idx:])])

                returns_sum[sa_pair] += G
                returns_count[sa_pair] += 1.0
                self.qtable[state][action] = returns_sum[sa_pair] / returns_count[sa_pair]

            epsilon = min_epsilon + (max_epsilon - min_epsilon) * np.exp(-decay_rate * episode)
            opt_rewards.append(self.opt_test())


        self.env.reset()
        print()
        # return episode_rewards
        return opt_rewards

    def act(self, state):
        return np.argmax(self.qtable[state,:])

    def opt_test(self):
        state = self.env.reset()
        total_reward = 0
        steps = 10

        for step in range(steps):
            action = self.act(state)
            new_state, reward, done = self.env.step(action)

            state = new_state
            total_reward += reward

            if done:
                break

        return total_reward