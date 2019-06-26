import random
import numpy as np


class mc_off:
    def __init__(self, e):
        self.env = e

    def train(self):

        opt_rewards = []

        state_size = self.env.state_size
        action_size = self.env.action_size

        self.qtable = np.zeros((state_size, action_size))
        C = np.zeros((state_size, action_size))

        state_space = self.env.state_space
        action_space = self.env.action_space

        for s, v in state_space.items():
            for a in action_space:
                out_of_bounds = self.env.bound_qtable(v, a)
                if out_of_bounds:
                    self.qtable[v, a] = -1000000

        total_episodes = 10000
        max_steps = 10
        gamma = 1.0

        for episode in range(total_episodes):

            print(episode, end='\r')
            episode_results = []
            state = self.env.reset()

            for step in range(max_steps):

                action = self.env.sample_action()
                new_state, reward, done = self.env.step(action)

                episode_results.append((state, action, reward))
                state = new_state

                if done:
                    break
            
            G = 0.0
            W = 1.0

            for step in range(len(episode_results))[::-1]:

                state, action, reward = episode_results[step]
                G = gamma * G + reward
                C[state][action] += W

                self.qtable[state, action] += (W / C[state, action]) * (G - self.qtable[state, action])

                if action != np.argmax(self.qtable[state,:]):
                    break

                sa_size = len(self.env.potential_moves(self.env.state_to_pos(state)))
                random_prob = 1.0 / float(sa_size)
                W = W * 1./random_prob
            
            opt_rewards.append(self.opt_test())

        self.env.reset()
        print()
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