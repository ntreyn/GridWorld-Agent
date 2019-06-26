import numpy as np
import random

class qlearner:
    def __init__(self, e):
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

        total_episodes = 100
        max_steps = 10
        learning_rate = 0.7
        gamma = 0.9

        epsilon = 1.0
        max_epsilon = 1.0
        min_epsilon = 0.005
        decay_rate = 0.1

        for episode in range(total_episodes):

            print(episode, end='\r')
            # print(epsilon, end='\r')

            state = self.env.reset()
            done = False
            total_reward = 0
            step = 0

            for step in range(max_steps):

                exp_exp_tradeoff = random.uniform(0,1)

                if exp_exp_tradeoff > epsilon:
                    action = np.argmax(self.qtable[state,:])
                else:
                    action = self.env.sample_action()

                new_state, reward, done = self.env.step(action)
                total_reward += reward

                self.qtable[state, action] = self.qtable[state, action] + learning_rate * (reward + gamma * np.max(self.qtable[new_state,:]) - self.qtable[state, action])

                state = new_state

                if done:
                    break

            episode_rewards.append(total_reward)
            opt_rewards.append(self.opt_test())
            epsilon = min_epsilon + (max_epsilon - min_epsilon) * np.exp(-decay_rate * episode)

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