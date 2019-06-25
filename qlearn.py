import numpy as np
import random

class qlearner:
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

        total_episodes = 1000
        max_steps = 50
        learning_rate = 0.7
        gamma = 0.9

        epsilon = 1.0
        max_epsilon = 1.0
        min_epsilon = 0.1
        decay_rate = 0.01

        for episode in range(total_episodes):

            print(episode, end='\r')

            state = self.env.reset()
            done = False
            reward = 0
            step = 0

            for step in range(max_steps):

                exp_exp_tradeoff = random.uniform(0,1)

                if exp_exp_tradeoff > epsilon:
                    action = np.argmax(self.qtable[state,:])
                else:
                    action = self.env.sample_action()

                new_state, reward, done = self.env.step(action)

                self.qtable[state, action] = self.qtable[state, action] + learning_rate * (reward + gamma * np.max(self.qtable[new_state,:]) - self.qtable[state, action])

                state = new_state

                if done:
                    break

            epsilon = min_epsilon + (max_epsilon - min_epsilon) * np.exp(-decay_rate * episode)

        self.env.reset()
        print()

    def act(self, state):
        return np.argmax(self.qtable[state,:])