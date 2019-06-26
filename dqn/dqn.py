import random
import torch
import numpy as np

from dqn_utils import ReplayMemory, Transition

class DQN:
    def __init__(self, e):
        self.env = e
        self.memory_size = 2560
        self.batch_size = 128

        self.cur_step = 0


        self.memory = ReplayMemory(self.memory_size)

    def train(self):

        total_episodes = 100
        max_steps = 10

        self.epsilon = 1.0
        max_epsilon = 1.0
        min_epsilon = 0.05
        decay_rate = 0.1

        for episode in range(total_episodes):

            state = self.env.reset()

            for step in range(max_steps):

                action = self.act(state)

                next_state, reward, done = self.env.step(action)

                if done:
                    next_state = None

                self.push(state, action, reward, next_state, done)

                state = next_state

                self.learn()

                if done:
                    break
            
            self.epsilon = min_epsilon + (max_epsilon - min_epsilon) * np.exp(-1.0 * self.cur_step / decay_rate)



    def act(self, state):

        exp_exp_tradeoff = random.uniform(0,1)

        if exp_exp_tradeoff > self.epsilon:
            action = 0
        else:
            action = self.env.sample_action()
        
        self.cur_step += 1

        return action

    def learn(self):

        if self.memory.size() <= self.batch_size:
            return

        batch = self.sample()
        


    def push(self, *args):
        state = args[0]
        action = args[1]
        reward = args[2]
        next_state = args[3]
        done = args[4]

        if done:
            next_state = None
        self.memory.push(state, action, reward, next_state, done)
    
    def sample(self):
        transitions = self.memory.sample(self.batch_size)
        batch = Transition(*list(zip(*transitions)))
        return batch