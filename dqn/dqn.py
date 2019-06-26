import random
import torch

from dqn_utils import ReplayMemory, Transition

class DQN:
    def __init__(self, e):
        self.env = e
        self.memory_size = 2560
        self.batch_size = 128


        self.memory = ReplayMemory(self.memory_size)

    def train(self):

        total_episodes = 100
        max_steps = 10

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



    def act(self, state):
        return self.env.sample_action()

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