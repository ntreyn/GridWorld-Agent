import random
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np

from .dqn_utils import ReplayMemory, Transition, NeuralNet

class DQN:
    def __init__(self, e, params):
        self.env = e
        self.memory_size = 2560
        self.batch_size = 128
        self.clip = 10
        self.cur_step = 0
        self.episode = 0
        self.training = True

        self.input_size = 10
        self.hidden_size = 128
        self.output_size = self.env.action_size


        if params.device == 'cpu' or params.device == 'cuda':
            self.device = torch.device(params.device)
        else:
            print("Invalid device: default to cpu")
            self.device = torch.device('cpu')

        dqn_net = NeuralNet(self.input_size, self.hidden_size, self.output_size)
        self.policy_net = dqn_net.to(self.device)
        self.target_net = dqn_net.to(self.device)

        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.policy_net.parameters())

        self.memory = ReplayMemory(self.memory_size)

    def train(self):

        opt_rewards = []

        total_episodes = 1000
        max_steps = 20
        self.gamma = 0.9

        self.epsilon = 1.0
        max_epsilon = 1.0
        min_epsilon = 0.2
        decay_rate = 1000

        for episode in range(total_episodes):
            
            print(episode, end='\r')
            # print(self.epsilon, end='\r')
            state = self.env.reset()

            for step in range(max_steps):
     
                action = self.act(state)

                next_state, reward, done = self.env.step(action)

                self.push(state, action, reward, next_state, done)

                state = next_state

                self.learn(episode)

                if done:
                    break
            
            self.epsilon = min_epsilon + (max_epsilon - min_epsilon) * np.exp(-1.0 * self.cur_step / decay_rate)
            opt_rewards.append(self.greedy_test())

        return opt_rewards

    def act(self, state):
        state = self.state_to_tensor(state)
        exp_exp_tradeoff = random.uniform(0,1)

        if self.training:
            self.cur_step += 1
        
        if self.training and exp_exp_tradeoff < self.epsilon:
            action = self.env.sample_action()
        else:
            self.policy_net.eval()
            with torch.no_grad():
                policy_out = self.policy_net(state.to(self.device))
            self.policy_net.train()
            action = policy_out.max(1)[1].view(1, 1).item()
        
        return action

    def learn(self, episode):
        self.episode = episode

        if self.memory.size() <= self.batch_size:
            return

        batch = self.sample()

        non_final_mask = torch.tensor(tuple([s is not None for s in batch.next_state]),
                                        device=self.device, dtype=torch.uint8)
        non_final_next_states = torch.cat([s for s in batch.next_state if s is not None])

        state_batch = torch.cat(batch.state)

        state_batch = state_batch.to(self.device)
        non_final_next_states = non_final_next_states.to(self.device)

        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward)

        state_q_vals = self.policy_net(state_batch).gather(1, action_batch)

        next_state_vals = torch.zeros(self.batch_size, device=self.device)
        next_state_vals[non_final_mask] = self.target_net(non_final_next_states).max(1)[0].detach()

        expected_state_q_values = (next_state_vals * self.gamma) + reward_batch

        loss = F.mse_loss(state_q_vals, expected_state_q_values.unsqueeze(1))

        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.policy_net.parameters(), self.clip)
        self.optimizer.step()

        if self.cur_step % 2 == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())

    def push(self, *args):
        state = self.state_to_tensor(args[0])
        action = torch.tensor([[args[1]]])
        reward = torch.tensor([args[2]], dtype=torch.float)
        next_state = self.state_to_tensor(args[3])
        done = torch.tensor([args[4]])

        if done:
            next_state = None

        self.memory.push(state, action, reward, next_state, done)
    
    def eval_on(self):
        self.training = False

    def eval_off(self):
        self.training = True

    def sample(self):
        transitions = self.memory.sample(self.batch_size)
        batch = Transition(*list(zip(*transitions)))
        return batch

    def state_to_tensor(self, state):
        state_list = []

        state_tuple = self.env.reverse_state_space[state]
        board_state_tuple = state_tuple[0]
        location_int = state_tuple[1]

        for tile in board_state_tuple:
            state_list.append(tile)
        
        state_list.append(location_int)
        state_tensor = torch.tensor([state_list], dtype=torch.float)
        return state_tensor
    
    def greedy_test(self):
        self.eval_on()
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

        self.eval_off()
        return total_reward
