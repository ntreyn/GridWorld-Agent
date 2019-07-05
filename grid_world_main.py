

from grid_world_env import grid_env
from qlearn import qlearner
from mc_on_policy import mc_on
from mc_off_policy import mc_off
from parameters import MODEL_MAP, core_argparser

import matplotlib.pyplot as plt
import argparse
import numpy as np

def human_play(env):

    while True:
        env.render()
        while True:
            moves = env.potential_moves(env.position)
            print(moves)

            action = int(input("Enter action: "))
            new_state, reward, done = env.step(action)

            env.render()
            state = new_state

            if done:
                env.render()
                env.reset()
                break
        
        temp = input("Would you like to play again? (y/n) ")
        if temp == 'n':
            break

def test_agent(env, agent):
    test_rewards = []
    agent.eval_on()

    for n in range(100):
        state = env.reset()
        total_reward = 0

        for s in range(10):
            action = agent.act(state)
            new_state, reward, done = env.step(action)

            state = new_state
            total_reward += reward

            if done:
                break
        
        test_rewards.append(total_reward)
    
    return test_rewards


def main(args):
    env = grid_env()

    episode_reward_list = []
    temp_list = []
    test_reward_list = []
    
    for n in range(10):
        agent = MODEL_MAP[args.model](env, args)

        episode_rewards = agent.train()
        print("Agent {} trained".format(n))

        temp = [r for r in episode_rewards if r > -100]

        episode_reward_list.append(episode_rewards)
        temp_list.append(temp)
        
        test_reward_list.append(test_agent(env, agent))
    
    for n in range(10):
        plt.plot(episode_reward_list[n])
        plt.show()

        plt.plot(temp_list[n])
        plt.show()
        
        plt.plot(test_reward_list[n])
        plt.show()

    


if __name__ == "__main__":
    ARGPARSER = argparse.ArgumentParser(parents=[core_argparser()])
    PARAMS = ARGPARSER.parse_args()
    main(PARAMS)