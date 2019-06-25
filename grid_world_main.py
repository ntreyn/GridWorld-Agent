

from grid_world_env import grid_env
from qlearn import qlearner
from mclearn import mclearner

import matplotlib.pyplot as plt

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


def main():
    env = grid_env()
    agent = qlearner(env)
    # agent = mclearner(env)
    episode_rewards = agent.learn()

    plt.plot(episode_rewards)
    plt.show()
    
    # env.render()
    state = env.reset()
    test_rewards = []
    
    for n in range(100):
        total_reward = 0
        for s in range(20):
            action = agent.act(state)
            new_state, reward, done = env.step(action)

            # env.render()
            state = new_state
            total_reward += reward
            # print(reward)

            if done:
                env.reset()
                break
        
        test_rewards.append(total_reward)
    
    plt.plot(test_rewards)
    plt.show()
    
    


if __name__ == "__main__":
    main()