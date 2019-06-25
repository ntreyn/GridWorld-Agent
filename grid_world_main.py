

from grid_world_env import grid_env
from qlearn import qlearner
from mclearn import mclearner

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
    agent.learn()

    print(agent.qtable)

    env.render()
    state = env.reset()
    total_reward = 0
    
    while True:
        action = agent.act(state)
        new_state, reward, done = env.step(action)

        env.render()
        state = new_state
        total_reward += reward
        print(reward)

        if done:
            print(total_reward)
            break


if __name__ == "__main__":
    main()