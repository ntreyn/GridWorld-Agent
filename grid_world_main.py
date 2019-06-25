

from grid_world_env import grid_env
from qlearn import qlearner


def main():
    env = grid_env()
    agent = qlearner(env)
    agent.learn()

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