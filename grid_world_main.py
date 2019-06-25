

from grid_world_env import grid_env


def main():
    env = grid_env()
    env.render()
    state = env.reset()
    
    while True:
        action = int(input("Choose action: "))
        new_state, reward, done = env.step(action)

        env.render()

        print(new_state)
        print(reward)

        if done:
            break


if __name__ == "__main__":
    main()