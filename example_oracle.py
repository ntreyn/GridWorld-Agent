
from grid_world_env import grid_env
from mc_on_policy import mc_on
from mc_oracle import mc_oracle
from parameters import core_argparser

import matplotlib.pyplot as plt
import argparse



def main(args):
    env = grid_env()

    oracle = mc_on(env, args)
    baseline_rewards = oracle.train()

    agent = mc_oracle(env, oracle)
    oracle_rewards = agent.train()
    
    plt.plot(baseline_rewards)
    plt.show()

    plt.plot(oracle_rewards)
    plt.show()
    
    


if __name__ == "__main__":
    ARGPARSER = argparse.ArgumentParser(parents=[core_argparser()])
    PARAMS = ARGPARSER.parse_args()
    main(PARAMS)