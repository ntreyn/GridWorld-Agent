
import argparse
from mc_off_policy import mc_off
from mc_on_policy import mc_on
from qlearn import qlearner
from dqn.dqn import DQN

MODEL_MAP = {
    'qlearn': qlearner,
    'mc_on': mc_on,
    'mc_off': mc_off,
    'dqn': DQN
}

def core_argparser():
    argparser = argparse.ArgumentParser(add_help=False)
    argparser.add_argument(
        '--model',
        default='qlearn',
        type=str,
        help='load model name'
    )
    return argparser