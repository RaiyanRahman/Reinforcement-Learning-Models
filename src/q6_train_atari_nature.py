import gym
import subprocess
from pathlib import Path
from utils.preprocess import greyscale
from utils.wrappers import PreproWrapper, MaxAndSkipEnv

from q2_schedule import LinearExploration, LinearSchedule
from q4_nature_torch import NatureQN

from configs.q6_train_atari_nature import config
"""
Use deep Q network for the Atari game. This is an optional part of the assignment and will not be graded.
Feel free to change the configurations (in the configs/ folder). 

You'll find the results, log and video recordings of your agent every 250k under
the corresponding file in the results folder. A good way to monitor the progress
of the training is to use Tensorboard. The starter code writes summaries of different
variables.
"""
if __name__ == '__main__':
    # make env
    env = gym.make(config.env_name)
    env = MaxAndSkipEnv(env, skip=config.skip_frame)
    env = PreproWrapper(env, prepro=greyscale, shape=(80, 80, 1), 
                        overwrite_render=config.overwrite_render)

    # exploration strategy
    exp_schedule = LinearExploration(env, config.eps_begin, 
            config.eps_end, config.eps_nsteps)

    # learning rate schedule
    lr_schedule  = LinearSchedule(config.lr_begin, config.lr_end,
            config.lr_nsteps)

    # Check weights
    load_path = Path(config.load_path)
    if load_path.is_file():
        print(f'File {load_path} exists.')
    else:
        print(f'You need to have the file {load_path}.')

    # train model
    model = NatureQN(env, config)
    model.run(exp_schedule, lr_schedule)
