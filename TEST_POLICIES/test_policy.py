import random

from PPO_ALG.utils.test_policy import load_policy_ppo, load_policy_i2a, run_policy
from PPO_ALG.algo.core import MLPActorCritic, I2AActorCritic
from I2A_WITH_PPO.i2a_components import ImaginationCore, ALTRolloutEncoder, ALTLSTMRolloutEncoder
from ENV_MODEL.env_models import EnvModel, CopyModel
import gym
import torch

# import env
from ENVIRONMENT.envs.utils.robot import Robot
from ENVIRONMENT.envs.utils.state import transform
import ENVIRONMENT.configs.i2a as i2a_config
# import placeholder policy so that env doesnt produce errors
from ENVIRONMENT.envs.policy.placeholder import Placeholder
from ENVIRONMENT.envs.utils.state import JointState
from ENVIRONMENT.envs.utils.info import *

# Instantiate environment
env_config = i2a_config.EnvConfig(False)
env = gym.make('CrowdSim-v0')
env.configure(env_config)
robot = Robot(env_config, 'robot')
robot.time_step = env.time_step
env.set_robot(robot)
policy = Placeholder()
robot.set_policy(policy)

# DOCUMENTATION
# https://spinningup.openai.com/en/latest/user/saving_and_loading.html#loading-and-running-trained-policies
I2A = '/home/WIN-UNI-DUE/sosiharp/Python Files/MA_Final/I2A_WITH_PPO/data/i2a_for_eval/2022-06-14_copy_model/2022-06-14_17-44-44-copy_model_s91'
PPO = '/home/WIN-UNI-DUE/sosiharp/Python Files/MA_Final/PPO_ALG/data/ppo_for_eval/2022-06-14_ppo_16000/2022-06-14_17-13-07-ppo_16000_s72'

fpath           = I2A

algo            = I2AActorCritic # MLPActorCritic
len             = 0
episodes        = 100
render          = True
render_num      = 0
itr             = -1
deterministic   = True
ac_kwargs={
    'hidden_sizes': [64, 64]
}
render_episodes = [random.randint(0,500) for x in range(render_num)]
ppo = False
if ppo:
    _, get_action = load_policy_ppo(fpath, algo, ac_kwargs,  env,
                                      itr if itr >=0 else 'last',
                                      deterministic)
else:
    # FOR INFOS ON IMPLEMENTED PROPERTIES SEE THE config.json file
    # this is only a temporary solution due to low time
    imagination_depth = 5
    type_rollout = 1
    encoder_size = 64

    PATH = "/home/WIN-UNI-DUE/sosiharp/Python Files/MA_Final/ENV_MODEL/data/5_humans_EnvModel/2022-05-30_ANNEnvModel/2022-05-30_00-49-36-ANNEnvModel_s547/"
    env_model = CopyModel(env.observation_space.shape, env.action_space.n)
    # env_model.load_state_dict(torch.load(PATH + "ANNEnvModel.pt"))
    distil_policy = MLPActorCritic(env.observation_space.shape, env.action_space, **ac_kwargs)
    imagination = ImaginationCore(imagination_depth, env.observation_space.shape, env.action_space.n, env_model, distil_policy,
                                  type_rollout=type_rollout)
    encoder = ALTLSTMRolloutEncoder(env.observation_space.shape, encoder_size)
    _, get_action = load_policy_i2a(fpath, itr if itr >=0 else 'last', algo, imagination, encoder, encoder_size, type_rollout, ac_kwargs, env, deterministic)
run_policy(env, get_action, len, episodes, render, render_episodes)
