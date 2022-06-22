import numpy as np
import os.path as osp
import time
import random

import gym
import torch
import torch.nn as nn
import torch.optim as optim
import torch.autograd as autograd

from PPO_ALG.utils.logx import EpochLogger
from PPO_ALG.utils.run_utils import setup_logger_kwargs
from IPython.display import clear_output
import matplotlib.pyplot as plt

from PPO_ALG.algo.core import MLPActorCritic
from ENV_MODEL.env_models import *

# import env
from ENVIRONMENT.envs.utils.robot import Robot
from ENVIRONMENT.envs.utils.state import transform
import ENVIRONMENT.configs.i2a as i2a_config
# import placeholder policy so that env doesnt produce errors
from ENVIRONMENT.envs.policy.placeholder import Placeholder
from ENVIRONMENT.envs.utils.state import JointState
from ENVIRONMENT.envs.utils.info import *
from ENVIRONMENT.envs.utils.multiprocessing_env import SubprocVecEnv

# 2 human_pp
# a2c_dir = '/home/WIN-UNI-DUE/sosiharp/Python Files/MA_Final/PPO_ALG/data/2_humans_asyn_ppo/EXP_NAME/EXP_NAME_s0/pyt_save/'

# 5 human_ppo
a2c_dir = "/home/WIN-UNI-DUE/sosiharp/Python Files/MA_Final/PPO_ALG/data/5_humans_ppo/2022-06-11_ppo_new_01_pen/2022-06-11_21-05-34-ppo_new_s96/pyt_save/"
a2c_name = 'model400.pt'
# should include num humans
save_dir = 'data/5_humans_EnvModel/'

# SHOULD INCLUDE Type of ENV_MODEL
exp_name = "MixedEnvModel"

num_envs = 4
num_updates = 50000

# Random seed
seed = random.randint(0,1000)
torch.manual_seed(seed)
np.random.seed(seed)

# Set up logger and save configuration
logger_kwargs = setup_logger_kwargs(exp_name,data_dir=save_dir, datestamp=True, seed=seed)
logger = EpochLogger(**logger_kwargs)
logger.save_config(locals())

# USE_CUDA = torch.cuda.is_available()
USE_CUDA = False
Variable = lambda *args, **kwargs: autograd.Variable(*args, **kwargs).cuda() if USE_CUDA else autograd.Variable(*args, **kwargs)

def make_env():
    def _thunk():
        # Instantiate environment
        env_config = i2a_config.EnvConfig(False)
        env = gym.make('CrowdSim-v0')
        env.configure(env_config)
        robot = Robot(env_config, 'robot')
        robot.time_step = env.time_step
        env.set_robot(robot)
        policy = Placeholder()
        robot.set_policy(policy)
        return env

    return _thunk

envs = [make_env() for i in range(num_envs)]
envs = SubprocVecEnv(envs)

state_shape = envs.observation_space.shape
num_actions = envs.action_space.n

env_model    = MixedEnvModel(envs.observation_space.shape, num_actions)
actor_critic = MLPActorCritic(envs.observation_space.shape, envs.action_space, type='discrete')

# Count variables
if not isinstance(env_model, LinearStateRewardPredictor):
    var_counts = tuple(sum([np.prod(p.shape) for p in module.parameters()]) for module in [env_model])
    logger.log('\nNumber of parameters: \t EnvModel: %d\n' % var_counts)

    optimizer = optim.Adam(env_model.parameters(), lr=0.001)

criterion = nn.MSELoss()


if USE_CUDA:
    env_model    = env_model.cuda()
    actor_critic = actor_critic.cuda()


# actor_critic.load_state_dict(torch.load(PATH + "actor_critic_" + mode + ".pth"))
actor_critic.load_state_dict(torch.load(a2c_dir + a2c_name))

# ORIGINAL:
# actor_critic.load_state_dict(torch.load("actor_critic_" + mode))

def smooth(scalars, weight) -> list:  # Weight between 0 and 1
    last = scalars[0]  # First value in the plot (first timestep)
    smoothed = list()
    for point in scalars:
        smoothed_val = last * weight + (1 - weight) * point  # Calculate smoothed value
        smoothed.append(smoothed_val)  # Save it
        last = smoothed_val  # Anchor the last smoothed value
    return smoothed

def plot(frame_idx, rewards, losses, name):
    smoothed_losses = smooth(losses, 0.999)
    # if len(smoothed_losses) > 100:
    titel = f'{name}, Loss: {(sum(smoothed_losses) / len(smoothed_losses)).item():.2f}'
    # else:
    #     titel = losses[-1]
    clear_output(True)
    plt.figure(figsize=(10, 10))
    plt.subplot(111)
    plt.title(titel)
    plt.plot(losses)
    plt.plot(smoothed_losses, 'orange')
    # mean_loss = running_mean(losses,1000)
    # plt.plot(mean_loss, 'red')
    plt.show()

def get_action(state):
    if state.ndim == 2:
        state = torch.FloatTensor(np.float32(state))
    else:
        state = torch.FloatTensor(np.float32(state)).unsqueeze(0)

    action = actor_critic.act(Variable(state, volatile=True))
    # action = action.data.cpu().squeeze(1).numpy()
    return action

def play_games(envs, frames):
    states = envs.reset()
    # for i in range(len(states)):
    #     states[i] = transform(states[i]).unsqueeze(0)

    for frame_idx in range(frames):
        actions = get_action(states)
        next_states, rewards, dones, _ = envs.step(actions)
        # for state in next_states:
        #     state = transform(state).unsqueeze(0)
        yield frame_idx, states, actions, rewards, next_states, dones
        # if dones:
        #     states = envs.reset()
        # else:
        states = next_states



# IMPORTANT TO CHANGE
reward_coef = 10

local_num_updates = int(num_updates/num_envs)

losses = []
all_rewards = []
start_time = time.time()

for frame_idx, states, actions, rewards, next_states, dones in play_games(envs, local_num_updates):
    states = torch.FloatTensor(states)
    actions = torch.LongTensor(actions)

    batch_size = states.size(0)

    onehot_actions = torch.zeros(batch_size, num_actions, *state_shape[1:])
    onehot_actions[range(batch_size), actions] = 1
    inputs = Variable(torch.cat([states, onehot_actions], 1))

    if USE_CUDA:
        inputs = inputs.cuda()

    imagined_state, imagined_reward = env_model(inputs)

    target_state = next_states
    target_reward = rewards
    # skip prediction if env gets reset
    # for i, done in enumerate(dones):
    #     if done:
    #         target_state[i] = imagined_state[i]
    #         target_reward[i] = imagined_reward[i][0]
    #     else:
    #         target_state[i] = next_states[i]
    #         target_reward[i] = rewards[i]

    target_state = Variable(torch.FloatTensor(target_state))
    target_reward = Variable(torch.FloatTensor([target_reward]))

    optimizer.zero_grad()
    state_loss = criterion(imagined_state, target_state)
    reward_loss = criterion(imagined_reward, target_reward)
    loss = state_loss + reward_coef * reward_loss
    loss.backward()
    optimizer.step()

    losses.append(loss.data)
    all_rewards.append(np.mean(rewards))

    logger.store(Loss=loss, RewardLoss=reward_loss, StateLoss=state_loss,
                 Reward=np.mean(rewards), Iteration=frame_idx)

    # make 100 entries to logging, inputs are all the values
    if frame_idx % (local_num_updates/1000) == 0:
        # Log info about epoch
        logger.log_tabular('Iteration', average_only=True)
        logger.log_tabular('Loss', with_min_and_max=True)
        logger.log_tabular('RewardLoss', average_only=True)
        logger.log_tabular('StateLoss', average_only=True)
        logger.log_tabular('Reward', average_only=True)
        logger.log_tabular('EnvInteractions', (frame_idx+1) * num_envs)
        logger.log_tabular('Time', time.time() - start_time)
        logger.dump_tabular()

    # immediately plot the final result for good measure
    if frame_idx == num_updates-1:
        plot(frame_idx, all_rewards, losses, env_model.name)

# CHANGE TO SAVE MORE MAYBE?
torch.save(env_model.state_dict(), logger_kwargs['output_dir'] + '/' + exp_name + ".pt")
