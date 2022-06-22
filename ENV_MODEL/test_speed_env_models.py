import torch
from gym import spaces
from I2A_WITH_PPO.i2a_components import ImaginationCore
from env_models import *
from PPO_ALG.algo.core import MLPActorCritic
import time

batch_size = 100000
obs = torch.randn((batch_size,42))
act = torch.randint(17, (batch_size,))

type_rollout = 'full'
imagination_depth = 10
obs_dim = (42,)
act_dim = 17
act_space = spaces.Discrete(17)
env_model = LinearStateRewardPredictor(obs_dim, act_dim)
ac_kwargs = {'hidden_sizes': [64, 64], 'type': 'discrete'}
distil_policy = MLPActorCritic(obs_dim, act_space, **ac_kwargs)
imagination = ImaginationCore(imagination_depth, obs_dim, act_dim, env_model, distil_policy, type_rollout=type_rollout)


### START ###

start_time = time.time()

imagined_state, imagined_reward = imagination(obs)

finish_time = time.time()

print(f'Dauer: {finish_time-start_time}')