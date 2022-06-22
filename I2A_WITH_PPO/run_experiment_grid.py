from PPO_ALG.utils.run_utils import ExperimentGrid
from I2A_WITH_PPO.i2a_ppo import i2a_ppo
import torch
import PPO_ALG.algo.core as core
from ENV_MODEL.env_models import *
from I2A_WITH_PPO.i2a_components import *


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--cpu', default=8)
    # parser.add_argument('--num_runs', type=int, default=1)

    parser.add_argument('--output_dir', type=str, default='data/final_preperation/1_CopyModel')

    args = parser.parse_args()

    eg = ExperimentGrid(name='i2a')
    # eg.add(param_name, values, shorthand, in_name)
    eg.add('seed', [0]) #muss so viel wie num_runs sein
    eg.add('epochs', 400)
    eg.add('steps_per_epoch', [8000])
    eg.add('gamma', 0.99)
    eg.add('ac_kwargs:hidden_sizes', (64,64), 'hid')
    eg.add('env_model', [CopyModel], '_', in_name=True) # EnvModel, LinearStateRewardPredictor, MixedEnvModel
    eg.add('encoder', [ALTRolloutEncoder], 'enc_')
    eg.add('encoder_size', [32], 'enc_siz_')
    eg.add('distill_policy', core.MLPActorCritic)
    eg.add('type_rollout', ['full'], 'width_', in_name=True) # 1, 3, 5, 'full'
    eg.add('imagination_depth', [1], 'depth_', in_name=True)

    # eg.add('ac_kwargs:activation', [torch.nn.Tanh, torch.nn.ReLU], '')
    eg.run(i2a_ppo, num_cpu=args.cpu, data_dir=args.output_dir)