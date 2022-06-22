import os
import os.path as osp
from PPO_ALG.utils.plot import *


DATA = ["/home/WIN-UNI-DUE/sosiharp/Python Files/MA_Final/ENV_MODEL/data/5_humans_EnvModel/"]
MIXED = ['/home/WIN-UNI-DUE/sosiharp/Python Files/MA_Final/ENV_MODEL/data/5_humans_EnvModel/2022-06-11_MixedEnvModel/']
ANN = ['/home/WIN-UNI-DUE/sosiharp/Python Files/MA_Final/ENV_MODEL/data/5_humans_EnvModel/2022-06-11_ANNEnvModel']
LINEAR = ['/home/WIN-UNI-DUE/sosiharp/Python Files/MA_Final/ENV_MODEL/data/5_humans_EnvModel/2022-06-11_LinPredEnvModel']
active_data_dir = MIXED + ANN + LINEAR


VALUE = [
# 'Iteration',
'AverageLoss',
# 'StdLoss',
'MaxLoss',
# 'MinLoss',
# 'RewardLoss',
# 'StateLoss',
# 'EnvInteractions',
]

if __name__ == '__main__':
    save_dir = osp.join(active_data_dir[0], 'plots')
    make_plots(active_data_dir, values=VALUE, xaxis='EnvInteractions', save_dir=save_dir)
