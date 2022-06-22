import os
import os.path as osp
from PPO_ALG.utils.plot import *

DATA = ["/home/WIN-UNI-DUE/sosiharp/Python Files/MA_Final/PPO_ALG/data/"]
active_data_dir = DATA
smooth = 20
legend = None  # ['I2A_PPO']
title = None # 'Umgebungsmodelle'  # 'Rollout-Breiten'



VALUE = [
'SuccessRate',
'CollRate',
# 'AverageEpRet',
# 'AverageVVals',
# 'ClipFrac',
# 'DeltaLossPi',
# 'DeltaLossV',
# 'Entropy',
'EpLen',
# 'Epoch',
# 'KL',
# 'LossPi',
# 'LossV',
# 'LossDist',
# 'TotalLoss'
# 'MaxEpRet',
# 'MinEpRet',
# 'MinVVals',
# 'Performance',
# 'StdEpRet',
# 'StdVVals',
]

if __name__ == '__main__':
    save_dir = osp.join(active_data_dir[0], 'plots')
    make_plots(active_data_dir, values=VALUE, save_dir=save_dir, smooth=smooth, legend=legend, title=title) # , legend=legend
