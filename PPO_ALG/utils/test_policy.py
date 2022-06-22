import time
import joblib
import os
import os.path as osp
import torch
from PPO_ALG.utils.logx import EpochLogger
from ENVIRONMENT.envs.utils.info import *


def load_policy_and_env(fpath, algo, algo_kwargs, env, itr='last', deterministic=False):
    """
    Load a policy from save, whether it's TF or PyTorch, along with RL env.

    Not exceptionally future-proof, but it will suffice for basic uses of the 
    Spinning Up implementations.

    Checks to see if there's a tf1_save folder. If yes, assumes the model
    is tensorflow and loads it that way. Otherwise, loads as if there's a 
    PyTorch save.
    """

    # determine if tf save or pytorch save
    if any(['tf1_save' in x for x in os.listdir(fpath)]):
        backend = 'tf1'
    else:
        backend = 'pytorch'

    # handle which epoch to load from
    if itr=='last':
        # check filenames for epoch (AKA iteration) numbers, find maximum value

        if backend == 'tf1':
            saves = [int(x[8:]) for x in os.listdir(fpath) if 'tf1_save' in x and len(x)>8]

        elif backend == 'pytorch':
            pytsave_path = osp.join(fpath, 'pyt_save')
            # Each file in this folder has naming convention 'modelXX.pt', where
            # 'XX' is either an integer or empty string. Empty string case
            # corresponds to len(x)==8, hence that case is excluded.
            saves = [int(x.split('.')[0][5:]) for x in os.listdir(pytsave_path) if len(x)>8 and 'model' in x]

        itr = '%d'%max(saves) if len(saves) > 0 else ''

    else:
        assert isinstance(itr, int), \
            "Bad value provided for itr (needs to be int or 'last')."
        itr = '%d'%itr

    # load the get_action function

    get_action = load_pytorch_policy(fpath, itr, algo, algo_kwargs, env, deterministic)

    # try to load environment from save
    # (sometimes this will fail because the environment could not be pickled)
    try:
        state = joblib.load(osp.join(fpath, 'vars'+itr+'.pkl'))
        env = state['env']
    except:
        env = None

    return env, get_action

def load_policy_ppo(fpath, algo, algo_kwargs, env, itr='last', deterministic=False):
    """
    Load a policy from save, whether it's TF or PyTorch, along with RL env.

    Not exceptionally future-proof, but it will suffice for basic uses of the
    Spinning Up implementations.

    Checks to see if there's a tf1_save folder. If yes, assumes the model
    is tensorflow and loads it that way. Otherwise, loads as if there's a
    PyTorch save.
    """

    # determine if tf save or pytorch save
    if any(['tf1_save' in x for x in os.listdir(fpath)]):
        backend = 'tf1'
    else:
        backend = 'pytorch'

    # handle which epoch to load from
    if itr=='last':
        # check filenames for epoch (AKA iteration) numbers, find maximum value

        if backend == 'tf1':
            saves = [int(x[8:]) for x in os.listdir(fpath) if 'tf1_save' in x and len(x)>8]

        elif backend == 'pytorch':
            pytsave_path = osp.join(fpath, 'pyt_save')
            # Each file in this folder has naming convention 'modelXX.pt', where
            # 'XX' is either an integer or empty string. Empty string case
            # corresponds to len(x)==8, hence that case is excluded.
            saves = [int(x.split('.')[0][5:]) for x in os.listdir(pytsave_path) if len(x)>8 and 'model' in x]

        itr = '%d'%max(saves) if len(saves) > 0 else ''

    else:
        assert isinstance(itr, int), \
            "Bad value provided for itr (needs to be int or 'last')."
        itr = '%d'%itr

    # load the get_action function

    get_action = load_pytorch_policy(fpath, itr, algo, algo_kwargs, env, deterministic)

    # try to load environment from save
    # (sometimes this will fail because the environment could not be pickled)
    try:
        state = joblib.load(osp.join(fpath, 'vars'+itr+'.pkl'))
        env = state['env']
    except:
        env = None

    return env, get_action

def load_policy_i2a(fpath, itr, algo, imagination, encoder, encoder_size, type_rollout, ac_kwargs,  env, deterministic=False):
    """
    Load a policy from save, whether it's TF or PyTorch, along with RL env.

    Not exceptionally future-proof, but it will suffice for basic uses of the
    Spinning Up implementations.

    Checks to see if there's a tf1_save folder. If yes, assumes the model
    is tensorflow and loads it that way. Otherwise, loads as if there's a
    PyTorch save.
    """

    # determine if tf save or pytorch save
    if any(['tf1_save' in x for x in os.listdir(fpath)]):
        backend = 'tf1'
    else:
        backend = 'pytorch'

    # handle which epoch to load from
    if itr=='last':
        # check filenames for epoch (AKA iteration) numbers, find maximum value

        if backend == 'tf1':
            saves = [int(x[8:]) for x in os.listdir(fpath) if 'tf1_save' in x and len(x)>8]

        elif backend == 'pytorch':
            pytsave_path = osp.join(fpath, 'pyt_save')
            # Each file in this folder has naming convention 'modelXX.pt', where
            # 'XX' is either an integer or empty string. Empty string case
            # corresponds to len(x)==8, hence that case is excluded.
            saves = [int(x.split('.')[0][5:]) for x in os.listdir(pytsave_path) if len(x)>8 and 'model' in x]

        itr = '%d'%max(saves) if len(saves) > 0 else ''

    else:
        assert isinstance(itr, int), \
            "Bad value provided for itr (needs to be int or 'last')."
        itr = '%d'%itr

    # load the get_action function

    get_action = load_i2a_policy(fpath, itr, algo, imagination, encoder, encoder_size, type_rollout, ac_kwargs, env, deterministic)

    # try to load environment from save
    # (sometimes this will fail because the environment could not be pickled)
    try:
        state = joblib.load(osp.join(fpath, 'vars'+itr+'.pkl'))
        env = state['env']
    except:
        env = None

    return env, get_action

def load_pytorch_policy(fpath, itr, algo, ac_kwargs, env, deterministic=False):
    """ Load a pytorch policy saved with Spinning Up Logger."""
    
    fname = osp.join(fpath, 'pyt_save', 'model'+itr+'.pt')
    print('\n\nLoading from %s.\n\n'%fname)

    # model = torch.load(fname)
    model = algo(env.observation_space.shape, env.action_space, **ac_kwargs)
    model.load_state_dict(torch.load(fname))

    # make function for producing an action given a single state
    def get_action(x):
        with torch.no_grad():
            x = torch.as_tensor(x, dtype=torch.float32)
            action = model.act(x)
        return action

    return get_action


def load_i2a_policy(fpath, itr, algo, imagination, encoder, encoder_size, type_rollout, ac_kwargs, env, deterministic=False):
    """ Load a pytorch policy saved with Spinning Up Logger."""

    fname = osp.join(fpath, 'pyt_save', 'model' + itr + '.pt')
    print('\n\nLoading from %s.\n\n' % fname)

    # model = torch.load(fname)

    model = algo(env.observation_space, env.action_space, imagination, encoder, encoder_size, type_rollout=type_rollout, **ac_kwargs)
    model.load_state_dict(torch.load(fname))

    # make function for producing an action given a single state
    def get_action(x):
        with torch.no_grad():
            x = torch.as_tensor(x, dtype=torch.float32)
            action = model.act(x)
        return action

    return get_action


def run_policy(env, get_action, max_ep_len=None, num_episodes=100, render=True, render_episodes=[]):

    assert env is not None, \
        "Environment not found!\n\n It looks like the environment wasn't saved, " + \
        "and we can't run the agent in it. :( \n\n Check out the readthedocs " + \
        "page on Experiment Outputs for how to handle this situation."

    logger = EpochLogger()
    o, r, d, ep_ret, ep_len, n = env.reset(), 0, False, 0, 0, 0
    succ, coll, timeout = 0, 0, 0
    rendered = 0
    while n < num_episodes:


        a = get_action(o)
        o, r, d, info = env.step(a)
        ep_ret += r
        ep_len += 1

        if d or (ep_len == max_ep_len):
            logger.store(EpRet=ep_ret, EpLen=ep_len*env.time_step)
            print('Episode %d \t EpRet %.3f \t EpLen %d'%(n, ep_ret, ep_len))
            if isinstance(info, ReachGoal):
                succ += 1
                if ep_len * env.time_step < 10:
                    env.render(mode='traj')
            elif isinstance(info, Collision):
                coll += 1
            elif isinstance(info, Timeout):
                timeout += 1
            else:
                raise ValueError('Invalid end signal from environment')
            if render_episodes is not False:
                if n in render_episodes:
                    env.render(mode='traj')
                # time.sleep(1e-3)
            o, r, d, ep_ret, ep_len = env.reset(), 0, False, 0, 0
            n += 1
    logger.store(SuccessRate=succ/num_episodes, CollisionRate=coll/num_episodes, TimeoutRate=timeout/num_episodes)

    logger.log_tabular('SuccessRate', average_only=True)
    logger.log_tabular('CollisionRate', average_only=True)
    logger.log_tabular('TimeoutRate', average_only=True)
    logger.log_tabular('EpRet', with_min_and_max=True)
    logger.log_tabular('EpLen', average_only=True)
    logger.dump_tabular()


def run_policy_blind(env, get_action, max_ep_len=None, num_episodes=100, render=True, render_episodes=[]):

    assert env is not None, \
        "Environment not found!\n\n It looks like the environment wasn't saved, " + \
        "and we can't run the agent in it. :( \n\n Check out the readthedocs " + \
        "page on Experiment Outputs for how to handle this situation."

    logger = EpochLogger()
    o, r, d, ep_ret, ep_len, n = env.reset(), 0, False, 0, 0, 0
    # delete all observations except goal
    o[2:] = 0.
    succ, coll, timeout = 0, 0, 0
    rendered = 0
    while n < num_episodes:


        a = get_action(o)
        o, r, d, info = env.step(a)
        # delete all observations except goal
        o[2:] = 0.
        ep_ret += r
        ep_len += 1

        if d or (ep_len == max_ep_len):
            logger.store(EpRet=ep_ret, EpLen=ep_len*env.time_step)
            print('Episode %d \t EpRet %.3f \t EpLen %d'%(n, ep_ret, ep_len))
            if isinstance(info, ReachGoal):
                succ += 1
                if ep_len * env.time_step < 10:
                    env.render(mode='traj')
            elif isinstance(info, Collision):
                coll += 1
            elif isinstance(info, Timeout):
                timeout += 1
            else:
                raise ValueError('Invalid end signal from environment')
            if render_episodes is not False:
                if n in render_episodes:
                    env.render(mode='traj')
                # time.sleep(1e-3)
            o, r, d, ep_ret, ep_len = env.reset(), 0, False, 0, 0
            # delete all observations except goal
            o[2:] = 0.
            n += 1
    logger.store(SuccessRate=succ/num_episodes, CollisionRate=coll/num_episodes, TimeoutRate=timeout/num_episodes)

    logger.log_tabular('SuccessRate', average_only=True)
    logger.log_tabular('CollisionRate', average_only=True)
    logger.log_tabular('TimeoutRate', average_only=True)
    logger.log_tabular('EpRet', with_min_and_max=True)
    logger.log_tabular('EpLen', average_only=True)
    logger.dump_tabular()


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('fpath', type=str)
    parser.add_argument('--len', '-l', type=int, default=0)
    parser.add_argument('--episodes', '-n', type=int, default=100)
    parser.add_argument('--norender', '-nr', action='store_true')
    parser.add_argument('--itr', '-i', type=int, default=-1)
    parser.add_argument('--deterministic', '-d', action='store_true')
    args = parser.parse_args()
    env, get_action = load_policy_and_env(args.fpath, 
                                          args.itr if args.itr >=0 else 'last',
                                          args.deterministic)
    run_policy(env, get_action, args.len, args.episodes, not(args.norender))