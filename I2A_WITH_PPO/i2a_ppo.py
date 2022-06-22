import numpy as np
import torch
from torch.optim import Adam, RMSprop
import gym
import time
import random

# import ppo components
import PPO_ALG.algo.core as core
from PPO_ALG.algo.buffer import PPOBuffer
from PPO_ALG.utils.logx import EpochLogger
from PPO_ALG.utils.mpi_pytorch import setup_pytorch_for_mpi, sync_params, mpi_avg_grads
from PPO_ALG.utils.mpi_tools import mpi_fork, mpi_avg, proc_id, mpi_statistics_scalar, num_procs

# import i2a components
from ENV_MODEL.env_models import *
# from I2A_WITH_PPO.i2a_components import ImaginationCore
from I2A_WITH_PPO.i2a_components import *


# import env
from ENVIRONMENT.envs.utils.robot import Robot
from ENVIRONMENT.envs.utils.state import transform
import ENVIRONMENT.configs.i2a as i2a_config
# import placeholder policy so that env doesnt produce errors
from ENVIRONMENT.envs.policy.placeholder import Placeholder
from ENVIRONMENT.envs.utils.state import JointState
from ENVIRONMENT.envs.utils.info import *

from manage_parallel_experiments.ip_adress import get_ip

def i2a_ppo(i2a=core.I2AActorCritic, env_model=LinearStateRewardPredictor, encoder=RolloutEncoder, encoder_size=64,
            distill_policy=core.MLPActorCritic, ac_kwargs=dict(), type_rollout = 'full', imagination_depth = 2,
            seed=0, steps_per_epoch=4000, epochs=100, gamma=0.99, clip_ratio=0.2, pi_lr=3e-4,vf_lr=1e-3,
            train_pi_iters=80, train_v_iters=80, train_dist_iters=80, lam=0.97, max_ep_len=1000,target_kl=0.01,
            logger_kwargs=dict(), save_freq=10):
    """
    Proximal Policy Optimization (by clipping),
    with early stopping based on approximate KL

    Args:
        distill_policy: The constructor method for a PyTorch Module with a
            ``step`` method, an ``act`` method, a ``pi`` module, and a ``v``
            module. The ``step`` method should accept a batch of observations
            and return:

            ===========  ================  ======================================
            Symbol       Shape             Description
            ===========  ================  ======================================
            ``a``        (batch, act_dim)  | Numpy array of actions for each
                                           | observation.
            ``v``        (batch,)          | Numpy array of value estimates
                                           | for the provided observations.
            ``logp_a``   (batch,)          | Numpy array of log probs for the
                                           | actions in ``a``.
            ===========  ================  ======================================

            The ``act`` method behaves the same as ``step`` but only returns ``a``.

            The ``pi`` module's forward call should accept a batch of
            observations and optionally a batch of actions, and return:

            ===========  ================  ======================================
            Symbol       Shape             Description
            ===========  ================  ======================================
            ``pi``       N/A               | Torch Distribution object, containing
                                           | a batch of distributions describing
                                           | the policy for the provided observations.
            ``logp_a``   (batch,)          | Optional (only returned if batch of
                                           | actions is given). Tensor containing
                                           | the log probability, according to
                                           | the policy, of the provided actions.
                                           | If actions not given, will contain
                                           | ``None``.
            ===========  ================  ======================================

            The ``v`` module's forward call should accept a batch of observations
            and return:

            ===========  ================  ======================================
            Symbol       Shape             Description
            ===========  ================  ======================================
            ``v``        (batch,)          | Tensor containing the value estimates
                                           | for the provided observations. (Critical:
                                           | make sure to flatten this!)
            ===========  ================  ======================================


        ac_kwargs (dict): Any kwargs appropriate for the ActorCritic object
            you provided to PPO.

        seed (int): Seed for random number generators.

        steps_per_epoch (int): Number of steps of interaction (state-action pairs)
            for the agent and the environment in each epoch.

        epochs (int): Number of epochs of interaction (equivalent to
            number of policy updates) to perform.

        gamma (float): Discount factor. (Always between 0 and 1.)

        clip_ratio (float): Hyperparameter for clipping in the policy objective.
            Roughly: how far can the new policy go from the old policy while
            still profiting (improving the objective function)? The new policy
            can still go farther than the clip_ratio says, but it doesn't help
            on the objective anymore. (Usually small, 0.1 to 0.3.) Typically
            denoted by :math:`\epsilon`.

        pi_lr (float): Learning rate for policy optimizer.

        vf_lr (float): Learning rate for value function optimizer.

        train_pi_iters (int): Maximum number of gradient descent steps to take
            on policy loss per epoch. (Early stopping may cause optimizer
            to take fewer than this.)

        train_v_iters (int): Number of gradient descent steps to take on
            value function per epoch.

        lam (float): Lambda for GAE-Lambda. (Always between 0 and 1,
            close to 1.)

        max_ep_len (int): Maximum length of trajectory / episode / rollout.

        target_kl (float): Roughly what KL divergence we think is appropriate
            between new and old policies after an update. This will get used
            for early stopping. (Usually small, 0.01 or 0.05.)

        logger_kwargs (dict): Keyword args for EpochLogger.

        save_freq (int): How often (in terms of gap between epochs) to save
            the current policy and value function.

    """

    intermediate_save_count = int(epochs / save_freq)

    # Special function to avoid certain slowdowns from PyTorch + MPI combo.
    setup_pytorch_for_mpi()

    # Set up logger and save configuration
    logger = EpochLogger(**logger_kwargs)

    ip = get_ip()

    # log info about experiment
    extra_info = {
        'type_rollout': type_rollout,
        'imagination_depth': imagination_depth,
        'encoder_size':encoder_size,
        'ip_adress': ip,

    }
    logger.save_config(locals(), extra_info=extra_info)

    # Random seed
    seed += 10000 * proc_id()
    torch.manual_seed(seed)
    np.random.seed(seed)


    # Instantiate environment
    env_config = i2a_config.EnvConfig(False)
    env = gym.make('CrowdSim-v0')
    env.configure(env_config)
    robot = Robot(env_config, 'robot')
    robot.time_step = env.time_step
    env.set_robot(robot)
    policy = Placeholder()
    robot.set_policy(policy)

    obs_dim = env.observation_space.shape
    act_dim = env.action_space.n

    '''
        Integrate I2a into regular PPO algo
        '''



    env_model = env_model(env.observation_space.shape, env.action_space.n)
    if isinstance(env_model, MixedEnvModel):
        PATH='/home/WIN-UNI-DUE/sosiharp/Python Files/MA_Final/ENV_MODEL/data/5_humans_EnvModel/2022-06-12_MixedEnvModel/2022-06-12_01-47-12-MixedEnvModel_s753/MixedEnvModel.pt'
        env_model.load_state_dict(torch.load(PATH))
    elif isinstance(env_model, EnvModel):
        PATH = '/home/WIN-UNI-DUE/sosiharp/Python Files/MA_Final/ENV_MODEL/data/5_humans_EnvModel/2022-06-12_ANNEnvModel/2022-06-12_01-45-54-ANNEnvModel_s290/ANNEnvModel.pt'
        env_model.load_state_dict(torch.load(PATH))
    else:
        pass



    # Create actor-critic module
    distil_policy = distill_policy(env.observation_space.shape, env.action_space, **ac_kwargs)

    imagination = ImaginationCore(imagination_depth, obs_dim, act_dim, env_model, distil_policy, type_rollout=type_rollout)

    encoder = encoder(obs_dim, encoder_size)

    ac = i2a(env.observation_space, env.action_space, imagination, encoder, encoder_size, type_rollout=type_rollout, **ac_kwargs)

    # Sync params across processes
    sync_params(ac)
    sync_params(distil_policy)

    # Set up experience buffer
    local_steps_per_epoch = int(steps_per_epoch / num_procs())
    buf = PPOBuffer(obs_dim, act_dim, local_steps_per_epoch, gamma, lam)


    # Count variables
    var_counts = tuple(core.count_vars(module) for module in [distil_policy, ac])
    logger.log('\nNumber of parameters: \t DistillPolicy: %d, \t I2A: %d\n' % var_counts)

    # Set up function for computing PPO policy loss
    def compute_loss_pi(data):
        obs, act, adv, logp_old = data['obs'], data['act'], data['adv'], data['logp']

        # Policy loss
        logits = ac.eval_logits(obs)
        pi, logp = ac.eval_actions(logits, act)
        ratio = torch.exp(logp - logp_old)
        clip_adv = torch.clamp(ratio, 1 - clip_ratio, 1 + clip_ratio) * adv
        loss_pi = -(torch.min(ratio * adv, clip_adv)).mean()

        # Useful extra info
        approx_kl = (logp_old - logp).mean().item()
        ent = pi.entropy().mean()
        clipped = ratio.gt(1 + clip_ratio) | ratio.lt(1 - clip_ratio)
        clipfrac = torch.as_tensor(clipped, dtype=torch.float32).mean().item()
        pi_info = dict(kl=approx_kl, ent=ent, cf=clipfrac)

        return loss_pi, pi_info, logits

    # Set up function for computing value loss
    def compute_loss_v(data, logits):
        ret = data['ret']
        return ((ac.eval_val(logits) - ret) ** 2).mean()

    def compute_loss_dist(data, logits):
        obs, act = data['obs'], data['act']
        with torch.no_grad():
            pi, logp = ac.eval_actions(logits, act)
        pi_dist, logp_dist = distil_policy.pi(obs, act)
        loss_dist = -0.01 * (pi.probs * pi_dist.logits)
        loss_dist = loss_dist.sum(1).mean()
        return loss_dist


    # Set up optimizers for policy and value function
    # different optimizer from i2a, delete
    # rmsprop hyperparams:
    # lr = 7e-4
    # eps = 1e-5
    # alpha = 0.99
    # optimizer = RMSprop(ac.parameters(), lr, eps=eps, alpha=alpha)

    # Set up optimizers for policy and value function
    pi_optimizer = Adam(ac.pi.parameters(), lr=pi_lr)
    vf_optimizer = Adam(ac.v.parameters(), lr=vf_lr)
    distil_optimizer = Adam(distil_policy.parameters(), lr=pi_lr)

    # Set up model saving
    logger.setup_pytorch_saver(ac)

    # torch.set_printoptions(profile='full')

    def update():
        data = buf.get()

        pi_l_old, pi_info_old, logits = compute_loss_pi(data)
        pi_l_old = pi_l_old.item()
        v_l_old = compute_loss_v(data, logits).item()
        dist_l_old = compute_loss_dist(data, logits).item()

        # Train policy with multiple steps of gradient descent
        for i in range(train_pi_iters):
            pi_optimizer.zero_grad()
            loss_pi, pi_info, logits = compute_loss_pi(data)
            kl = mpi_avg(pi_info['kl'])
            if kl > 1.5 * target_kl:
                logger.log('Early stopping at step %d due to reaching max kl.' % i)
                break
            loss_pi.backward()
            mpi_avg_grads(ac.pi)  # average grads across MPI processes
            pi_optimizer.step()

        logger.store(StopIter=i)
        logits = logits.detach()

        # Value function learning
        for i in range(train_v_iters):
            vf_optimizer.zero_grad()
            loss_v = compute_loss_v(data, logits)
            loss_v.backward()
            mpi_avg_grads(ac.v)  # average grads across MPI processes
            vf_optimizer.step()

        # Distilled Policy learning
        for i in range(train_dist_iters):
            distil_optimizer.zero_grad()
            loss_dist = compute_loss_dist(data, logits)
            loss_dist.backward()
            # unsicher obs so klappt oder .pi und .v sein muss
            # wahrscheinlich .pi
            mpi_avg_grads(distil_policy.pi)
            distil_optimizer.step()

        # Log changes from update
        kl, ent, cf = pi_info['kl'], pi_info_old['ent'], pi_info['cf']
        logger.store(LossPi=pi_l_old, LossV=v_l_old, LossDist=dist_l_old,
                     KL=kl, Entropy=ent, ClipFrac=cf,
                     DeltaLossPi=(loss_pi.item() - pi_l_old),
                     DeltaLossV=(loss_v.item() - v_l_old))



    # Prepare for interaction with environment
    start_time = time.time()
    o, ep_ret, ep_len = env.reset(), 0, 0
    o = o.unsqueeze(0)

    # Main loop: collect experience in env and update/log each epoch
    for epoch in range(epochs):
        episodes_per_epoch = 0
        success_per_epoch = 0
        collision_per_epoch = 0
        timeouts_per_epoch = 0
        for t in range(local_steps_per_epoch):
            a, v, logp = ac.step(torch.as_tensor(o, dtype=torch.float32))
            a = torch.as_tensor(a, dtype=torch.float32).unsqueeze(0)

            next_o, r, d, info = env.step(int(a.item()))
            next_o = next_o.unsqueeze(0)
            ep_ret += r
            ep_len += 1

            # save and log
            buf.store(o, a, r, v, logp)
            logger.store(VVals=v)

            # Update obs (critical!)
            o = next_o

            timeout = ep_len == max_ep_len
            terminal = d or timeout
            epoch_ended = t == local_steps_per_epoch - 1

            if terminal or epoch_ended:
                if epoch_ended and not (terminal):
                    print('Warning: trajectory cut off by epoch at %d steps.' % ep_len, flush=True)
                # if trajectory didn't reach terminal state, bootstrap value target
                if timeout or epoch_ended:
                    _, v, _ = ac.step(torch.as_tensor(o, dtype=torch.float32))
                else:
                    v = 0
                buf.finish_path(v)
                if terminal:
                    # only save EpRet / EpLen if trajectory finished
                    logger.store(EpRet=ep_ret, EpLen=ep_len*env.time_step)
                    episodes_per_epoch += 1
                    if isinstance(info, ReachGoal):
                        success_per_epoch += 1
                    elif isinstance(info, Collision):
                        collision_per_epoch += 1
                    elif isinstance(info, Timeout):
                        timeouts_per_epoch += 1
                    else:
                        raise ValueError('Invalid end signal from environment')
                o, ep_ret, ep_len = env.reset(), 0, 0
                o = o.unsqueeze(0)

        # log success and collision rate
        logger.store(SuccessRate=success_per_epoch/episodes_per_epoch, CollRate=collision_per_epoch/episodes_per_epoch)

        # Save model
        if (epoch % intermediate_save_count == 0) or (epoch == epochs - 1):
            logger.save_state({'env': env}, epoch+1)

        # Perform PPO update!
        update()

        # Log info about epoch
        logger.log_tabular('Epoch', epoch)
        logger.log_tabular('SuccessRate', average_only=True)
        logger.log_tabular('CollRate', average_only=True)
        logger.log_tabular('EpRet', with_min_and_max=True)
        logger.log_tabular('EpLen', average_only=True)
        logger.log_tabular('VVals', with_min_and_max=True)
        logger.log_tabular('TotalEnvInteracts', (epoch + 1) * steps_per_epoch)
        logger.log_tabular('LossPi', average_only=True)
        logger.log_tabular('LossV', average_only=True)
        logger.log_tabular('LossDist', average_only=True)
        logger.log_tabular('DeltaLossPi', average_only=True)
        logger.log_tabular('DeltaLossV', average_only=True)
        logger.log_tabular('Entropy', average_only=True)
        logger.log_tabular('KL', average_only=True)
        logger.log_tabular('ClipFrac', average_only=True)
        logger.log_tabular('StopIter', average_only=True)
        logger.log_tabular('Time', time.time() - start_time)
        logger.dump_tabular()


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--env_model', type=int, default=EnvModel)
    parser.add_argument('--encoder', default=ALTLSTMRolloutEncoder)
    parser.add_argument('--distill_policy', type=int, default=core.MLPActorCritic)

    parser.add_argument('--hid', type=int, default=64)
    parser.add_argument('--l', type=int, default=2)
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--type_rollout', default=1)
    parser.add_argument('--imagination_depth', type=int, default=5)
    parser.add_argument('--seed', '-s', type=int, default=random.randint(0,100))
    parser.add_argument('--cpu', type=int, default=8)
    parser.add_argument('--steps', type=int, default=32000)
    parser.add_argument('--epochs', type=int, default=400)
    parser.add_argument('--exp_name', type=str, default='lstm_32000_time_pen')
    parser.add_argument('--output_dir', type=str, default= 'data/i2a_for_eval')
    args = parser.parse_args()

    mpi_fork(args.cpu)  # run parallel code with mpi

    from PPO_ALG.utils.run_utils import setup_logger_kwargs
    logger_kwargs = setup_logger_kwargs(args.exp_name, args.seed, args.output_dir, datestamp=True)

    i2a_ppo(i2a=core.I2AActorCritic, env_model=args.env_model, encoder=args.encoder, encoder_size=64,
            distill_policy=args.distill_policy,
        imagination_depth=args.imagination_depth, type_rollout=args.type_rollout,
        ac_kwargs=dict(hidden_sizes=[args.hid]*args.l), gamma=args.gamma,
        seed=args.seed, steps_per_epoch=args.steps, epochs=args.epochs,
        logger_kwargs=logger_kwargs)

