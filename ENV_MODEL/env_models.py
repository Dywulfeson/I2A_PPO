import torch
import torch.nn as nn
from ENVIRONMENT.envs.utils.action import build_torch_action_space

'''
Available Env-Models:

OldEnvModel:                NN for both state and reward
                            common basis 2 layer [128, 256]
                            state-head 1 layer [128]
                            reward-head 1 layer [128]
                            
EnvModel:                   NN for both state and reward
                            common basis 2 layer [act_space+obs_space, act_space+obs_space]
                            state-head 0 layer []
                            reward-head 0 layer []       
                            
MixedEnvModel:              NN for state, calc for reward
                            state-net [act_space+obs_space, act_space+obs_space]
                            reward calc based on next state like in env
                            
LinearStateRewardPredictor  linear state prediction
                            reward calc based on next state like in env
             

'''

class CopyModel(object):
    def __init__(self, state_dims, num_actions):
        self.name = 'Copy Model'
        self.state_dims =state_dims
        self.num_actions = num_actions

    # PLACEHOLDER METHODS
    def load_state_dict(self, a):
        pass

    def cuda(self):
        pass

    def __call__(self, inputs):
        batch_size = inputs.size(0)
        new_states, _ = torch.split(inputs, [self.state_dims[0], self.num_actions], dim=1)
        new_rewards = torch.zeros([batch_size]).unsqueeze(1).to(device=inputs.device)

        return new_states, new_rewards

class OldEnvModel(nn.Module):
    def __init__(self, state_dims, num_actions):
        super(OldEnvModel, self).__init__()
        self.name = 'Standard EnvModel'

        self.features = nn.Sequential(
            nn.Linear(num_actions + state_dims[0], 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
        )

        self.state_fc = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, state_dims[0]),
            nn.ReLU()
        )

        self.reward_fc = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
            nn.ReLU()
        )

    def forward(self, inputs):
        batch_size = inputs.size(0)

        x = self.features(inputs)

        state = self.state_fc(x)
        reward = self.reward_fc(x)

        return state, reward

class EnvModel(OldEnvModel):
    def __init__(self, state_dims, num_actions):
        super().__init__(state_dims, num_actions)
        self.name = '[38, 38] EnvModel'
        self.features = nn.Sequential(
            nn.Linear(num_actions + state_dims[0], num_actions + state_dims[0]),
            nn.ReLU(),
            nn.Linear(num_actions + state_dims[0], num_actions + state_dims[0]),
            nn.ReLU(),
        )

        self.state_fc = nn.Sequential(
            nn.Linear(num_actions + state_dims[0], state_dims[0]),
            nn.ReLU()
        )

        self.reward_fc = nn.Sequential(
            nn.Linear(num_actions + state_dims[0], 1),
            nn.ReLU()
        )

class MixedEnvModel(nn.Module):
    def __init__(self, state_dims, num_actions):
        super(MixedEnvModel, self).__init__()
        self.name = 'MixedStateRewardPredictor'
        self.state_dims = state_dims
        self.num_actions = num_actions
        self.time_step = 0.25
        self.robot_state_dims = 7
        self.humans_state_dims = 7
        self.kinematics = 'holonomic'
        self.success_rew = 1
        self.collision_pen = -0.25
        self.discomfort_dist = 0.2
        self.discomfort_penalty_factor = 0.5
        self.approaching_rew = 0.1
        self.elapsed_time_penalty = -0.01

        self.state_net = nn.Sequential(
            nn.Linear(num_actions + state_dims[0], num_actions + state_dims[0]),
            nn.ReLU(),
            nn.Linear(num_actions + state_dims[0], num_actions + state_dims[0]),
            nn.ReLU(),
            nn.Linear(num_actions + state_dims[0], state_dims[0]),
            nn.ReLU()
        )

    def forward(self, inputs):
        batch_size = inputs.size(0)
        new_states = self.state_net(inputs)
        states, one_hot_actions = torch.split(inputs, [self.state_dims[0], self.num_actions], dim=1)
        rewards = self.calculate_rewards(states, new_states)

        return new_states, rewards

    def calculate_rewards(self, states, new_states):
        batch_size = states.size(0)
        old_dist_goal = torch.norm(states[:,:2],2,dim=1)
        new_dist_goal = torch.norm(new_states[:,:2],2,dim=1)
        delta_to_goal = new_dist_goal - old_dist_goal

        # calculate goal condition
        reaching_goal = new_dist_goal < new_states[:,4]

        # reward = self.elapsed_time_penalty
        reward = - delta_to_goal * self.approaching_rew

        # calculate collisions & dmins
        dmin = torch.full(([batch_size]), float('inf')).to(device=states.device)
        collision = torch.full(([batch_size]), False).to(device=states.device)
        true_tensor = torch.full(([batch_size]), True).to(device=states.device)
        for j in range(self.robot_state_dims, new_states.shape[1], self.humans_state_dims):
            smaller = torch.le(new_states[:,j+5], dmin)
            dmin = torch.where(smaller, new_states[:,j+5], dmin)
            collision = torch.where(torch.le(new_states[:,j+5],new_states[:,j+6]), true_tensor, collision)
        reward = reward + (dmin - 2 * new_states[:, 4] - self.discomfort_dist) * self.discomfort_penalty_factor
        reward = torch.where(reaching_goal,torch.ones([batch_size]).to(device=states.device)*self.success_rew,reward)
        reward = torch.where(collision,torch.ones([batch_size]).to(device=states.device)*self.collision_pen,reward)

        return reward.unsqueeze(1)


class LinearStateRewardPredictor(object):
    def __init__(self, state_dims, num_actions):
        self.name = 'LinearStateRewardPredictor'
        self.state_dims = state_dims
        self.num_actions = num_actions
        self.time_step = 0.25
        self.robot_state_dims = 7
        self.humans_state_dims = 7
        self.kinematics = 'holonomic'
        self.success_rew = 1
        self.collision_pen = -0.25
        self.discomfort_dist = 0.2
        self.discomfort_penalty_factor = 0.5
        self.approaching_rew = 0.1
        self.elapsed_time_penalty = -0.01

        if self.num_actions == 17:
            speed_samples = 2
            rotation_samples = 8
        elif self.num_actions == 81:
            speed_samples = 5
            rotation_samples = 16
        else:
            raise NotImplementedError()
        self.action_space = build_torch_action_space(speed_samples, rotation_samples, self.kinematics)


    #PLACEHOLDER METHODS
    def load_state_dict(self,a):
        pass

    def cuda(self):
        pass

    def __call__(self, inputs):
        batch_size = inputs.size(0)

        states, one_hot_actions = torch.split(inputs, [self.state_dims[0], self.num_actions], dim=1)
        actions = torch.argmax(one_hot_actions, dim=1)
        new_states = torch.clone(states).to(device=states.device)

        # rephrase action from discrete to cont
        actions = self.action_space[actions]
        new_states[:, 2] = actions[:, 0] # vx
        new_states[:, 3] = actions[:, 1] # vy

        #calc new pos robot
        # gx
        new_states[:, 0] = new_states[:, 0] - new_states[:, 2] * self.time_step
        # gy
        new_states[:, 1] = new_states[:, 1] - new_states[:, 3] * self.time_step

        # calc new pos humans
        for j in range(self.robot_state_dims, new_states.shape[1], self.humans_state_dims):
            # Px
            new_states[:, j+0] = new_states[:, j+0]\
                                       + (new_states[:, j+2] - new_states[:, 2]) * self.time_step
            # Py
            new_states[:, j + 1] = new_states[:, j + 1] \
                                         + (new_states[:, j + 3] - new_states[:, 3]) * self.time_step
            # Da
            new_states[:, j + 5] = torch.norm(new_states[:, :2]-new_states[:, j + 0:j + 2], 2, dim=1)

        # calc reward
        new_rewards = self.calculate_rewards(states, new_states)

        return new_states, new_rewards

    def calculate_rewards(self, states, new_states):
        batch_size = states.size(0)
        old_dist_goal = torch.norm(states[:, :2], 2, dim=1)
        new_dist_goal = torch.norm(new_states[:, :2], 2, dim=1)
        delta_to_goal = new_dist_goal - old_dist_goal

        # calculate goal condition
        reaching_goal = new_dist_goal < new_states[:, 4]

        # reward = self.elapsed_time_penalty
        reward = - delta_to_goal * self.approaching_rew

        # calculate collisions & dmins
        dmin = torch.full(([batch_size]), float('inf')).to(device=states.device)
        collision = torch.full(([batch_size]), False).to(device=states.device)
        true_tensor = torch.full(([batch_size]), True).to(device=states.device)
        for j in range(self.robot_state_dims, new_states.shape[1], self.humans_state_dims):
            smaller = torch.le(new_states[:, j + 5], dmin)
            dmin = torch.where(smaller, new_states[:, j + 5], dmin)
            collision = torch.where(torch.le(new_states[:, j + 5], new_states[:, j + 6]), true_tensor, collision)
        reward = reward + (dmin - 2 * new_states[:, 4] - self.discomfort_dist) * self.discomfort_penalty_factor
        reward = torch.where(reaching_goal, torch.ones([batch_size]).to(device=states.device) * self.success_rew,
                             reward)
        reward = torch.where(collision, torch.ones([batch_size]).to(device=states.device) * self.collision_pen, reward)

        return reward.unsqueeze(1)

class Action(object):
    def __init__(self):
        pass