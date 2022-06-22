import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd as autograd

# USE_CUDA = torch.cuda.is_available()
USE_CUDA = False
Variable = lambda *args, **kwargs: autograd.Variable(*args, **kwargs).cuda() if USE_CUDA else autograd.Variable(*args, **kwargs)


class RolloutEncoder(nn.Module):
    def __init__(self, obs_dim, hidden_size):
        super(RolloutEncoder, self).__init__()

        self.obs_dim = obs_dim[0]
        self.rew_dim = 1

        self.gru = nn.GRU(input_size=self.obs_dim + self.rew_dim, hidden_size=hidden_size)


    def forward(self, state, reward):
        # num_steps  = state.size(0) # seq_len
        # batch_size = state.size(1) # batch

        rnn_input = torch.cat([state, reward], 2) #shape: (seq_len, batch_size, input_size)
        _, hidden = self.gru(rnn_input)
        return hidden.squeeze(0)

    # def feature_size(self):
    #     return self.features(autograd.Variable(torch.zeros(1, *self.in_shape))).view(1, -1).size(1)

class LSTMRolloutEncoder(nn.Module):
    def __init__(self, obs_dim, hidden_size):
        super().__init__()

        self.obs_dim = obs_dim[0]
        self.rew_dim = 1

        self.lstm = nn.LSTM(input_size=self.obs_dim + self.rew_dim, hidden_size=hidden_size)


    def forward(self, state, reward):
        # num_steps  = state.size(0) # seq_len
        # batch_size = state.size(1) # batch

        rnn_input = torch.cat([state, reward], 2) #shape: (seq_len, batch_size, input_size)
        _, (h_n, c_n) = self.lstm(rnn_input)
        return c_n.squeeze(0)

class ALTLSTMRolloutEncoder(nn.Module):
    def __init__(self, obs_dim, hidden_size):
        super().__init__()

        self.obs_dim = obs_dim
        self.rew_dim = 1

        self.features = nn.Sequential(
            nn.Linear(self.obs_dim[0], 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
        )

        self.lstm = nn.LSTM(input_size=65, hidden_size=hidden_size)


    def forward(self, state, reward):
        num_steps  = state.size(0) # seq_len
        batch_size = state.size(1) # batch

        state = state.view(-1, *self.obs_dim)
        state = self.features(state)
        state = state.view(num_steps, batch_size, -1)

        rnn_input = torch.cat([state, reward], 2) #shape: (seq_len, batch_size, input_size)
        _, (h_n, c_n) = self.lstm(rnn_input)
        return c_n.squeeze(0)

class ALTRolloutEncoder(nn.Module):
    def __init__(self, obs_dim, hidden_size):
        super(ALTRolloutEncoder, self).__init__()

        self.obs_dim = obs_dim
        self.rew_dim = 1

        self.features = nn.Sequential(
            nn.Linear(self.obs_dim[0], 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
        )

        self.gru = nn.GRU(input_size=65, hidden_size=hidden_size)


    def forward(self, state, reward):
        num_steps = state.size(0)
        batch_size = state.size(1)

        state = state.view(-1, *self.obs_dim)
        state = self.features(state)
        state = state.view(num_steps, batch_size, -1)
        rnn_input = torch.cat([state, reward], 2)
        _, hidden = self.gru(rnn_input)
        return hidden.squeeze(0)

class ALTRolloutEncoder32(nn.Module):
    def __init__(self, obs_dim, hidden_size):
        super().__init__()

        self.obs_dim = obs_dim
        self.rew_dim = 1

        self.features = nn.Sequential(
            nn.Linear(self.obs_dim[0], 32),
            nn.ReLU(),
            nn.Linear(32, 32),
            nn.ReLU(),
        )

        self.gru = nn.GRU(input_size=33, hidden_size=hidden_size)

    def forward(self, state, reward):
        num_steps = state.size(0)
        batch_size = state.size(1)

        state = state.view(-1, *self.obs_dim)
        state = self.features(state)
        state = state.view(num_steps, batch_size, -1)
        rnn_input = torch.cat([state, reward], 2)
        _, hidden = self.gru(rnn_input)
        return hidden.squeeze(0)


class OldRolloutEncoder(nn.Module):
    def __init__(self, in_shape, hidden_size):
        super(RolloutEncoder, self).__init__()

        self.in_shape = in_shape

        self.features = nn.Sequential(
            nn.Linear(in_shape[0], 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
        )

        self.gru = nn.GRU(self.feature_size() + 1, hidden_size)

    def forward(self, state, reward):
        num_steps  = state.size(0)
        batch_size = state.size(1)

        state = state.view(-1, *self.in_shape)
        state = self.features(state)
        state = state.view(num_steps, batch_size, -1)
        rnn_input = torch.cat([state, reward], 2)
        _, hidden = self.gru(rnn_input)
        return hidden.squeeze(0)

    def feature_size(self):
        return self.features(autograd.Variable(torch.zeros(1, *self.in_shape))).view(1, -1).size(1)



class ImaginationCore(object):
    def __init__(self, num_rollouts, in_shape, num_actions, env_model, distil_policy, type_rollout='full'):
        self.num_rollouts = num_rollouts
        self.in_shape = in_shape
        self.num_actions = num_actions
        self.env_model = env_model
        self.distil_policy = distil_policy
        self.type_rollout = type_rollout

    def __call__(self, state):
        # state = state.cpu()
        batch_size = state.size(0)

        rollout_states = []
        rollout_rewards = []

        if self.type_rollout == 'full':
            state = state.unsqueeze(0).repeat(self.num_actions, 1, 1, 1).view(-1, *self.in_shape)
            action = torch.LongTensor([[i] for i in range(self.num_actions)] * batch_size)
            action = action.view(-1)
            rollout_batch_size = batch_size * self.num_actions
        elif isinstance(self.type_rollout, int):
            pi = self.distil_policy.pi._distribution(state)
            probs = pi.probs
            action = torch.topk(probs, self.type_rollout).indices
            action = action.view(-1)
            state = state.unsqueeze(0).repeat(self.type_rollout, 1, 1).view(-1, *self.in_shape)
            rollout_batch_size = batch_size * self.type_rollout
        else:
            with torch.no_grad():
                action = self.distil_policy.act(Variable(state))
                action = action.cpu()
                rollout_batch_size = batch_size

        for step in range(self.num_rollouts):
            onehot_action = torch.zeros(rollout_batch_size, self.num_actions, *self.in_shape[1:])
            onehot_action[range(rollout_batch_size), action] = 1
            inputs = torch.cat([state, onehot_action], 1)

            with torch.no_grad():
                imagined_state, imagined_reward = self.env_model(Variable(inputs))

            rollout_states.append(imagined_state.unsqueeze(0))
            rollout_rewards.append(imagined_reward.unsqueeze(0))

            if step == self.num_rollouts-1:
                break
            state = imagined_state
            with torch.no_grad():
                action = self.distil_policy.act(Variable(state))
                action = action.cpu()

        return torch.cat(rollout_states), torch.cat(rollout_rewards)
