import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal

LOG_SIG_MAX = 2
LOG_SIG_MIN = -20
epsilon = 1e-6

# Initialize Policy weights
def weights_init_(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight, gain=1)
        torch.nn.init.constant_(m.bias, 0)


class ValueNetwork(nn.Module):
    def __init__(self, num_inputs, hidden_dim):
        super(ValueNetwork, self).__init__()

        self.linear1 = nn.Linear(num_inputs, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)
        self.linear3 = nn.Linear(hidden_dim, 1)

        self.apply(weights_init_)

    def forward(self, state):
        x = F.relu(self.linear1(state))
        x = F.relu(self.linear2(x))
        x = self.linear3(x)
        return x


class QNetwork(nn.Module):
    def __init__(self, num_inputs, num_actions, hidden_dim):
        super(QNetwork, self).__init__()

        # Q1 architecture
        self.linear1 = nn.Linear(num_inputs + num_actions, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)
        self.linear3 = nn.Linear(hidden_dim, 1)

        # Q2 architecture
        self.linear4 = nn.Linear(num_inputs + num_actions, hidden_dim)
        self.linear5 = nn.Linear(hidden_dim, hidden_dim)
        self.linear6 = nn.Linear(hidden_dim, 1)

        self.apply(weights_init_)

    def forward(self, state, action):
        xu = torch.cat([state, action], 1)
        
        x1 = F.relu(self.linear1(xu))
        x1 = F.relu(self.linear2(x1))
        x1 = self.linear3(x1)

        x2 = F.relu(self.linear4(xu))
        x2 = F.relu(self.linear5(x2))
        x2 = self.linear6(x2)

        return x1, x2


class QNetwork_out(nn.Module):
    def __init__(self, num_inputs, num_actions, hidden_dim):
        super(QNetwork_out, self).__init__()

        # Q1 architecture
        self.linear1 = nn.Linear(num_inputs, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)
        self.linear3 = nn.Linear(hidden_dim, num_actions)

        self.apply(weights_init_)

    def forward(self, state):
        x1 = F.relu(self.linear1(state))
        x1 = F.relu(self.linear2(x1))
        x1 = self.linear3(x1)

        return x1


class QNetwork_phi(nn.Module):
    def __init__(self, num_inputs, num_actions, hidden_dim, abs_range, tanh_output):
        super(QNetwork_phi, self).__init__()

        # Q1 network
        # obs encoder
        obs_models = [nn.Linear(num_inputs-2, hidden_dim)]
        obs_models += [nn.ReLU(), nn.Linear(hidden_dim, hidden_dim)]
        obs_models += [nn.ReLU(), nn.Linear(hidden_dim, 2)]

        self.obs_encoder = nn.Sequential(*obs_models)

        # goal input
        self.action_input = nn.Linear(num_actions+2, int(hidden_dim / 2))
        self.dynamics_layer = nn.Linear(int(hidden_dim / 2) + 2, hidden_dim)
        self.output_layer = nn.Linear(hidden_dim, 1)

        # Q2 architecture
        self.linear4 = nn.Linear(num_inputs + num_actions, hidden_dim)
        self.linear5 = nn.Linear(hidden_dim, hidden_dim)
        self.linear6 = nn.Linear(hidden_dim, 1)

        self.tanh_output = tanh_output
        self.abs_range = abs_range

        self.apply(weights_init_)

    def forward(self, state, action):
        xu = torch.cat([state, action], 1)

        x2 = F.relu(self.linear4(xu))
        x2 = F.relu(self.linear5(x2))
        x2 = self.linear6(x2)

        state = state[:, :-2]
        action = torch.cat([state[:, -2:], action], 1)
        latent_s = self.obs_encoder(state)
        if self.tanh_output:
            latent_s = self.abs_range * torch.tanh(latent_s)
        action_out = self.action_input(action)
        action_out = F.relu(action_out)
        x = torch.cat([latent_s, action_out], 1)
        x = self.dynamics_layer(x)
        x = F.relu(x)
        x1 = self.output_layer(x)

        return x1, x2

    def phi(self, obs):
        if len(obs.shape) is 1:
            obs = obs.unsqueeze(0)
        s = self.obs_encoder(obs)
        if self.tanh_output:
            s = self.abs_range * torch.tanh(s)
        return s


class GaussianPolicy(nn.Module):
    def __init__(self, num_inputs, num_actions, hidden_dim, action_space, goal_dim):
        super(GaussianPolicy, self).__init__()
        
        # self.linear1 = nn.Linear(num_inputs - goal_dim, hidden_dim)
        # self.goal_input = nn.Linear(goal_dim, hidden_dim)

        self.linear1 = nn.Linear(num_inputs, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)
        self.goal_dim = goal_dim

        self.mean_linear = nn.Linear(hidden_dim, num_actions)
        self.log_std_linear = nn.Linear(hidden_dim, num_actions)

        # add phi layer
        # self.phi_layer1 = nn.Linear(hidden_dim, hidden_dim)
        # self.phi_layer2 = nn.Linear(hidden_dim, 2)
        # self.phi_layer3 = nn.Linear(hidden_dim, 2)

        self.apply(weights_init_)

        # action rescaling
        if action_space is None:
            self.action_scale = torch.tensor(1.)
            self.action_bias = torch.tensor(0.)
        else:
            self.action_scale = torch.FloatTensor(
                (action_space.high - action_space.low) / 2.)
            self.action_bias = torch.FloatTensor(
                (action_space.high + action_space.low) / 2.)

    def forward(self, state):
        # x = self.linear1(state[..., :-self.goal_dim]) + self.goal_input(state[..., -self.goal_dim:])
        x = self.linear1(state)
        x = F.relu(x)

        x = F.relu(self.linear2(x))
        mean = self.mean_linear(x)
        log_std = self.log_std_linear(x)
        log_std = torch.clamp(log_std, min=LOG_SIG_MIN, max=LOG_SIG_MAX)
        return mean, log_std

    def phi(self, state):
        x = F.relu(self.linear1(state))
        x = F.relu(self.phi_layer1(x))
        phi = self.phi_layer2(x)
        return phi


    def sample(self, state):
        mean, log_std = self.forward(state)
        std = log_std.exp()
        normal = Normal(mean, std)
        x_t = normal.rsample()  # for reparameterization trick (mean + std * N(0,1))
        y_t = torch.tanh(x_t)
        action = y_t * self.action_scale + self.action_bias
        log_prob = normal.log_prob(x_t)
        # Enforcing Action Bound
        log_prob -= torch.log(self.action_scale * (1 - y_t.pow(2)) + epsilon)
        log_prob = log_prob.sum(1, keepdim=True)
        mean = torch.tanh(mean) * self.action_scale + self.action_bias
        return action, log_prob, mean

    def to(self, device):
        self.action_scale = self.action_scale.to(device)
        self.action_bias = self.action_bias.to(device)
        return super(GaussianPolicy, self).to(device)


class DeterministicPolicy(nn.Module):
    def __init__(self, num_inputs, num_actions, hidden_dim, action_space=None):
        super(DeterministicPolicy, self).__init__()
        self.linear1 = nn.Linear(num_inputs, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)

        self.mean = nn.Linear(hidden_dim, num_actions)
        self.noise = torch.Tensor(num_actions)

        self.apply(weights_init_)

        # action rescaling
        if action_space is None:
            self.action_scale = 1.
            self.action_bias = 0.
        else:
            self.action_scale = torch.FloatTensor(
                (action_space.high - action_space.low) / 2.)
            self.action_bias = torch.FloatTensor(
                (action_space.high + action_space.low) / 2.)

    def forward(self, state):
        x = F.relu(self.linear1(state))
        x = F.relu(self.linear2(x))
        mean = torch.tanh(self.mean(x)) * self.action_scale + self.action_bias
        return mean

    def sample(self, state):
        mean = self.forward(state)
        noise = self.noise.normal_(0., std=0.1)
        noise = noise.clamp(-0.25, 0.25)
        action = mean + noise
        return action, torch.tensor(0.), mean

    def to(self, device):
        self.action_scale = self.action_scale.to(device)
        self.action_bias = self.action_bias.to(device)
        self.noise = self.noise.to(device)
        return super(DeterministicPolicy, self).to(device)
