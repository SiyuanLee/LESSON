import torch.nn.functional as F
import sys

sys.path.append('../')
from models.distance import *
import numpy as np
from torch.distributions.multivariate_normal import MultivariateNormal
import torch.distributions as D

"""
the input x in both networks should be [o, g], where o is the observation and g is the goal.
"""


def initialize_metrics(metric, dim):
    if metric == 'L1':
        return L1()
    elif metric == 'L2':
        return L2()
    elif metric == 'dot':
        return DotProd()
    elif metric == 'MLP':
        return MLPDist(dim)
    else:
        raise NotImplementedError


# define the actor network
class actor(nn.Module):
    def __init__(self, env_params, goal_dim):
        super(actor, self).__init__()
        self.max_action = env_params['action_max']
        self.fc1 = nn.Linear(env_params['low_dim'] + goal_dim, 400)
        self.fc2 = nn.Linear(400, 400)
        self.fc3 = nn.Linear(400, 400)
        self.fc4 = nn.Linear(400, 400)
        self.action_out = nn.Linear(400, env_params['action'])

    def forward(self, obs, goal):
        x = torch.cat([obs, goal], dim=1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        actions = self.max_action * torch.tanh(self.action_out(x))
        return actions


# define the actor network
class Inverse_goal(nn.Module):
    def __init__(self, env_params, goal_dim, hi_max_action):
        super(Inverse_goal, self).__init__()
        self.max_action = hi_max_action
        self.fc1 = nn.Linear(env_params['obs'] * 2, 400)
        self.fc2 = nn.Linear(400, 400)
        self.fc3 = nn.Linear(400, 400)
        self.fc4 = nn.Linear(400, 400)
        self.action_out = nn.Linear(400, goal_dim)

    def forward(self, obs, goal):
        x = torch.cat([obs, goal], dim=1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        actions = self.max_action * torch.tanh(self.action_out(x))
        return actions


# define the high-level actor network
class Hi_actor(nn.Module):
    def __init__(self, env_params, real_goal_dim, maze_high, shallow, sigmoid=False):
        super(Hi_actor, self).__init__()
        self.max_action = maze_high
        self.fc1 = nn.Linear(env_params['hi_dim'] + env_params['goal'], 400)
        self.fc2 = nn.Linear(400, 400)
        self.fc3 = nn.Linear(400, 400)
        self.fc4 = nn.Linear(400, 400)
        self.action_out = nn.Linear(400, real_goal_dim)
        self.sigmoid = sigmoid
        self.shallow = shallow

    def forward(self, obs, goal):
        x = torch.cat([obs, goal], dim=1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        if not self.shallow:
            x = F.relu(self.fc3(x))
            x = F.relu(self.fc4(x))
        if self.sigmoid:
            actions = self.max_action * torch.sigmoid(self.action_out(x))
        else:
            actions = self.max_action * torch.tanh(self.action_out(x))
        return actions


class Hi_critic(nn.Module):
    def __init__(self, env_params, args, real_goal_dim, maze_high):
        super(Hi_critic, self).__init__()
        self.max_action = maze_high
        self.inp_dim = env_params['hi_dim'] + real_goal_dim + env_params['goal']
        self.out_dim = 1
        self.mid_dim = 400
        self.gamma = args.gamma

        if args.hi_layer == 1:
            models = [nn.Linear(self.inp_dim, self.out_dim)]
        else:
            models = [nn.Linear(self.inp_dim, self.mid_dim)]
        if args.hi_layer > 2:
            for i in range(args.layer - 2):
                models += [nn.ReLU(), nn.Linear(self.mid_dim, self.mid_dim)]
        if args.hi_layer > 1:
            models += [nn.ReLU(), nn.Linear(self.mid_dim, self.out_dim)]

        self.base = nn.Sequential(*models)

    def forward(self, obs, goal, actions):
        x = torch.cat([obs, actions / self.max_action], dim=1)
        x = torch.cat([x, goal], dim=1)
        dist = self.base(x)
        return dist


class critic(nn.Module):
    def __init__(self, env_params, args, goal_dim):
        super(critic, self).__init__()
        self.max_action = env_params['action_max']
        self.inp_dim = env_params['low_dim'] + env_params['action'] + goal_dim
        self.out_dim = 1
        self.mid_dim = 400

        if args.layer == 1:
            models = [nn.Linear(self.inp_dim, self.out_dim)]
        else:
            models = [nn.Linear(self.inp_dim, self.mid_dim)]
        if args.layer > 2:
            for i in range(args.layer - 2):
                models += [nn.ReLU(), nn.Linear(self.mid_dim, self.mid_dim)]
        if args.layer > 1:
            models += [nn.ReLU(), nn.Linear(self.mid_dim, self.out_dim)]

        self.base = nn.Sequential(*models)

    def forward(self, obs, goal, actions):
        x = torch.cat([obs, actions / self.max_action], dim=1)
        x = torch.cat([x, goal], dim=1)
        dist = self.base(x)
        return dist

class Critic_double(nn.Module):
    def __init__(self, env_params, args):
        super(Critic_double, self).__init__()
        self.max_action = env_params['action_max']
        self.inp_dim = env_params['obs'] + env_params['action'] + env_params['goal']
        self.out_dim = 1
        self.mid_dim = 400

        if args.layer == 1:
            models = [nn.Linear(self.inp_dim, self.out_dim)]
        else:
            models = [nn.Linear(self.inp_dim, self.mid_dim)]
        if args.layer > 2:
            for i in range(args.layer - 2):
                models += [nn.ReLU(), nn.Linear(self.mid_dim, self.mid_dim)]
        if args.layer > 1:
            models += [nn.ReLU(), nn.Linear(self.mid_dim, self.out_dim)]

        self.base = nn.Sequential(*models)

        if args.layer == 1:
            models1 = [nn.Linear(self.inp_dim, self.out_dim)]
        else:
            models1 = [nn.Linear(self.inp_dim, self.mid_dim)]
        if args.layer > 2:
            for i in range(args.layer - 2):
                models1 += [nn.ReLU(), nn.Linear(self.mid_dim, self.mid_dim)]
        if args.layer > 1:
            models1 += [nn.ReLU(), nn.Linear(self.mid_dim, self.out_dim)]

        self.base1 = nn.Sequential(*models1)

    def forward(self, obs, goal, actions):
        x = torch.cat([obs, actions / self.max_action], dim=1)
        x = torch.cat([x, goal], dim=1)
        dist = self.base(x)
        dist1 = self.base1(x)
        return dist, dist1


class criticWrapper(nn.Module):
    def __init__(self, env_params, args, goal_dim):
        super(criticWrapper, self).__init__()
        self.base = critic(env_params, args, goal_dim)
        self.args = args
        self.gamma = args.gamma

    def forward(self, obs, goal, actions):
        dist = self.base(obs, goal, actions)
        self.alpha = np.log(self.gamma)
        return -(1 - torch.exp(dist * self.alpha)) / (1 - self.gamma)


class doubleWrapper(nn.Module):
    def __init__(self, env_params, args):
        super(doubleWrapper, self).__init__()
        self.base = Critic_double(env_params, args)
        self.args = args
        self.gamma = args.gamma

    def forward(self, obs, goal, actions):
        dist, dist1 = self.base(obs, goal, actions)
        self.alpha = np.log(self.gamma)
        return -(1 - torch.exp(dist * self.alpha)) / (1 - self.gamma), -(1 - torch.exp(dist1 * self.alpha)) / (1 - self.gamma)

    def Q1(self, obs, goal, actions):
        dist, _ = self.base(obs, goal, actions)
        self.alpha = np.log(self.gamma)
        return -(1 - torch.exp(dist * self.alpha)) / (1 - self.gamma)


class EmbedNet(nn.Module):
    def __init__(self, env_params, args):
        super(EmbedNet, self).__init__()
        self.max_action = env_params['action_max']
        self.obs_dim = env_params['obs'] + env_params['action']
        self.goal_dim = env_params['goal']
        self.out_dim = 128
        self.mid_dim = 400

        if args.layer == 1:
            obs_models = [nn.Linear(self.obs_dim, self.out_dim)]
            goal_models = [nn.Linear(self.goal_dim, self.out_dim)]
        else:
            obs_models = [nn.Linear(self.obs_dim, self.mid_dim)]
            goal_models = [nn.Linear(self.goal_dim, self.mid_dim)]
        if args.layer > 2:
            for i in range(args.layer - 2):
                obs_models += [nn.ReLU(), nn.Linear(self.mid_dim, self.mid_dim)]
                goal_models += [nn.ReLU(), nn.Linear(self.mid_dim, self.mid_dim)]
        if args.layer > 1:
            obs_models += [nn.ReLU(), nn.Linear(self.mid_dim, self.out_dim)]
            goal_models += [nn.ReLU(), nn.Linear(self.mid_dim, self.out_dim)]

        self.obs_encoder = nn.Sequential(*obs_models)
        self.goal_encoder = nn.Sequential(*goal_models)
        self.metric = initialize_metrics(args.metric, self.out_dim)

    def forward(self, obs, goal, actions):
        s = torch.cat([obs, actions / self.max_action], dim=1)
        s = self.obs_encoder(s)
        g = self.goal_encoder(goal)
        dist = self.metric(s, g)
        return dist


class Qnet(nn.Module):
    def __init__(self, env_params, args):
        super(Qnet, self).__init__()
        self.mid_dim = 100
        self.metric = args.metric

        self.action_n = env_params['action_dim']
        self.obs_fc1 = nn.Linear(env_params['obs'], 256)
        self.obs_fc2 = nn.Linear(256, self.mid_dim * self.action_n)

        self.goal_fc1 = nn.Linear(env_params['goal'], 256)
        self.goal_fc2 = nn.Linear(256, self.mid_dim)
        if self.metric == 'MLP':
            self.mlp = nn.Sequential(
                nn.Linear(self.mid_dim * (self.action_n + 1), 128),
                nn.ReLU(),
                nn.Linear(128, self.action_n),
            )

    def forward(self, obs, goal):
        s = F.relu(self.obs_fc1(obs))
        s = F.relu(self.obs_fc2(s))
        s = s.view(s.size(0), self.action_n, self.mid_dim)

        g = F.relu(self.goal_fc1(goal))
        g = F.relu(self.goal_fc2(g))

        if self.metric == 'L1':
            dist = torch.abs(s - g[:, None, :]).sum(dim=2)
        elif self.metric == 'dot':
            dist = -(s * g[:, None, :]).sum(dim=2)
        elif self.metric == 'L2':
            dist = ((torch.abs(s - g[:, None, :]) ** 2).sum(dim=2) + 1e-14) ** 0.5
        elif self.metric == 'MLP':
            s = s.view(s.size(0), -1)
            x = torch.cat([s, g], dim=1)
            dist = self.mlp(x)
        else:
            raise NotImplementedError
        return dist


class QNetWrapper(nn.Module):
    def __init__(self, env_params, args):
        super(QNetWrapper, self).__init__()
        self.base = Qnet(env_params, args)
        self.args = args
        self.gamma = args.gamma

    def forward(self, obs, goal):
        dist = self.base(obs, goal)
        self.alpha = np.log(self.gamma)
        qval = -(1 - torch.exp(dist * self.alpha)) / (1 - self.gamma)
        return qval


class EmbedNetWrapper(nn.Module):
    def __init__(self, env_params, args):
        super(EmbedNetWrapper, self).__init__()
        self.base = EmbedNet(env_params, args)
        self.args = args
        self.gamma = args.gamma

    def forward(self, obs, goal, actions):
        dist = self.base(obs, goal, actions)
        self.alpha = np.log(self.gamma)
        return -(1 - torch.exp(dist * self.alpha)) / (1 - self.gamma)


class RepresentationNetwork(nn.Module):
    def __init__(self, env_params, layer, abs_range, out_dim):
        super(RepresentationNetwork, self).__init__()
        self.obs_dim = env_params['obs']
        self.out_dim = out_dim
        self.mid_dim = 100

        if layer == 1:
            obs_models = [nn.Linear(self.obs_dim, self.out_dim)]
        else:
            obs_models = [nn.Linear(self.obs_dim, self.mid_dim)]
        if layer > 2:
            for i in range(layer - 2):
                obs_models += [nn.ReLU(), nn.Linear(self.mid_dim, self.mid_dim)]
        if layer > 1:
            obs_models += [nn.ReLU(), nn.Linear(self.mid_dim, self.out_dim)]

        self.obs_encoder = nn.Sequential(*obs_models)
        self.abs_range = abs_range

    def forward(self, obs):
        if len(obs.shape) is 1:
            obs = obs.unsqueeze(0)
        s = self.obs_encoder(obs)
        return s


class DynamicsNetwork(nn.Module):
    def __init__(self, env_params, abs_range, out_dim, tanh_output, use_prob, device):
        super(DynamicsNetwork, self).__init__()
        self.obs_dim = env_params['obs']
        self.out_dim = out_dim
        self.mid_dim = 100

        # obs encoder
        obs_models = [nn.Linear(self.obs_dim, self.mid_dim)]
        obs_models += [nn.ReLU(), nn.Linear(self.mid_dim, self.mid_dim)]
        obs_models += [nn.ReLU(), nn.Linear(self.mid_dim, self.out_dim)]

        self.obs_encoder = nn.Sequential(*obs_models)
        self.abs_range = abs_range

        # goal input
        self.goal_input = nn.Linear(out_dim, int(self.mid_dim / 2))
        self.dynamics_layer = nn.Linear(int(self.mid_dim / 2) + self.out_dim, self.mid_dim)
        self.output_layer = nn.Linear(self.mid_dim, self.out_dim)

        self.tanh_output = tanh_output
        self.probabilistic_output = use_prob
        self.device = device

    def phi(self, obs):
        if len(obs.shape) is 1:
            obs = obs.unsqueeze(0)
        s = self.obs_encoder(obs)
        return s

    def forward(self, obs, hi_action):
        latent_s = self.obs_encoder(obs)
        action_out = self.goal_input(hi_action)
        action_out = F.relu(action_out)
        x = torch.cat([latent_s, action_out], 1)
        x = self.dynamics_layer(x)
        x = F.relu(x)
        x = self.output_layer(x)

        if self.tanh_output:
            x = self.abs_range * torch.tanh(x)
            return x
        elif self.probabilistic_output:
            std_dev = torch.ones(x.shape[0], self.out_dim).to(self.device)
            return D.Independent(D.Normal(x, std_dev), 1)
        else:
           return x
