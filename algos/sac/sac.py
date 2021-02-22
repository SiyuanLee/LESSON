import os
import torch
import torch.nn.functional as F
from torch.optim import Adam
from algos.sac.utils import soft_update, hard_update
from algos.sac.model import GaussianPolicy, QNetwork, DeterministicPolicy, QNetwork_phi


class SAC(object):
    def __init__(self, num_inputs, action_space, args, pri_replay, goal_dim, gradient_flow_value, abs_range, tanh_output):

        self.gamma = args.gamma
        self.tau = args.tau
        self.alpha = args.alpha
        self.pri_replay = pri_replay

        self.policy_type = args.policy
        self.target_update_interval = args.target_update_interval
        self.automatic_entropy_tuning = args.automatic_entropy_tuning

        self.device = args.device
        self.gradient_flow_value = gradient_flow_value

        if not gradient_flow_value:
            self.critic = QNetwork(num_inputs, action_space.shape[0], args.hidden_size).to(device=self.device)
            self.critic_optim = Adam(self.critic.parameters(), lr=args.lr)

            self.critic_target = QNetwork(num_inputs, action_space.shape[0], args.hidden_size).to(self.device)
            hard_update(self.critic_target, self.critic)
        else:
            self.critic = QNetwork_phi(num_inputs, action_space.shape[0], args.hidden_size, abs_range, tanh_output).to(device=self.device)
            self.critic_optim = Adam(self.critic.parameters(), lr=args.lr)

            self.critic_target = QNetwork_phi(num_inputs, action_space.shape[0], args.hidden_size, abs_range, tanh_output).to(self.device)
            hard_update(self.critic_target, self.critic)


        if self.policy_type == "Gaussian":
            # Target Entropy = −dim(A) (e.g. , -6 for HalfCheetah-v2) as given in the paper
            if self.automatic_entropy_tuning is True:
                self.target_entropy = -torch.prod(torch.Tensor(action_space.shape).to(self.device)).item()
                self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
                self.alpha_optim = Adam([self.log_alpha], lr=args.lr)

            self.policy = GaussianPolicy(num_inputs, action_space.shape[0], args.hidden_size, action_space, goal_dim).to(self.device)
            self.policy_optim = Adam(self.policy.parameters(), lr=args.lr)

            self.policy_target = GaussianPolicy(num_inputs, action_space.shape[0], args.hidden_size, action_space,
                                         goal_dim).to(self.device)
            hard_update(self.policy_target, self.policy)


        else:
            self.alpha = 0
            self.automatic_entropy_tuning = False
            self.policy = DeterministicPolicy(num_inputs, action_space.shape[0], args.hidden_size, action_space).to(self.device)
            self.policy_optim = Adam(self.policy.parameters(), lr=args.lr)

    def select_action(self, state, evaluate=False):
        state = torch.FloatTensor(state).to(self.device).unsqueeze(0)
        if evaluate is False:
            action, _, _ = self.policy.sample(state)
        else:
            _, _, action = self.policy.sample(state)
        return action.detach().cpu().numpy()[0]

    def update_parameters(self, memory, batch_size, env_params, hi_sparse, feature_data):
        # Sample a batch from memory
        if self.pri_replay:
            state_batch, action_batch, reward_batch, next_state_batch, mask_batch = memory.pri_sample(batch_size=batch_size)
        else:
            state_batch, action_batch, reward_batch, next_state_batch, mask_batch = memory.sample(batch_size=batch_size)

        state_batch = torch.FloatTensor(state_batch).to(self.device)
        next_state_batch = torch.FloatTensor(next_state_batch).to(self.device)
        action_batch = torch.FloatTensor(action_batch).to(self.device)
        reward_batch = torch.FloatTensor(reward_batch).to(self.device).unsqueeze(1)
        mask_batch = torch.FloatTensor(mask_batch).to(self.device).unsqueeze(1)

        with torch.no_grad():
            next_state_action, next_state_log_pi, _ = self.policy.sample(next_state_batch)
            qf1_next_target, qf2_next_target = self.critic_target(next_state_batch, next_state_action)
            min_qf_next_target = torch.min(qf1_next_target, qf2_next_target) - self.alpha * next_state_log_pi
            # print("min_qf_target", min_qf_next_target.shape)
            next_q_value = reward_batch + mask_batch * self.gamma * (min_qf_next_target)
            if hi_sparse:
                # clip target value
                next_q_value = torch.clamp(next_q_value, -env_params['max_timesteps'], 0.)
        qf1, qf2 = self.critic(state_batch, action_batch)  # Two Q-functions to mitigate positive bias in the policy improvement step
        # print("qf1", qf1.shape)
        # print("next_q", next_q_value.shape)
        qf1_loss = F.mse_loss(qf1, next_q_value)  # JQ = 𝔼(st,at)~D[0.5(Q1(st,at) - r(st,at) - γ(𝔼st+1~p[V(st+1)]))^2]
        qf2_loss = F.mse_loss(qf2, next_q_value)  # JQ = 𝔼(st,at)~D[0.5(Q1(st,at) - r(st,at) - γ(𝔼st+1~p[V(st+1)]))^2]

        pi, log_pi, _ = self.policy.sample(state_batch)

        qf1_pi, qf2_pi = self.critic(state_batch, pi)
        min_qf_pi = torch.min(qf1_pi, qf2_pi)

        policy_loss = ((self.alpha * log_pi) - min_qf_pi).mean() # Jπ = 𝔼st∼D,εt∼N[α * logπ(f(εt;st)|st) − Q(st,f(εt;st))]
        if feature_data is not None:
            if self.gradient_flow_value:
                obs, obs_next = self.critic.phi(feature_data[0]), self.critic.phi(feature_data[1])
                min_dist = torch.clamp((obs - obs_next).pow(2).mean(dim=1), min=0.)
                hi_obs, hi_obs_next = self.critic.phi(feature_data[2]), self.critic.phi(feature_data[3])
                max_dist = torch.clamp(1 - (hi_obs - hi_obs_next).pow(2).mean(dim=1), min=0.)
                representation_loss = (min_dist + max_dist).mean()
                qf1_loss = qf1_loss * 0.1 + representation_loss
            else:
                obs, obs_next = self.policy.phi(feature_data[0]), self.policy.phi(feature_data[1])
                min_dist = torch.clamp((obs - obs_next).pow(2).mean(dim=1), min=0.)
                hi_obs, hi_obs_next = self.policy.phi(feature_data[2]), self.policy.phi(feature_data[3])
                max_dist = torch.clamp(1 - (hi_obs - hi_obs_next).pow(2).mean(dim=1), min=0.)
                representation_loss = (min_dist + max_dist).mean()
                policy_loss += representation_loss


        self.critic_optim.zero_grad()
        qf1_loss.backward()
        self.critic_optim.step()

        self.critic_optim.zero_grad()
        qf2_loss.backward()
        self.critic_optim.step()

        self.policy_optim.zero_grad()
        policy_loss.backward()
        self.policy_optim.step()

        if self.automatic_entropy_tuning:
            alpha_loss = -(self.log_alpha * (log_pi + self.target_entropy).detach()).mean()

            self.alpha_optim.zero_grad()
            alpha_loss.backward()
            self.alpha_optim.step()

            self.alpha = self.log_alpha.exp()
            alpha_tlogs = self.alpha.clone() # For TensorboardX logs
        else:
            alpha_loss = torch.tensor(0.).to(self.device)
            alpha_tlogs = torch.tensor(self.alpha) # For TensorboardX logs


        soft_update(self.critic_target, self.critic, self.tau)
        soft_update(self.policy_target, self.policy, self.tau)

        return qf1_loss.item(), qf2_loss.item(), policy_loss.item(), alpha_loss.item(), alpha_tlogs.item()

    # Save model parameters
    def save_model(self, env_name, suffix="", actor_path=None, critic_path=None):
        if not os.path.exists('models/'):
            os.makedirs('models/')

        if actor_path is None:
            actor_path = "models/sac_actor_{}_{}".format(env_name, suffix)
        if critic_path is None:
            critic_path = "models/sac_critic_{}_{}".format(env_name, suffix)
        print('Saving models to {} and {}'.format(actor_path, critic_path))
        torch.save(self.policy.state_dict(), actor_path)
        torch.save(self.critic.state_dict(), critic_path)

    # Load model parameters
    def load_model(self, actor_path, critic_path):
        print('Loading models from {} and {}'.format(actor_path, critic_path))
        if actor_path is not None:
            self.policy.load_state_dict(torch.load(actor_path))
        if critic_path is not None:
            self.critic.load_state_dict(torch.load(critic_path))

