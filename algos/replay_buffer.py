import threading
import numpy as np
import torch

"""
the replay buffer here is basically from the openai baselines code
"""


class replay_buffer:
    def __init__(self, env_params, buffer_size, sample_func, name='max_timesteps'):
        self.env_params = env_params
        self.T = env_params[name]
        if name == 'max_timesteps':
            # low level
            goal_dim = env_params['real_goal_dim']
            action_dim = self.env_params['action']
            obs_dim = self.env_params['low_dim']
        else:
            # high level
            goal_dim = env_params['goal']
            action_dim = env_params['real_goal_dim']
            obs_dim = self.env_params['hi_dim']
        self.size = buffer_size // self.T
        # memory management
        self.current_size = 0
        self.n_transitions_stored = 0
        self.sample_func = sample_func
        # create the buffer to store info
        self.buffers = {'obs': np.empty([self.size, self.T + 1, obs_dim]),
                        'ag': np.empty([self.size, self.T + 1, goal_dim]),
                        'g': np.empty([self.size, self.T, goal_dim]),
                        'actions': np.empty([self.size, self.T, action_dim]),
                        'success': np.empty([self.size]),
                        'done': np.empty([self.size, self.T, 1])
                        }

        self.position = 0  # record the index to update

    # store the episode
    def store_episode(self, episode_batch):
        mb_obs, mb_ag, mb_g, mb_actions, success, done = episode_batch
        batch_size = mb_obs.shape[0]
        idxs = self._get_storage_idx(inc=batch_size)
        # store the informations
        self.buffers['obs'][idxs] = mb_obs
        self.buffers['ag'][idxs] = mb_ag
        self.buffers['g'][idxs] = mb_g
        self.buffers['actions'][idxs] = mb_actions
        self.buffers['success'][idxs] = success
        self.buffers['done'][idxs] = done
        self.n_transitions_stored += self.T * batch_size

    # sample the data from the replay buffer
    def sample(self, batch_size):
        temp_buffers = {}
        for key in self.buffers.keys():
            if key != 'success':
                temp_buffers[key] = self.buffers[key][:self.current_size]
        temp_buffers['obs_next'] = temp_buffers['obs'][:, 1:, :]
        temp_buffers['ag_next'] = temp_buffers['ag'][:, 1:, :]
        # sample transitions
        transitions = self.sample_func(temp_buffers, batch_size)
        return transitions

    def random_sample(self, batch_size):
        temp_buffers = {}
        for key in self.buffers.keys():
            temp_buffers[key] = self.buffers[key][:self.current_size]
        temp_buffers['obs_next'] = temp_buffers['obs'][:, 1:, :]
        temp_buffers['ag_next'] = temp_buffers['ag'][:, 1:, :]
        # sample transitions
        # print('start random sample', self.current_size)
        T = temp_buffers['actions'].shape[1]  # 50 steps per traj
        rollout_batch_size = temp_buffers['actions'].shape[0]  # 2 trajs
        batch_size = batch_size  # target batches we want to sample
        episode_idxs = np.random.randint(0, rollout_batch_size, batch_size)
        # which traj to sample
        t_samples = np.random.randint(T, size=batch_size)
        # which step to sample
        transitions = {key: temp_buffers[key][episode_idxs, t_samples].copy() for key in temp_buffers.keys()}
        transitions = {k: transitions[k].reshape(batch_size, *transitions[k].shape[1:]) for k in transitions.keys()}
        return transitions

    def sample_traj(self, batch_size):
        temp_buffers = {}
        for key in self.buffers.keys():
            temp_buffers[key] = self.buffers[key][:self.current_size]
        temp_buffers['obs_next'] = temp_buffers['obs'][:, 1:, :]
        temp_buffers['ag_next'] = temp_buffers['ag'][:, 1:, :]
        T = temp_buffers['actions'].shape[1]  # 50 steps per traj
        num_traj = temp_buffers['actions'].shape[0]  # number of all the trajs
        episode_idxs = np.random.randint(0, num_traj, batch_size)
        traj = {key: temp_buffers[key][episode_idxs, :].copy() for key in temp_buffers.keys()}
        # remember obs and ag has a larger shape
        return traj

    def get_all_data(self):
        temp_buffers = {}
        for key in self.buffers.keys():
            temp_buffers[key] = self.buffers[key][:self.current_size]
        temp_buffers['obs_next'] = temp_buffers['obs'][:, 1:, :]
        temp_buffers['ag_next'] = temp_buffers['ag'][:, 1:, :]
        return temp_buffers

    def _get_storage_idx(self, inc=None):
        inc = inc or 1
        assert inc == 1
        if self.current_size + inc <= self.size:
            idx = np.arange(self.current_size, self.current_size + inc)
        elif self.current_size < self.size:
            overflow = inc - (self.size - self.current_size)
            idx_a = np.arange(self.current_size, self.size)
            idx_b = np.random.randint(0, self.current_size, overflow)
            idx = np.concatenate([idx_a, idx_b])
        else:
            idx = np.array([self.position])
            # idx = np.random.randint(0, self.size, inc)
        self.current_size = min(self.size, self.current_size + inc)
        self.position = (self.position + 1) % self.size
        if inc == 1:
            idx = idx[0]
        return idx

    # update achieved_goal in the buffer
    def update_ag(self, phi, device):
        all_obs = self.buffers['obs'][:self.current_size].copy()
        obs = all_obs.reshape(-1, all_obs.shape[2])
        obs_tensor = torch.Tensor(obs).to(device)
        ag = phi(obs_tensor).detach().cpu().numpy()
        goal_dim = self.buffers['ag'].shape[-1]
        ag_new = ag.reshape(self.current_size, -1, goal_dim)
        self.buffers["ag"][:self.current_size] = ag_new


class replay_buffer_energy:
    def __init__(self, env_params, buffer_size, sample_func, env_name, name='max_timesteps'):
        self.env_params = env_params
        self.T = env_params[name]
        if name == 'max_timesteps':
            goal_dim = env_params['real_goal_dim']
            action_dim = self.env_params['action']
        else:
            goal_dim = env_params['goal']
            action_dim = env_params['real_goal_dim']
        self.size = buffer_size // self.T
        # memory management
        self.current_size = 0
        self.n_transitions_stored = 0
        self.sample_func = sample_func
        # create the buffer to store info
        self.buffers = {'obs': np.empty([self.size, self.T + 1, self.env_params['obs']]),
                        'ag': np.empty([self.size, self.T + 1, goal_dim]),
                        'g': np.empty([self.size, self.T, goal_dim]),
                        'actions': np.empty([self.size, self.T, action_dim]),
                        'e': np.empty([self.size, 1]),  # energy
                        }
        self.env_name = env_name

    # store the episode
    def store_episode(self, episode_batch, w_potential=1.0, w_linear=1.0, clip_energy=0.5):
        mb_obs, mb_ag, mb_g, mb_actions = episode_batch
        batch_size = mb_obs.shape[0]
        idxs = self._get_storage_idx(inc=batch_size)
        # store the informations
        self.buffers['obs'][idxs] = mb_obs
        self.buffers['ag'][idxs] = mb_ag
        self.buffers['g'][idxs] = mb_g
        self.buffers['actions'][idxs] = mb_actions
        self.n_transitions_stored += self.T * batch_size

        buffers = {}
        for key in self.buffers.keys():
            buffers[key] = self.buffers[key][idxs][None].copy()

        # calculate energy
        if self.env_name[:5] == 'Fetch':
            g, m, delta_t = 9.81, 1, 0.04
            if self.env_name[:9] == 'FetchPush':
                potential_energy = 0.
            else:
                height = buffers['ag'][:, :, 2]
                height_0 = np.repeat(height[:, 0].reshape(-1, 1), height[:, 1::].shape[1], axis=1)
                height = height[:, 1::] - height_0
                potential_energy = g * m * height
            diff = np.diff(buffers['ag'], axis=1)
            velocity = diff / delta_t
            kinetic_energy = 0.5 * m * np.power(velocity, 2)
            kinetic_energy = np.sum(kinetic_energy, axis=2)
            energy_totoal = w_potential * potential_energy + w_linear * kinetic_energy
            energy_diff = np.diff(energy_totoal, axis=1)
            energy_transition = energy_totoal.copy()
            energy_transition[:, 1::] = energy_diff.copy()
            energy_transition = np.clip(energy_transition, 0, clip_energy)
            energy_transition_total = np.sum(energy_transition, axis=1)
            energy_final = np.sum(energy_transition_total.reshape(-1, 1))
            self.buffers['e'][idxs, 0] = energy_final
        else:
            print('Trajectory Energy Function Not Implemented')

    # sample the data from the replay buffer
    def sample(self, batch_size):
        temp_buffers = {}
        for key in self.buffers.keys():
            temp_buffers[key] = self.buffers[key][:self.current_size]
        temp_buffers['obs_next'] = temp_buffers['obs'][:, 1:, :]
        temp_buffers['ag_next'] = temp_buffers['ag'][:, 1:, :]
        # sample transitions
        transitions = self.sample_func(temp_buffers, batch_size)
        return transitions

    def random_sample(self, batch_size):
        temp_buffers = {}
        for key in self.buffers.keys():
            temp_buffers[key] = self.buffers[key][:self.current_size]
        temp_buffers['obs_next'] = temp_buffers['obs'][:, 1:, :]
        temp_buffers['ag_next'] = temp_buffers['ag'][:, 1:, :]
        # sample transitions
        # print('start random sample', self.current_size)
        T = temp_buffers['actions'].shape[1]  # 50 steps per traj
        rollout_batch_size = temp_buffers['actions'].shape[0]  # 2 trajs
        batch_size = batch_size  # target batches we want to sample
        episode_idxs = np.random.randint(0, rollout_batch_size, batch_size)
        # which traj to sample
        t_samples = np.random.randint(T, size=batch_size)
        # which step to sample
        transitions = {key: temp_buffers[key][episode_idxs, t_samples].copy() for key in temp_buffers.keys()}
        transitions = {k: transitions[k].reshape(batch_size, *transitions[k].shape[1:]) for k in transitions.keys()}
        return transitions

    def sample_traj(self, batch_size):
        temp_buffers = {}
        for key in self.buffers.keys():
            temp_buffers[key] = self.buffers[key][:self.current_size]
        temp_buffers['obs_next'] = temp_buffers['obs'][:, 1:, :]
        temp_buffers['ag_next'] = temp_buffers['ag'][:, 1:, :]
        T = temp_buffers['actions'].shape[1]  # 50 steps per traj
        num_traj = temp_buffers['actions'].shape[0]  # number of all the trajs
        episode_idxs = np.random.randint(0, num_traj, batch_size)
        traj = {key: temp_buffers[key][episode_idxs, :].copy() for key in temp_buffers.keys()}
        # remember obs and ag has a larger shape
        return traj

    def get_all_data(self):
        temp_buffers = {}
        for key in self.buffers.keys():
            temp_buffers[key] = self.buffers[key][:self.current_size]
        temp_buffers['obs_next'] = temp_buffers['obs'][:, 1:, :]
        temp_buffers['ag_next'] = temp_buffers['ag'][:, 1:, :]
        return temp_buffers

    def _get_storage_idx(self, inc=None):
        inc = inc or 1
        if self.current_size + inc <= self.size:
            idx = np.arange(self.current_size, self.current_size + inc)
        elif self.current_size < self.size:
            overflow = inc - (self.size - self.current_size)
            idx_a = np.arange(self.current_size, self.size)
            idx_b = np.random.randint(0, self.current_size, overflow)
            idx = np.concatenate([idx_a, idx_b])
        else:
            idx = np.random.randint(0, self.size, inc)
        self.current_size = min(self.size, self.current_size + inc)
        if inc == 1:
            idx = idx[0]
        return idx
