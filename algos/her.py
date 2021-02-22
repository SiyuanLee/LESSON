import numpy as np


class her_sampler:
    def __init__(self, replay_strategy, replay_k, threshold, future_step, dense_reward, direction_reward, low_reward_coeff):
        self.replay_strategy = replay_strategy
        self.replay_k = replay_k
        if self.replay_strategy == 'future':
            self.future_p = 1 - (1. / (1 + replay_k))
        else:
            self.future_p = 0
        self.threshold = threshold
        self.furture_step = future_step
        self.border_index = None
        self.direction_reward = direction_reward
        # reward type not use in direction reward
        if not dense_reward:
            self.reward_type = 'sparse'
        else:
            self.reward_type = 'dense'
        self.reward_coeff = low_reward_coeff


    def reward_func(self, state, goal, info=None):
        assert state.shape == goal.shape
        dist = np.linalg.norm(state - goal, axis=-1)
        if self.reward_type == 'sparse':
            return -(dist > self.threshold).astype(np.float32)
        else:
            return -dist * self.reward_coeff

    def direction_reward_func(self, ag_next, goal, ag):

        # l2 distance reward
        assert ag.shape == goal.shape
        dist = np.linalg.norm(ag + goal - ag_next, axis=-1)
        return -dist

        # # cosine distance reward
        # a_direction = ag_next - ag  # achieved direction
        # cos_dist = np.sum(np.multiply(a_direction, goal), axis=1) / (
        #         (np.linalg.norm(a_direction, axis=1) * np.linalg.norm(goal, axis=1)) + 1e-6)
        # return cos_dist


    def sample_her_transitions(self, episode_batch, batch_size_in_transitions):
        T = episode_batch['actions'].shape[1]
        rollout_batch_size = episode_batch['actions'].shape[0]
        batch_size = batch_size_in_transitions
        # select which rollouts and which timesteps to be used
        episode_idxs = np.random.randint(0, rollout_batch_size, batch_size)
        t_samples = np.random.randint(T, size=batch_size)
        transitions = {key: episode_batch[key][episode_idxs, t_samples].copy() for key in episode_batch.keys()}
        # her idx
        her_indexes = np.where(np.random.uniform(size=batch_size) < self.future_p)
        # cheat in her for large step length

        target_index = np.minimum(T, t_samples + self.furture_step)
        future_offset = np.random.uniform(size=batch_size) * (target_index - t_samples)

        future_offset = future_offset.astype(int)
        future_t = (t_samples + 1 + future_offset)[her_indexes]
        # replace goal with achieved goal
        future_ag = episode_batch['ag'][episode_idxs[her_indexes], future_t]
        transitions['g'][her_indexes] = future_ag
        # to get the params to re-compute reward
        if not self.direction_reward:
            transitions['r'] = np.expand_dims(
                self.reward_func(transitions['ag_next'], transitions['g'],
                                 None), 1)
        else:
            transitions['r'] = np.expand_dims(
                self.direction_reward_func(transitions['ag_next'].copy(), transitions['g'].copy(),
                                           transitions['ag'].copy()), 1)
        transitions = {k: transitions[k].reshape(batch_size, *transitions[k].shape[1:]) for k in transitions.keys()}
        return transitions

    def sample_her_energy(self, episode_batch, batch_size_in_transitions, temperature=1.0):
        T = episode_batch['actions'].shape[1]
        rollout_batch_size = episode_batch['actions'].shape[0]
        batch_size = batch_size_in_transitions
        # select which rollouts and which timesteps to be used
        energy_trajectory = episode_batch['e']
        p_trajectory = np.power(energy_trajectory, 1 / (temperature + 1e-2))
        p_trajectory = p_trajectory / p_trajectory.sum()
        episode_idxs = np.random.choice(rollout_batch_size, size=batch_size, replace=True, p=p_trajectory.flatten())

        t_samples = np.random.randint(T, size=batch_size)

        transitions = {}
        for key in episode_batch.keys():
            if not key == 'e':
                transitions[key] = episode_batch[key][episode_idxs, t_samples].copy()

        # her idx
        her_indexes = np.where(np.random.uniform(size=batch_size) < self.future_p)
        # cheat in her for large step length

        target_index = np.minimum(T, t_samples + self.furture_step)
        future_offset = np.random.uniform(size=batch_size) * (target_index - t_samples)

        future_offset = future_offset.astype(int)
        future_t = (t_samples + 1 + future_offset)[her_indexes]
        # replace go with achieved goal
        future_ag = episode_batch['ag'][episode_idxs[her_indexes], future_t]
        transitions['g'][her_indexes] = future_ag
        # to get the params to re-compute reward
        if not self.direction_reward:
            transitions['r'] = np.expand_dims(
                self.reward_func(transitions['ag_next'], transitions['g'],
                                 None), 1)
        else:
            transitions['r'] = np.expand_dims(
                self.direction_reward_func(transitions['ag_next'], transitions['g'],
                                 transitions['ag']), 1)
        transitions = {k: transitions[k].reshape(batch_size, *transitions[k].shape[1:]) for k in transitions.keys()}
        return transitions

    def adjust_replay_k(self):
        if self.replay_k > 1:
            self.replay_k -= 1

        if self.replay_strategy == 'future':
            self.future_p = 1 - (1. / (1 + self.replay_k))
        else:
            self.future_p = 0
