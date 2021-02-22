import random
import numpy as np

class ReplayMemory:
    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer = []
        self.position = 0

    def push(self, state, action, reward, next_state, done, epoch):
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.position] = (state, action, reward, next_state, done, epoch+1)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done, _ = map(np.stack, zip(*batch))
        return state, action, reward, next_state, done

    def __len__(self):
        return len(self.buffer)

    def get_obs(self):
        obs = [x[0] for x in self.buffer]
        obs = np.array(obs)
        obs_next = [x[3] for x in self.buffer]
        obs_next = np.array(obs_next)
        return obs.copy(), obs_next.copy()

    def pri_sample(self, batch_size, temperature=1.):
        tmp_buffer = np.array(self.buffer)
        epoch = tmp_buffer[:, -1]
        p_trajectory = np.power(epoch, 1 / (temperature + 1e-2))
        p_trajectory = p_trajectory / p_trajectory.sum()
        p_trajectory = p_trajectory.astype(np.float64)
        idxs = np.random.choice(len(self.buffer), size=batch_size, replace=False, p=p_trajectory)
        batch = [self.buffer[i] for i in idxs]
        state, action, reward, next_state, done, _ = map(np.stack, zip(*batch))
        return state, action, reward, next_state, done

    def random_sample(self, batch_size):
        idxs = np.random.randint(0, len(self.buffer), batch_size)
        obs = [self.buffer[i][0] for i in idxs]
        obs = np.array(obs)
        obs_next = [self.buffer[i][3] for i in idxs]
        obs_next = np.array(obs_next)
        return obs, obs_next

class Array_ReplayMemory:
    def __init__(self, capacity, env_params):
        self.capacity = capacity
        action_dim = env_params['real_goal_dim']
        obs_dim = env_params['obs'] + env_params['goal']
        # create the buffer to store info
        self.buffers = {'obs': np.empty([capacity, obs_dim]),
                        'actions': np.empty([capacity, action_dim]),
                        'reward': np.empty([capacity]),
                        'next_obs': np.empty([capacity, obs_dim]),
                        'done': np.empty([capacity])
                        }
        self.position = 0
        self.current_size = 0

    def push(self, state, action, reward, next_state, done, epoch):
        self.buffers['obs'][self.position] = state
        self.buffers['actions'][self.position] = action
        self.buffers['reward'][self.position] = reward
        self.buffers['next_obs'][self.position] = next_state
        self.buffers['done'][self.position] = done

        self.position = (self.position + 1) % self.capacity
        if self.current_size + 1 < self.capacity:
            self.current_size += 1


    def sample(self, batch_size):
        idx = np.random.randint(0, self.current_size, batch_size)
        state = self.buffers['obs'][idx]
        action = self.buffers['actions'][idx]
        reward = self.buffers['reward'][idx]
        next_state = self.buffers['next_obs'][idx]
        done = self.buffers['done'][idx]
        return state, action, reward, next_state, done

    def __len__(self):
        return self.current_size




