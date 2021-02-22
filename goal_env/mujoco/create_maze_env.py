from .ant_maze_env import AntMazeEnv
from .point_maze_env import PointMazeEnv
from .swimmer_maze_env import SwimmerMazeEnv
from collections import OrderedDict
import gym
import numpy as np
import copy
from gym import Wrapper
from gym.envs.registration import EnvSpec


class GoalWrapper(Wrapper):
    def __init__(self, env, maze_size_scaling, random_start, low, high, fix_goal=True, top_down=False, test=None):
        super(GoalWrapper, self).__init__(env)
        ob_space = env.observation_space
        self.maze_size_scaling = maze_size_scaling

        row_num, col_num = len(self.env.MAZE_STRUCTURE), len(self.env.MAZE_STRUCTURE[0])
        contain_r = [1 if "r" in row else 0 for row in self.env.MAZE_STRUCTURE]
        row_r = contain_r.index(1)
        col_r = self.env.MAZE_STRUCTURE[row_r].index("r")
        y_low = (0.5 - row_r) * self.maze_size_scaling
        x_low = (0.5 - col_r) * self.maze_size_scaling
        y_high = (row_num - 1.5 - row_r) * self.maze_size_scaling
        x_high = (col_num - 1.5 - col_r) * self.maze_size_scaling
        self.maze_low = maze_low = np.array([x_low, y_low],
                            dtype=ob_space.dtype)
        self.maze_high = maze_high = np.array([x_high, y_high],
                             dtype=ob_space.dtype)
        print("maze_low, maze_high", self.maze_low, self.maze_high)

        goal_low, goal_high = maze_low, maze_high

        self.goal_space = gym.spaces.Box(low=goal_low, high=goal_high)
        self.maze_space = gym.spaces.Box(low=maze_low, high=maze_high)

        if self.env._maze_id == "Fall":
            self.goal_dim = 3
        else:
            self.goal_dim = goal_low.size
        print("goal_dim in create_maze", self.goal_dim)
        self.distance_threshold = 1.5
        print("distance threshold in create_maze", self.distance_threshold)

        self.observation_space = gym.spaces.Dict(OrderedDict({
            'observation': ob_space,
            'desired_goal': self.goal_space,
            'achieved_goal': self.goal_space,
        }))
        self.random_start = random_start

        # fix goal
        self.fix_goal = fix_goal
        print("fix goal", self.fix_goal)
        contain_g = [1 if "g" in row else 0 for row in self.env.MAZE_STRUCTURE]
        if 1 in contain_g and self.fix_goal and test == "Test":
            row = contain_g.index(1)
            col = self.env.MAZE_STRUCTURE[row].index("g")
            y = (row - row_r) * self.maze_size_scaling
            x = (col - col_r) * self.maze_size_scaling
            self.fix_goal_xy = np.array([x, y])
            if env._maze_id == "Fall":
                self.fix_goal_xy = np.concatenate((self.fix_goal_xy, [self.maze_size_scaling * 0.5 + 0.5]))
            print("fix goal xy", self.fix_goal_xy)
        elif test == "Test1":
            if env._maze_id == "Push":
                self.fix_goal_xy = np.array([-4, 0])
            elif env._maze_id == "Maze1":
                self.fix_goal_xy = np.array([8, 0])
            else:
                print("Unknown env", env._maze_id)
                assert False
            print("fix goal xy", self.fix_goal_xy)
        elif test == "Test2":
            if env._maze_id == "Push":
                self.fix_goal_xy = np.array([-4, 4])
            elif env._maze_id == "Maze1":
                self.fix_goal_xy = np.array([8, 8])
            else:
                print("Unknown env", env._maze_id)
                assert False
            print("fix goal xy", self.fix_goal_xy)
        else:
            # get vacant rowcol
            structure = self.env.MAZE_STRUCTURE
            self.vacant_rowcol = []
            for i in range(len(structure)):
                for j in range(len(structure[0])):
                    if structure[i][j] not in [1, -1, 'r']:
                        self.vacant_rowcol.append((i, j))
        self.reward_type = "dense"

        self.top_down = top_down

    def step(self, action):
        observation, reward, _, info = self.env.step(action)
        out = {'observation': observation,
               'desired_goal': self.goal,
               # 'achieved_goal': observation[..., 3:5]}
               'achieved_goal': observation[..., :self.goal_dim]}
        distance = np.linalg.norm(observation[..., :self.goal_dim] - self.goal[..., :self.goal_dim], axis=-1)
        info['is_success'] = done = (distance < self.distance_threshold)
        if self.reward_type == "sparse":
            reward = -(distance > self.distance_threshold).astype(np.float32)
        else:
            # normlization
            reward = -distance * 0.1
        if self.top_down:
            mask = np.array([0.0] * 2 + [1.0] * (out['observation'].shape[0] - 2))
            out['observation'] = out['observation'] * mask
        return out, reward, done, info

    def reset(self):
        if self.fix_goal:
            self.goal = self.fix_goal_xy
        else:
            self.goal = self.goal_space.sample()
            if self.env._maze_id == "Push":
                while (self.env.old_invalid_goal(self.goal[:2])):
                    self.goal = self.goal_space.sample()
            else:
                while (self.env.invalid_goal(self.goal[:2])):
                    self.goal = self.goal_space.sample()
            if self.env._maze_id == "Fall":
                self.goal = np.concatenate((self.goal, [self.maze_size_scaling * 0.5 + 0.5]))
        observation = self.env.reset(self.goal)

        # random start a position without collision
        if self.random_start:
            xy = self.maze_space.sample()
            while (self.env._is_in_collision(xy)):
                xy = self.maze_space.sample()
            self.env.wrapped_env.set_xy(xy)
            observation = self.env._get_obs()

        out = {'observation': observation, 'desired_goal': self.goal}
        out['achieved_goal'] = observation[..., :self.goal_dim]
        # out['achieved_goal'] = observation[..., 3:5]
        if self.top_down:
            # print("obs", out['observation'].shape)
            mask = np.array([0.0] * 2 + [1.0] * (out['observation'].shape[0] - 2))
            out['observation'] = out['observation'] * mask
        return out


def create_maze_env(env_name=None, top_down_view=False, maze_size_scaling=4, random_start=True, goal_args=None,
                    fix_goal=True, test=None):
    n_bins = 0
    if env_name.startswith('Ego'):
        n_bins = 8
        env_name = env_name[3:]
    if env_name.startswith('Ant'):
        manual_collision = True
        cls = AntMazeEnv
        env_name = env_name[3:]
        maze_size_scaling = maze_size_scaling
    elif env_name.startswith('Point'):
        cls = PointMazeEnv
        manual_collision = True
        env_name = env_name[5:]
        maze_size_scaling = maze_size_scaling
    elif env_name.startswith('Swimmer'):
        cls = SwimmerMazeEnv
        manual_collision = True
        env_name = env_name[7:]
        maze_size_scaling = maze_size_scaling
    else:
        assert False, 'unknown env %s' % env_name

    observe_blocks = False
    put_spin_near_agent = False
    if env_name == 'Maze':
        maze_id = 'Maze'
    elif env_name == 'Maze1':
        maze_id = 'Maze1'
        maze_size_scaling = 4
    elif env_name == 'Push':
        maze_id = 'Push'
        manual_collision = True
        maze_size_scaling = 4
    elif env_name == 'Fall':
        maze_id = 'Fall'
    elif env_name == 'Block':
        maze_id = 'Block'
        put_spin_near_agent = True
        observe_blocks = True
    elif env_name == 'BlockMaze':
        maze_id = 'BlockMaze'
        put_spin_near_agent = True
        observe_blocks = True
    else:
        raise ValueError('Unknown maze environment %s' % env_name)

    gym_mujoco_kwargs = {
        'maze_id': maze_id,
        'n_bins': n_bins,
        'observe_blocks': observe_blocks,
        'put_spin_near_agent': put_spin_near_agent,
        'top_down_view': top_down_view,
        'manual_collision': manual_collision,
        'maze_size_scaling': maze_size_scaling,
    }
    gym_env = cls(**gym_mujoco_kwargs)
    # gym_env.reset()
    # goal_args = np.array(goal_args) / 8 * maze_size_scaling
    return GoalWrapper(gym_env, maze_size_scaling, random_start, *goal_args, fix_goal=fix_goal, top_down=top_down_view, test=test)
