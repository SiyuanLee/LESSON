from .maze_env import MazeEnv
from .swimmer import SwimmerEnv


class SwimmerMazeEnv(MazeEnv):
    MODEL_CLASS = SwimmerEnv
