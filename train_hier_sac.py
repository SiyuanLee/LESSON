import numpy as np
import gym
from arguments.arguments_hier_sac import get_args_ant, get_args_chain
from algos.hier_sac import hier_sac_agent
from goal_env.mujoco import *
import random
import torch


def get_env_params(env):
    obs = env.reset()
    # close the environment
    params = {'obs': obs['observation'].shape[0], 'goal': obs['desired_goal'].shape[0],
              'action': env.action_space.shape[0], 'action_max': env.action_space.high[0],
              'max_timesteps': env._max_episode_steps}
    return params


def launch(args):
    # create the ddpg_agent
    env = gym.make(args.env_name)
    test_env = gym.make(args.test)
    # if args.env_name == "AntPush-v1":
    #     test_env1 = gym.make("AntPushTest1-v1")
    #     test_env2 = gym.make("AntPushTest2-v1")
    # elif args.env_name == "AntMaze1-v1":
    #     test_env1 = gym.make("AntMaze1Test1-v1")
    #     test_env2 = gym.make("AntMaze1Test2-v1")
    # else:
    test_env1 = test_env2 = None
    print("test_env", test_env1, test_env2)

    # set random seeds for reproduce
    env.seed(args.seed)
    if args.env_name != "NChain-v1":
        env.env.env.wrapped_env.seed(args.seed)
        test_env.env.env.wrapped_env.seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.device is not 'cpu':
        torch.cuda.manual_seed(args.seed)
    gym.spaces.prng.seed(args.seed)
    # get the environment parameters
    if args.env_name[:3] in ["Ant", "Poi", "Swi"]:
        env.env.env.visualize_goal = args.animate
        test_env.env.env.visualize_goal = args.animate
    env_params = get_env_params(env)
    env_params['max_test_timesteps'] = test_env._max_episode_steps
    # create the ddpg agent to interact with the environment
    sac_trainer = hier_sac_agent(args, env, env_params, test_env, test_env1, test_env2)
    if args.eval:
        if not args.resume:
            print("random policy !!!")
        # sac_trainer._eval_hier_agent(test_env)
        # sac_trainer.vis_hier_policy()
        # sac_trainer.cal_slow()
        # sac_trainer.visualize_representation(100)
        # sac_trainer.vis_learning_process()
        # sac_trainer.picvideo('fig/final/', (1920, 1080))
    else:
        sac_trainer.learn()


# get the params
args = get_args_ant()
# args = get_args_chain()
# args = get_args_fetch()
# args = get_args_point()
if __name__ == '__main__':
    launch(args)
