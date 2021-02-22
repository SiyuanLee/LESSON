from gym.envs.registration import register
import sys

print("path", sys.argv[0].split('/')[-1], "!!!")
if sys.argv[0].split('/')[-1] in ["train_ddpg.py", "visitation_plot.py", "vis_fetch.py"]:
    from train_ddpg import args
elif sys.argv[0].split('/')[-1] == "train_hier_ddpg.py":
    from train_hier_ddpg import args
elif sys.argv[0].split('/')[-1] == "train_hier_sac.py":
    from train_hier_sac import args
elif sys.argv[0].split('/')[-1] == "train_hier_ppo.py":
    from train_hier_ppo import args
elif sys.argv[0].split('/')[-1] == "train_covering.py":
    from train_covering import args
else:
    raise Exception("Unknown main file !!!")

robots = ['Point', 'Ant', 'Swimmer']
task_types = ['Maze', 'Maze1', 'Push', 'Fall', 'Block', 'BlockMaze']
all_name = [x + y for x in robots for y in task_types]
random_start = False

if args.image:
    top_down = True
else:
    top_down = False

for name_t in all_name:
    # episode length
    if name_t == "AntMaze":
        max_timestep = 1000
    else:
        max_timestep = 500
    for Test in ['', 'Test', 'Test1', 'Test2']:

        if Test in ['Test', 'Test1', 'Test2']:
            fix_goal = True
        else:
            if name_t == "AntBlock":
                fix_goal = True
            else:
                fix_goal = False
        goal_args = [[-5, -5], [5, 5]]

        register(
            id=name_t + Test + '-v0',
            entry_point='goal_env.mujoco.create_maze_env:create_maze_env',
            kwargs={'env_name': name_t, 'goal_args': goal_args, 'maze_size_scaling': 8, 'random_start': random_start},
            max_episode_steps=max_timestep,
        )

        # v1 is the one we use in the main paper
        register(
            id=name_t + Test + '-v1',
            entry_point='goal_env.mujoco.create_maze_env:create_maze_env',
            kwargs={'env_name': name_t, 'goal_args': goal_args, 'maze_size_scaling': 4, 'random_start': random_start,
                    "fix_goal": fix_goal, "top_down_view": top_down, 'test':Test},
            max_episode_steps=max_timestep,
        )

        register(
            id=name_t + Test + '-v2',
            entry_point='goal_env.mujoco.create_maze_env:create_maze_env',
            kwargs={'env_name': name_t, 'goal_args': goal_args, 'maze_size_scaling': 2, 'random_start': random_start},
            max_episode_steps=max_timestep,
        )
