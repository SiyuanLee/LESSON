import argparse

"""
Here are the param for the training

"""




def get_args_ant():
    parser = argparse.ArgumentParser()
    # the environment setting
    parser.add_argument('--env-name', type=str, default='AntMaze1-v1', help='the environment name')
    parser.add_argument('--test', type=str, default='AntMaze1Test-v1')
    parser.add_argument('--n-epochs', type=int, default=20000, help='the number of epochs to train the agent')
    parser.add_argument('--n-batches', type=int, default=200, help='the times to update the network')
    parser.add_argument('--seed', type=int, default=125, help='random seed')

    parser.add_argument('--replay-strategy', type=str, default='none', help='the HER strategy')
    parser.add_argument('--save-dir', type=str, default='saved_models/', help='the path to save the models')

    parser.add_argument('--noise-eps', type=float, default=0.2, help='noise factor for Gaussian')
    parser.add_argument('--random-eps', type=float, default=0.2, help="prob for acting randomly")

    parser.add_argument('--buffer-size', type=int, default=int(1e6), help='the size of the buffer')
    parser.add_argument('--replay-k', type=int, default=5, help='ratio to be replaced')
    parser.add_argument('--future-step', type=int, default=200, help='future step to be sampled')
    parser.add_argument('--batch-size', type=int, default=128, help='the sample batch size')
    parser.add_argument('--gamma', type=float, default=0.99, help='the discount factor')
    parser.add_argument('--action-l2', type=float, default=0.0, help='l2 reg')
    parser.add_argument('--lr-actor', type=float, default=0.0002, help='the learning rate of the actor')
    parser.add_argument('--lr-critic', type=float, default=0.0002, help='the learning rate of the critic')
    parser.add_argument('--polyak', type=float, default=0.99, help='the average coefficient')
    parser.add_argument('--n-test-rollouts', type=int, default=10, help='the number of tests')

    parser.add_argument('--metric', type=str, default='MLP', help='the metric for the distance embedding')
    parser.add_argument('--device', type=str, default="cuda:3", help='cuda device')

    parser.add_argument('--lr-decay-actor', type=int, default=3000, help='actor learning rate decay')
    parser.add_argument('--lr-decay-critic', type=int, default=3000, help='critic learning rate decay')
    parser.add_argument('--layer', type=int, default=6, help='number of layers for critic')

    parser.add_argument('--period', type=int, default=3, help='target update period')
    parser.add_argument('--distance', type=float, default=0.1, help='distance threshold for HER')

    parser.add_argument('--resume', type=bool, default=False, help='resume or not')
    # Will be considered only if resume is True
    parser.add_argument('--resume-epoch', type=int, default=0, help='resume epoch')
    parser.add_argument('--resume-path', type=str, default='saved_models/AntPush-v1_Nov16_08-30-42', help='resume path')

    # add for hier policy
    parser.add_argument('--save', type=bool, default=True, help='save model and tensorboard data')
    parser.add_argument('--animate', type=bool, default=False)
    parser.add_argument("--eval", type=bool, default=False)
    parser.add_argument('--eval_interval', type=int, default=50, help="every n episodes to eval once")
    parser.add_argument('--c', type=int, default=50, help="interval of high-level action")
    parser.add_argument('--gradient_flow', type=bool, default=False, help='end-to-end learn feature and policy')
    parser.add_argument('--gradient_flow_value', type=bool, default=False, help='slow feature as a embedding of value function')
    parser.add_argument('--abs_range', type=float, default=20.0, help='range of high-level action space')
    parser.add_argument('--use_target', type=bool, default=False, help='use target network for learning feature')
    parser.add_argument('--early_stop', type=bool, default=False, help='early stop the learning of low-level')
    parser.add_argument('--low_reward_coeff', type=float, default=0.1, help='low-level reward coeff')
    parser.add_argument("--use_prediction", type=bool, default=False, help='use prediction error to learn feature')
    parser.add_argument("--start_update_phi", type=int, default=10, help='use prediction error to learn feature')
    parser.add_argument("--image", type=bool, default=False, help='use image input')
    parser.add_argument("--old_sample", type=bool, default=False, help='sample the absolute goal in the abs_range')

    # args of sac
    parser.add_argument('--policy', default="Gaussian",
                        help='Policy Type: Gaussian | Deterministic (default: Gaussian)')
    parser.add_argument('--tau', type=float, default=0.005, metavar='G',
                        help='target smoothing coefficient(τ) (default: 0.005)')
    parser.add_argument('--lr', type=float, default=0.0003, metavar='G',
                        help='learning rate (default: 0.0003)')
    parser.add_argument('--alpha', type=float, default=0.2, metavar='G',
                        help='Temperature parameter α determines the relative importance of the entropy\
                                    term against the reward (default: 0.2)')
    parser.add_argument('--automatic_entropy_tuning', type=bool, default=False, metavar='G',
                        help='Automaically adjust α (default: False)')
    parser.add_argument('--hidden_size', type=int, default=256, metavar='N',
                        help='hidden size (default: 256)')
    parser.add_argument('--start_epoch', type=int, default=300, metavar='N',
                        help='Epochs sampling random actions (default: 50)')
    parser.add_argument('--target_update_interval', type=int, default=1, metavar='N',
                        help='Value target update per no. of updates per step (default: 1)')

    args = parser.parse_args()
    return args




def get_args_chain():
    parser = argparse.ArgumentParser()
    # the environment setting
    parser.add_argument('--env-name', type=str, default='NChain-v1', help='the environment name')
    parser.add_argument('--test', type=str, default='NChain-v1')
    parser.add_argument('--n-epochs', type=int, default=100, help='the number of epochs to train the agent')
    parser.add_argument('--n-batches', type=int, default=200, help='the times to update the network')
    parser.add_argument('--seed', type=int, default=160, help='random seed')

    parser.add_argument('--replay-strategy', type=str, default='none', help='the HER strategy')
    parser.add_argument('--save-dir', type=str, default='saved_models/', help='the path to save the models')

    parser.add_argument('--noise-eps', type=float, default=0.2, help='noise factor for Gaussian')
    parser.add_argument('--random-eps', type=float, default=0.2, help="prob for acting randomly")

    parser.add_argument('--buffer-size', type=int, default=int(1e6), help='the size of the buffer')
    parser.add_argument('--replay-k', type=int, default=5, help='ratio to be replaced')
    parser.add_argument('--future-step', type=int, default=200, help='future step to be sampled')
    parser.add_argument('--batch-size', type=int, default=128, help='the sample batch size')
    parser.add_argument('--gamma', type=float, default=0.99, help='the discount factor')
    parser.add_argument('--action-l2', type=float, default=0.0, help='l2 reg')
    parser.add_argument('--lr-actor', type=float, default=0.0002, help='the learning rate of the actor')
    parser.add_argument('--lr-critic', type=float, default=0.0002, help='the learning rate of the critic')
    parser.add_argument('--polyak', type=float, default=0.99, help='the average coefficient')
    parser.add_argument('--n-test-rollouts', type=int, default=10, help='the number of tests')

    parser.add_argument('--metric', type=str, default='MLP', help='the metric for the distance embedding')
    parser.add_argument('--device', type=str, default="cuda:8", help='cuda device')

    parser.add_argument('--lr-decay-actor', type=int, default=3000, help='actor learning rate decay')
    parser.add_argument('--lr-decay-critic', type=int, default=3000, help='critic learning rate decay')
    parser.add_argument('--layer', type=int, default=6, help='number of layers for critic')

    parser.add_argument('--period', type=int, default=3, help='target update period')
    parser.add_argument('--distance', type=float, default=0.1, help='distance threshold for HER')

    parser.add_argument('--resume', type=bool, default=False, help='resume or not')
    # Will be considered only if resume is True
    parser.add_argument('--resume-epoch', type=int, default=0, help='resume epoch')
    parser.add_argument('--resume-path', type=str, default='saved_models/NChain-v1_Jul29_11-02-57', help='resume path')

    # add for hier policy
    parser.add_argument('--save', type=bool, default=True, help='save model and tensorboard data')
    parser.add_argument('--animate', type=bool, default=False)
    parser.add_argument("--eval", type=bool, default=False)
    parser.add_argument('--eval_interval', type=int, default=50, help="every n episodes to eval once")
    parser.add_argument('--c', type=int, default=30, help="interval of high-level action")
    parser.add_argument('--gradient_flow', type=bool, default=False, help='end-to-end learn feature and policy')
    parser.add_argument('--gradient_flow_value', type=bool, default=False, help='slow feature as a embedding of value function')
    parser.add_argument('--abs_range', type=float, default=100.0, help='range of high-level action space')
    parser.add_argument('--use_target', type=bool, default=False, help='use target network for learning feature')
    parser.add_argument('--early_stop', type=bool, default=True, help='early stop the learning of low-level')
    parser.add_argument('--low_reward_coeff', type=float, default=0.01, help='low-level reward coeff')
    parser.add_argument("--use_prediction", type=bool, default=False, help='use prediction error to learn feature')
    parser.add_argument("--start_update_phi", type=int, default=2, help='use prediction error to learn feature')
    parser.add_argument("--image", type=bool, default=False, help='use image input')

    # args of sac (high-level learning)
    parser.add_argument('--policy', default="Gaussian",
                        help='Policy Type: Gaussian | Deterministic (default: Gaussian)')
    parser.add_argument('--tau', type=float, default=0.005, metavar='G',
                        help='target smoothing coefficient(τ) (default: 0.005)')
    parser.add_argument('--lr', type=float, default=0.0003, metavar='G',
                        help='learning rate (default: 0.0003)')
    parser.add_argument('--alpha', type=float, default=0.2, metavar='G',
                        help='Temperature parameter α determines the relative importance of the entropy\
                                    term against the reward (default: 0.2)')
    parser.add_argument('--automatic_entropy_tuning', type=bool, default=False, metavar='G',
                        help='Automaically adjust α (default: False)')
    parser.add_argument('--hidden_size', type=int, default=256, metavar='N',
                        help='hidden size (default: 256)')
    parser.add_argument('--start_epoch', type=int, default=20000, metavar='N',
                        help='Epochs sampling random actions (default: 50)')
    parser.add_argument('--target_update_interval', type=int, default=1, metavar='N',
                        help='Value target update per no. of updates per step (default: 1)')

    args = parser.parse_args()
    return args
