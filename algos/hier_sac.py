import os
import sys

sys.path.append('../')
from datetime import datetime
from tensorboardX import SummaryWriter
from models.networks import *
from algos.replay_buffer import replay_buffer, replay_buffer_energy
from algos.her import her_sampler
# from planner.goal_plan import *
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import time
from algos.sac.sac import SAC
from algos.sac.replay_memory import ReplayMemory, Array_ReplayMemory
import gym
import pickle
# from planner.simhash import HashingBonusEvaluator
from PIL import Image
import imageio
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns
sns.set_color_codes()

SUBGOAL_RANGE = 200.0


class hier_sac_agent:
    def __init__(self, args, env, env_params, test_env, test_env1=None, test_env2=None):
        self.args = args
        self.env = env
        self.test_env = test_env
        self.env_params = env_params
        self.device = args.device
        self.resume = args.resume
        self.resume_epoch = args.resume_epoch
        self.not_train_low = False
        self.test_env1 = test_env1
        self.test_env2 = test_env2
        self.old_sample = args.old_sample

        self.low_dim = env_params['obs']
        self.env_params['low_dim'] = self.low_dim
        self.hi_dim = env_params['obs']
        print("hi_dim", self.hi_dim)

        self.learn_goal_space = True
        self.whole_obs = False  # use whole observation space as subgoal space
        self.abs_range = abs_range = args.abs_range  # absolute goal range
        self.feature_reg = 0.0  # feature l2 regularization
        print("abs_range", abs_range)

        if args.env_name[:5] == "Fetch":
            maze_low = self.env.env.initial_gripper_xpos[:2] - self.env.env.target_range
            maze_high = self.env.env.initial_gripper_xpos[:2] + self.env.env.target_range
            self.hi_act_space = gym.spaces.Box(low=maze_low, high=maze_high)
        else:
            if args.env_name != "NChain-v1":
                self.hi_act_space = self.env.env.maze_space
            else:
                self.hi_act_space = gym.spaces.Box(low=np.array([-1]), high=np.array([1]))
        if self.learn_goal_space:
            if args.env_name == "NChain-v1":
                self.hi_act_space = gym.spaces.Box(low=np.array([-abs_range]), high=np.array([abs_range]))
            else:
                self.hi_act_space = gym.spaces.Box(low=np.array([-abs_range, -abs_range]), high=np.array([abs_range, abs_range]))
        if self.whole_obs:
            vel_low = [-10.] * 4
            vel_high = [10.] * 4
            maze_low = np.concatenate((self.env.env.maze_low, np.array(vel_low)))
            maze_high = np.concatenate((self.env.env.maze_high, np.array(vel_high)))
            self.hi_act_space = gym.spaces.Box(low=maze_low, high=maze_high)


        dense_low = True
        self.low_use_clip = not dense_low  # only sparse reward use clip
        if args.replay_strategy == "future":
            self.low_forward = True
            assert self.low_use_clip is True
        else:
            self.low_forward = False
            assert self.low_use_clip is False
        self.hi_sparse = (self.env.env.reward_type == "sparse")

        # # params of learning phi
        resume_phi = args.resume
        self.not_update_phi = False
        phi_path = args.resume_path

        # resume_phi = True
        # phi_path = 'saved_models/AntMaze1-v1_Jun01_19-26-19'
        # self.not_update_phi = True

        self.save_fig = False
        self.save_model = False
        self.start_update_phi = args.start_update_phi
        self.early_stop = args.early_stop  # after success rate converge, don't update low policy and feature
        if args.env_name in ['AntPush-v1', 'AntFall-v1']:
            if self.not_update_phi:
                self.early_stop_thres = 900
            else:
                self.early_stop_thres = 3500
        elif args.env_name in ["PointMaze1-v1"]:
            self.early_stop_thres = 2000
        elif args.env_name == "AntMaze1-v1":
            self.early_stop_thres = 3000
        else:
            self.early_stop_thres = args.n_epochs
        print("early_stop_threshold", self.early_stop_thres)
        self.success_log = []

        # scaling = self.env.env.env.MAZE_SIZE_SCALING
        # print("scaling", scaling)

        self.count_latent = False
        if self.count_latent:
            self.hash = HashingBonusEvaluator(512, 2)
        self.count_obs = False
        if self.count_obs:
            self.hash = HashingBonusEvaluator(512, env_params['obs'])

        self.high_correct = False
        self.k = args.c
        self.delta_k = 0
        self.prediction_coeff = 0.0
        tanh_output = False
        self.use_prob = False
        print("prediction_coeff", self.prediction_coeff)

        if args.save:
            current_time = datetime.now().strftime('%b%d_%H-%M-%S')
            self.log_dir = 'runs/hier/' + str(args.env_name) + '/RB_Decay_' + current_time + \
                            "_C_" + str(args.c) + "_Image_" + str(args.image) + \
                            "_Seed_" + str(args.seed) + "_Reward_" + str(args.low_reward_coeff) + \
                            "_NoPhi_" + str(self.not_update_phi) + "_LearnG_" + str(self.learn_goal_space) + "_Early_" + str(self.early_stop_thres) + str(args.early_stop)
            self.writer = SummaryWriter(log_dir=self.log_dir)
            if not os.path.exists(self.args.save_dir):
                os.mkdir(self.args.save_dir)
            # path to save the model
            self.model_path = os.path.join(self.args.save_dir, self.args.env_name + "_" + current_time)
            if not os.path.exists(self.model_path):
                os.mkdir(self.model_path)
        # init low-level network
        self.real_goal_dim = self.hi_act_space.shape[0]  # low-level goal space and high-level action space
        self.init_network()
        # init high-level agent
        self.hi_agent = SAC(self.hi_dim + env_params['goal'], self.hi_act_space, args, False, env_params['goal'],
                            args.gradient_flow_value, args.abs_range, tanh_output)
        self.env_params['real_goal_dim'] = self.real_goal_dim
        self.hi_buffer = ReplayMemory(args.buffer_size)

        # her sampler
        self.c = self.args.c  # interval of high level action
        self.low_her_module = her_sampler(args.replay_strategy, args.replay_k, args.distance, args.future_step,
                                          dense_reward=dense_low, direction_reward=False, low_reward_coeff=args.low_reward_coeff)
        if args.env_name[:5] == "Fetch":
            self.low_buffer = replay_buffer_energy(self.env_params, self.args.buffer_size,
                                               self.low_her_module.sample_her_energy, args.env_name)
        else:
            self.low_buffer = replay_buffer(self.env_params, self.args.buffer_size, self.low_her_module.sample_her_transitions)

        not_load_buffer, not_load_high = True, False
        if self.resume is True:
            self.start_epoch = self.resume_epoch
            if not not_load_high:
                self.hi_agent.policy.load_state_dict(torch.load(self.args.resume_path + \
                                                              '/hi_actor_model.pt', map_location='cuda:4')[0])
                # self.hi_agent.critic.load_state_dict(torch.load(self.args.resume_path + \
                #                                                '/hi_critic_model.pt', map_location='cuda:4')[0])

            # print("not load low !!!")
            print("load low !!!")
            self.low_actor_network.load_state_dict(torch.load(self.args.resume_path + \
                                                             '/low_actor_model.pt', map_location='cuda:4')[0])
            self.low_critic_network.load_state_dict(torch.load(self.args.resume_path + \
                                                              '/low_critic_model.pt', map_location='cuda:4')[0])

            if not not_load_buffer:
                # self.hi_buffer = torch.load(self.args.resume_path + '/hi_buffer.pt', map_location='cuda:1')
                self.low_buffer = torch.load(self.args.resume_path + '/low_buffer.pt', map_location='cuda:1')

        # sync target network of low-level
        self.sync_target()

        if hasattr(self.env.env, 'env'):
            self.animate = self.env.env.env.visualize_goal
        else:
            self.animate = self.args.animate
        self.distance_threshold = self.args.distance

        if not (args.gradient_flow or args.use_prediction or args.gradient_flow_value):
            self.representation = RepresentationNetwork(env_params, 3, self.abs_range, self.real_goal_dim).to(args.device)
            if args.use_target:
                self.target_phi = RepresentationNetwork(env_params, 3, self.abs_range, 2).to(args.device)
                # load the weights into the target networks
                self.target_phi.load_state_dict(self.representation.state_dict())
            self.representation_optim = torch.optim.Adam(self.representation.parameters(), lr=0.0001)
            if resume_phi is True:
                print("load phi from: ", phi_path)
                self.representation.load_state_dict(torch.load(phi_path + \
                                                               '/phi_model_4000.pt', map_location='cuda:4')[0])
        elif args.use_prediction:
            self.representation = DynamicsNetwork(env_params, self.abs_range, 2, tanh_output=tanh_output, use_prob=self.use_prob, device=args.device).to(args.device)
            self.representation_optim = torch.optim.Adam(self.representation.parameters(), lr=0.0001)
            if resume_phi is True:
                print("load phi from: ", phi_path)
                self.representation.load_state_dict(torch.load(phi_path + \
                                                               '/phi_model_4000.pt', map_location='cuda:1')[0])



        print("learn goal space", self.learn_goal_space, " update phi", not self.not_update_phi)
        self.train_success = 0
        self.furthest_task = 0.

    def adjust_lr_actor(self, epoch):
        lr_actor = self.args.lr_actor * (0.5 ** (epoch // self.args.lr_decay_actor))
        for param_group in self.low_actor_optim.param_groups:
            param_group['lr'] = lr_actor

    def adjust_lr_critic(self, epoch):
        lr_critic = self.args.lr_critic * (0.5 ** (epoch // self.args.lr_decay_critic))
        for param_group in self.low_critic_optim.param_groups:
            param_group['lr'] = lr_critic

    def learn(self):
        for epoch in range(self.start_epoch, self.args.n_epochs):
            if epoch > 0 and epoch % self.args.lr_decay_actor == 0:
                self.adjust_lr_actor(epoch)
            if epoch > 0 and epoch % self.args.lr_decay_critic == 0:
                self.adjust_lr_critic(epoch)

            ep_obs, ep_ag, ep_g, ep_actions = [], [], [], []
            last_hi_obs = None
            success = 0
            observation = self.env.reset()
            obs = observation['observation']
            ag = observation['achieved_goal'][:self.real_goal_dim]
            g = observation['desired_goal']
            # identify furthest task
            if g[1] >= 8:
                self.furthest_task += 1
                is_furthest_task = True
            else:
                is_furthest_task = False
            if self.learn_goal_space:
                if self.args.gradient_flow:
                    if self.args.use_target:
                        ag = self.hi_agent.policy_target.phi(torch.Tensor(obs).to(self.device)).detach().cpu().numpy()
                    else:
                        ag = self.hi_agent.policy.phi(torch.Tensor(obs).to(self.device)).detach().cpu().numpy()
                elif self.args.gradient_flow_value:
                    ag = self.hi_agent.critic.phi(torch.Tensor(obs).to(self.device)).detach().cpu().numpy()[0]
                elif self.args.use_prediction:
                    ag = self.representation.phi(torch.Tensor(obs).to(self.device)).detach().cpu().numpy()[0]
                else:
                    if self.args.use_target:
                        ag = self.target_phi(torch.Tensor(obs).to(self.device)).detach().cpu().numpy()[0]
                    else:
                        ag = self.representation(torch.Tensor(obs).to(self.device)).detach().cpu().numpy()[0]
            if self.whole_obs:
                ag = obs.copy()

            for t in range(self.env_params['max_timesteps']):
                act_obs, act_g = self._preproc_inputs(obs, g)
                if t % self.c == 0:
                    hi_act_obs = np.concatenate((obs[:self.hi_dim], g))
                    # append high-level rollouts
                    if last_hi_obs is not None:
                        mask = float(not done)
                        if self.high_correct:
                            last_hi_a = ag
                        self.hi_buffer.push(last_hi_obs, last_hi_a, last_hi_r, hi_act_obs, mask, epoch)
                    if epoch < self.args.start_epoch:
                        hi_action = self.hi_act_space.sample()
                        # print("sample", hi_action)
                    else:
                        hi_action = self.hi_agent.select_action(hi_act_obs)
                    last_hi_obs = hi_act_obs.copy()
                    last_hi_a = hi_action.copy()
                    last_hi_r = 0.
                    done = False
                    if self.old_sample:
                        hi_action_for_low = hi_action
                    else:
                        # make hi_action a delta phi(s)
                        hi_action_for_low = ag.copy() + hi_action.copy()
                        hi_action_for_low = np.clip(hi_action_for_low, -SUBGOAL_RANGE, SUBGOAL_RANGE)
                    hi_action_tensor = torch.tensor(hi_action_for_low, dtype=torch.float32).unsqueeze(0).to(self.device)
                    # update high-level policy
                    if len(self.hi_buffer) > self.args.batch_size:
                        self.update_hi(epoch)
                with torch.no_grad():
                    if self.not_train_low:
                        action = self.test_policy(act_obs[:, :self.low_dim], hi_action_tensor)
                    else:
                        action = self.explore_policy(act_obs[:, :self.low_dim], hi_action_tensor)
                # feed the actions into the environment
                observation_new, r, _, info = self.env.step(action)
                if info['is_success']:
                    done = True
                    # only record the first success
                    if success == 0 and is_furthest_task:
                        success = t
                        self.train_success += 1
                if self.animate:
                    self.env.render()
                obs_new = observation_new['observation']
                ag_new = observation_new['achieved_goal'][:self.real_goal_dim]
                if self.learn_goal_space:
                    if self.args.gradient_flow:
                        if self.args.use_target:
                            ag_new = self.hi_agent.policy_target.phi(
                                torch.Tensor(obs_new).to(self.device)).detach().cpu().numpy()
                        else:
                            ag_new = self.hi_agent.policy.phi(torch.Tensor(obs_new).to(self.device)).detach().cpu().numpy()
                    elif self.args.gradient_flow_value:
                        ag_new = self.hi_agent.critic.phi(torch.Tensor(obs_new).to(self.device)).detach().cpu().numpy()[0]
                    elif self.args.use_prediction:
                        ag_new = self.representation.phi(torch.Tensor(obs_new).to(self.device)).detach().cpu().numpy()[0]
                    else:
                        if self.args.use_target:
                            ag_new = self.target_phi(torch.Tensor(obs_new).to(self.device)).detach().cpu().numpy()[0]
                        else:
                            ag_new = self.representation(torch.Tensor(obs_new).to(self.device)).detach().cpu().numpy()[0]
                if self.whole_obs:
                    ag_new = obs_new.copy()
                if done is False:
                    if self.count_latent:
                        self.hash.inc_hash(ag[None])
                        r += self.hash.predict(ag_new[None])[0] * 0.1
                    if self.count_obs:
                        self.hash.inc_hash(obs[None])
                        r += self.hash.predict(obs_new[None])[0] * 0.1
                    last_hi_r += r
                # append rollouts
                ep_obs.append(obs[:self.low_dim].copy())
                ep_ag.append(ag.copy())
                ep_g.append(hi_action_for_low.copy())
                ep_actions.append(action.copy())
                # re-assign the observation
                obs = obs_new
                ag = ag_new
                # slowly update phi
                if epoch > self.start_update_phi and not self.not_update_phi and not self.args.gradient_flow and not self.args.gradient_flow_value:
                    self.slow_update_phi(epoch)
                    if t % self.args.period == 0 and self.args.use_target:
                        self._soft_update_target_network(self.target_phi, self.representation)
            ep_obs.append(obs[:self.low_dim].copy())
            ep_ag.append(ag.copy())
            mask = float(not done)
            hi_act_obs = np.concatenate((obs[:self.hi_dim], g))
            self.hi_buffer.push(last_hi_obs, last_hi_a, last_hi_r, hi_act_obs, mask, epoch)

            mb_obs = np.array([ep_obs])
            mb_ag = np.array([ep_ag])
            mb_g = np.array([ep_g])
            mb_actions = np.array([ep_actions])
            self.low_buffer.store_episode([mb_obs, mb_ag, mb_g, mb_actions, success, False])

            if self.args.save and self.args.env_name == "NChain-v1":
                self.writer.add_scalar('Explore/coverage_' + self.args.env_name, self.env.env.coverage, epoch)
            # print("coverage", self.env.env.coverage)

            # update low-level
            if not self.not_train_low:
                for n_batch in range(self.args.n_batches):
                    self._update_network(epoch, self.low_buffer, self.low_actor_target_network,
                                         self.low_critic_target_network,
                                         self.low_actor_network, self.low_critic_network, 'max_timesteps',
                                         self.low_actor_optim, self.low_critic_optim, use_forward_loss=self.low_forward, clip=self.low_use_clip)
                    if n_batch % self.args.period == 0:
                        self._soft_update_target_network(self.low_actor_target_network, self.low_actor_network)
                        self._soft_update_target_network(self.low_critic_target_network, self.low_critic_network)


            # start to do the evaluation
            if epoch % self.args.eval_interval == 0 and epoch != 0:
                if self.test_env1 is not None:
                    eval_success1, _ = self._eval_hier_agent(env=self.test_env1)
                    eval_success2, _ = self._eval_hier_agent(env=self.test_env2)
                farthest_success_rate, _ = self._eval_hier_agent(env=self.test_env)
                random_success_rate, _ = self._eval_hier_agent(env=self.env)
                self.success_log.append(farthest_success_rate)
                mean_success = np.mean(self.success_log[-5:])
                # stop updating phi and low
                if self.early_stop and (mean_success >= 0.9 or epoch > self.early_stop_thres):
                    print("early stop !!!")
                    self.not_update_phi = True
                    self.not_train_low = True
                print('[{}] epoch is: {}, eval hier success rate is: {:.3f}'.format(datetime.now(), epoch, random_success_rate))
                if self.save_fig:
                    self.vis_hier_policy(epoch=epoch)
                    self.visualize_representation(epoch=epoch)
                if self.args.save:
                    print("log_dir: ", self.log_dir)
                    torch.save([self.hi_agent.critic.state_dict()], self.model_path + '/hi_critic_model.pt')
                    torch.save([self.low_critic_network.state_dict()], self.model_path + '/low_critic_model.pt')
                    torch.save(self.hi_buffer, self.model_path + '/hi_buffer.pt')
                    torch.save(self.low_buffer, self.model_path + '/low_buffer.pt')
                    if not self.args.gradient_flow and not self.args.gradient_flow_value:
                        if self.save_model:
                            # self.cal_MIV(epoch)
                            torch.save([self.representation.state_dict()], self.model_path + '/phi_model_{}.pt'.format(epoch))
                            torch.save([self.hi_agent.policy.state_dict()], self.model_path + '/hi_actor_{}.pt'.format(epoch))
                            torch.save([self.low_actor_network.state_dict()], self.model_path + '/low_actor_{}.pt'.format(epoch))
                        else:
                            torch.save([self.representation.state_dict()], self.model_path + '/phi_model.pt')
                            torch.save([self.hi_agent.policy.state_dict()], self.model_path + '/hi_actor_model.pt')
                            torch.save([self.low_actor_network.state_dict()], self.model_path + '/low_actor_model.pt')
                    self.writer.add_scalar('Success_rate/hier_farthest_' + self.args.env_name, farthest_success_rate, epoch)
                    self.writer.add_scalar('Success_rate/hier_random_' + self.args.env_name, random_success_rate, epoch)
                    self.writer.add_scalar('Explore/furthest_task_' + self.args.env_name, self.furthest_task, epoch)
                    if self.test_env1 is not None:
                        self.writer.add_scalar('Success_rate/eval1_' + self.args.env_name,
                                               eval_success1, epoch)
                        self.writer.add_scalar('Success_rate/eval2_' + self.args.env_name, eval_success2,
                                               epoch)

    # pre_process the inputs
    def _preproc_inputs(self, obs, g):
        obs = torch.tensor(obs, dtype=torch.float32).unsqueeze(0).to(self.device)
        g = torch.tensor(g, dtype=torch.float32).unsqueeze(0).to(self.device)
        return obs, g

    # this function will choose action for the agent and do the exploration
    def _select_actions(self, pi):
        action = pi.cpu().numpy().squeeze()
        if action.shape == ():
            action = np.array([action])
        # add the gaussian
        action += self.args.noise_eps * self.env_params['action_max'] * np.random.randn(*action.shape)
        action = np.clip(action, -self.env_params['action_max'], self.env_params['action_max'])
        # random actions...
        if np.random.rand() < self.args.random_eps:
            action = np.random.uniform(low=-self.env_params['action_max'], high=self.env_params['action_max'], \
                                       size=self.env_params['action'])
        return action

    def explore_policy(self, obs, goal):
        pi = self.low_actor_network(obs, goal)
        action = self._select_actions(pi)
        return action

    def update_hi(self, epoch):
        if self.args.gradient_flow or self.args.gradient_flow_value:
            sample_data, _ = self.slow_collect()
            sample_data = torch.tensor(sample_data, dtype=torch.float32).to(self.device)
        else:
            sample_data = None
        critic_1_loss, critic_2_loss, policy_loss, _, _ = self.hi_agent.update_parameters(self.hi_buffer,
                                                                                          self.args.batch_size,
                                                                                          self.env_params,
                                                                                          self.hi_sparse,
                                                                                          sample_data)
        if self.args.save:
            self.writer.add_scalar('Loss/hi_critic_1', critic_1_loss, epoch)
            self.writer.add_scalar('Loss/hi_critic_2', critic_2_loss, epoch)
            self.writer.add_scalar('Loss/hi_policy', policy_loss, epoch)

    def random_policy(self, obs, goal):
        random_actions = np.random.uniform(low=-self.env_params['action_max'], high=self.env_params['action_max'], \
                                           size=self.env_params['action'])
        return random_actions

    def test_policy(self, obs, goal):
        pi = self.low_actor_network(obs, goal)
        # convert the actions
        actions = pi.detach().cpu().numpy().squeeze()
        if actions.shape == ():
            actions = np.array([actions])
        return actions

    # soft update
    def _soft_update_target_network(self, target, source):
        for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_((1 - self.args.polyak) * param.data + self.args.polyak * target_param.data)

    # update the network
    def _update_network(self, epoch, buffer, actor_target, critic_target, actor, critic, T, actor_optim, critic_optim, use_forward_loss=True, clip=True):
        # sample the episodes
        transitions = buffer.sample(self.args.batch_size)
        # pre-process the observation and goal
        o, o_next, g, ag = transitions['obs'], transitions['obs_next'], transitions['g'], transitions['ag']
        transitions['obs'], transitions['g'] = o, g
        transitions['obs_next'], transitions['g_next'] = o_next, g
        ag_next = transitions['ag_next']

        # start to do the update
        obs_cur = transitions['obs']
        g_cur = transitions['g']
        obs_next = transitions['obs_next']
        g_next = transitions['g_next']

        # done
        dist = np.linalg.norm(ag_next - g_next, axis=1)
        not_done = (dist > self.distance_threshold).astype(np.int32).reshape(-1, 1)

        # transfer them into the tensor
        obs_cur = torch.tensor(obs_cur, dtype=torch.float32).to(self.device)
        g_cur = torch.tensor(g_cur, dtype=torch.float32).to(self.device)
        obs_next = torch.tensor(obs_next, dtype=torch.float32).to(self.device)
        g_next = torch.tensor(g_next, dtype=torch.float32).to(self.device)
        ag_next = torch.tensor(ag_next, dtype=torch.float32).to(self.device)
        not_done = torch.tensor(not_done, dtype=torch.int32).to(self.device)

        actions_tensor = torch.tensor(transitions['actions'], dtype=torch.float32).to(self.device)
        r_tensor = torch.tensor(transitions['r'], dtype=torch.float32).to(self.device)

        # calculate the target Q value function
        with torch.no_grad():
            actions_next = actor_target(obs_next, g_next)
            q_next_value = critic_target(obs_next, g_next, actions_next)
            q_next_value = q_next_value.detach()
            target_q_value = r_tensor + critic_target.gamma * q_next_value * not_done
            target_q_value = target_q_value.detach()
            if clip:
                clip_return = self.env_params[T]
                target_q_value = torch.clamp(target_q_value, -clip_return, 0.)
        # the q loss
        real_q_value = critic(obs_cur, g_cur, actions_tensor)
        critic_loss = (target_q_value - real_q_value).pow(2).mean()
        if use_forward_loss:
            forward_loss = critic(obs_cur, ag_next, actions_tensor).pow(2).mean()
            critic_loss += forward_loss
        # the actor loss
        actions_real = actor(obs_cur, g_cur)
        actor_loss = -critic(obs_cur, g_cur, actions_real).mean()
        actor_loss += self.args.action_l2 * (actions_real / self.env_params['action_max']).pow(2).mean()

        # start to update the network
        actor_optim.zero_grad()
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.low_actor_network.parameters(), 1.0)
        actor_optim.step()
        # update the critic_network
        critic_optim.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.low_critic_network.parameters(), 1.0)
        critic_optim.step()

        if self.args.save:
            if T == 'max_timesteps':
                name = 'low'
            else:
                name = 'high'
            self.writer.add_scalar('Loss/' + name + '_actor_loss' + self.args.metric, actor_loss, epoch)
            self.writer.add_scalar('Loss/' + name + '_critic_loss' + self.args.metric, critic_loss, epoch)

    def _eval_hier_agent(self, env, n_test_rollouts=10):
        total_success_rate = []
        if not self.args.eval:
            n_test_rollouts = self.args.n_test_rollouts
        discount_reward = np.zeros(n_test_rollouts)
        for roll in range(n_test_rollouts):
            per_success_rate = []
            observation = env.reset()
            obs = observation['observation']
            g = observation['desired_goal']
            for num in range(self.env_params['max_test_timesteps']):
                with torch.no_grad():
                    act_obs, act_g = self._preproc_inputs(obs, g)
                    if num % self.c == 0:
                        hi_act_obs = np.concatenate((obs[:self.hi_dim], g))
                        hi_action = self.hi_agent.select_action(hi_act_obs, evaluate=True)
                        if self.old_sample:
                            new_hi_action = hi_action
                        else:
                            ag = self.representation(torch.Tensor(obs).to(self.device)).detach().cpu().numpy()[0]
                            new_hi_action = ag + hi_action
                            new_hi_action = np.clip(new_hi_action, -SUBGOAL_RANGE, SUBGOAL_RANGE)
                        hi_action_tensor = torch.tensor(new_hi_action, dtype=torch.float32).unsqueeze(0).to(self.device)
                    action = self.test_policy(act_obs[:, :self.low_dim], hi_action_tensor)
                observation_new, rew, done, info = env.step(action)
                if self.animate:
                    env.render()
                obs = observation_new['observation']
                g = observation_new['desired_goal']
                if done:
                    per_success_rate.append(info['is_success'])
                    if bool(info['is_success']):
                        # print("t:", num)
                        discount_reward[roll] = 1 - 1. / self.env_params['max_test_timesteps'] * num
                    break
            total_success_rate.append(per_success_rate)
        total_success_rate = np.array(total_success_rate)
        global_success_rate = np.mean(total_success_rate[:, -1])
        global_reward = np.mean(discount_reward)
        if self.args.eval:
            print("hier success rate", global_success_rate, global_reward)
        return global_success_rate, global_reward

    def init_network(self):
        self.low_actor_network = actor(self.env_params, self.real_goal_dim).to(self.device)
        self.low_actor_target_network = actor(self.env_params, self.real_goal_dim).to(self.device)
        self.low_critic_network = criticWrapper(self.env_params, self.args, self.real_goal_dim).to(self.device)
        self.low_critic_target_network = criticWrapper(self.env_params, self.args, self.real_goal_dim).to(self.device)

        self.start_epoch = 0

        # create the optimizer
        self.low_actor_optim = torch.optim.Adam(self.low_actor_network.parameters(), lr=self.args.lr_actor)
        self.low_critic_optim = torch.optim.Adam(self.low_critic_network.parameters(), lr=self.args.lr_critic, weight_decay=1e-5)

    def sync_target(self):
        # load the weights into the target networks
        self.low_actor_target_network.load_state_dict(self.low_actor_network.state_dict())
        self.low_critic_target_network.load_state_dict(self.low_critic_network.state_dict())

    def slow_update_phi(self, epoch):
        sample_data, hi_action = self.slow_collect()
        sample_data = torch.tensor(sample_data, dtype=torch.float32).to(self.device)
        if not self.args.use_prediction:
            obs, obs_next = self.representation(sample_data[0]), self.representation(sample_data[1])
            min_dist = torch.clamp((obs - obs_next).pow(2).mean(dim=1), min=0.)
            hi_obs, hi_obs_next = self.representation(sample_data[2]), self.representation(sample_data[3])
            max_dist = torch.clamp(1 - (hi_obs - hi_obs_next).pow(2).mean(dim=1), min=0.)
            representation_loss = (min_dist + max_dist).mean()
            # add l2 regularization
            representation_loss += self.feature_reg * (obs / self.abs_range).pow(2).mean()
        else:
            hi_action = torch.tensor(hi_action, dtype=torch.float32).to(self.device)
            with torch.no_grad():
                target_next_obs = self.representation.phi(sample_data[3])
            obs, obs_next = self.representation.phi(sample_data[0]), self.representation.phi(sample_data[1])
            min_dist = torch.clamp((obs - obs_next).pow(2).mean(dim=1), min=0.)
            hi_obs, hi_obs_next = self.representation.phi(sample_data[2]), self.representation.phi(sample_data[3])
            max_dist = torch.clamp(1 - (hi_obs - hi_obs_next).pow(2).mean(dim=1), min=0.)
            representation_loss = (min_dist + max_dist).mean()
            # prediction loss
            if self.use_prob:
                predict_distribution = self.representation(sample_data[2], hi_action)
                prediction_loss = - predict_distribution.log_prob(target_next_obs).mean()
            else:
                predict_state = self.representation(sample_data[2], hi_action)
                prediction_loss = (predict_state - target_next_obs).pow(2).mean()
            representation_loss += self.prediction_coeff * prediction_loss
        self.representation_optim.zero_grad()
        representation_loss.backward()
        self.representation_optim.step()
        if self.args.save:
            self.writer.add_scalar('Loss/phi_loss' + self.args.metric, representation_loss, epoch)

    def slow_collect(self, batch_size=100):
        if self.args.use_prediction:
            transitions = self.low_buffer.sample(batch_size)
            obs, obs_next = transitions['obs'], transitions['obs_next']

            hi_obs, hi_action, _, hi_obs_next, _ = self.hi_buffer.sample(batch_size)
            hi_obs, hi_obs_next = hi_obs[:, :self.env_params['obs']], hi_obs_next[:, :self.env_params['obs']]
            train_data = np.array([obs, obs_next, hi_obs, hi_obs_next])
            return train_data, hi_action
        else:
            # new negative samples
            episode_num = self.low_buffer.current_size
            obs_array = self.low_buffer.buffers['obs'][:episode_num]
            episode_idxs = np.random.randint(0, episode_num, batch_size)
            t_samples = np.random.randint(self.env_params['max_timesteps'] - self.k - self.delta_k, size=batch_size)
            if self.delta_k > 0:
                delta = np.random.randint(self.delta_k, size=batch_size)
            else:
                delta = 0

            hi_obs = obs_array[episode_idxs, t_samples]
            hi_obs_next = obs_array[episode_idxs, t_samples + self.k + delta]
            obs = hi_obs
            obs_next = obs_array[episode_idxs, t_samples + 1 + delta]

            train_data = np.array([obs, obs_next, hi_obs, hi_obs_next])
            return train_data, None

    def visualize_representation(self, epoch):
        transitions = self.low_buffer.sample(800)
        obs = transitions['obs']
        # with open('fig/final/' + "sampled_states.pkl", 'wb') as output:
        #     pickle.dump(obs, output)

        index1 = np.where((obs[:, 0] < 4) & (obs[:, 1] < 4))
        index2 = np.where((obs[:, 0] < 4) & (obs[:, 1] > 4))
        index3 = np.where((obs[:, 0] > 4) & (obs[:, 1] < 4))
        index4 = np.where((obs[:, 0] > 4) & (obs[:, 1] > 4))
        index_lst = [index1, index2, index3, index4]

        obs_tensor = torch.Tensor(obs).to(self.device)
        features = self.representation(obs_tensor).detach().cpu().numpy()
        plt.scatter(features[:, 0], features[:, 1], color='green')
        plt.show()

        # rep = []
        # for index in index_lst:
        #     rep.append(features[index])
        #
        # self.plot_fig(rep, 'slow_feature', epoch)
        #
        #
        # obs_list = []
        # for index in index_lst:
        #     obs_list.append(obs[index])
        # self.plot_fig(obs_list, 'obs', epoch)

        '''
        tsne_list = []
        res_tsne = TSNE(n_components=2).fit_transform(obs)
        for index in index_lst:
            tsne_list.append(res_tsne[index])
        self.plot_fig(tsne_list, 'tsne_feature', epoch)
        '''

    def plot_fig(self, rep, name, epoch):
        fig = plt.figure()
        axes = fig.add_subplot(111)
        rep1, rep2, rep3, rep4 = rep
        def scatter_rep(rep1, c, marker):
            if rep1.shape[0] > 0:
                l1 = axes.scatter(rep1[:, 0], rep1[:, 1], c=c, marker=marker)
            else:
                l1 = axes.scatter([], [], c=c, marker=marker)
            return l1
        l1 = scatter_rep(rep1, c='y', marker='s')
        l2 = scatter_rep(rep2, c='r', marker='o')
        l3 = scatter_rep(rep3, c='b', marker='1')
        l4 = scatter_rep(rep4, c='g', marker='2')

        plt.xlabel('x')
        plt.ylabel('y')
        axes.legend((l1, l2, l3, l4), ('space1', 'space2', 'space3', 'space4'))
        plt.savefig('fig/final/' + name + str(epoch) + '.png')
        plt.close()

    def vis_hier_policy(self, epoch=0, load_obs=None, color_map='RdYlBu'):
        obs_vec = []
        hi_action_vec = []
        env = self.test_env
        observation = env.reset()
        obs = observation['observation']
        obs_vec.append(obs)
        g = observation['desired_goal']
        if load_obs is None:
            for num in range(self.env_params['max_test_timesteps']):
                with torch.no_grad():
                    act_obs, act_g = self._preproc_inputs(obs, g)
                    if num % self.c == 0:
                        hi_act_obs = np.concatenate((obs[:self.hi_dim], g))
                        hi_action = self.hi_agent.select_action(hi_act_obs, evaluate=True)
                        hi_action_tensor = torch.tensor(hi_action, dtype=torch.float32).unsqueeze(0).to(self.device)
                        ag = self.representation(torch.Tensor(obs).to(self.device)).detach().cpu().numpy()[0]
                        distance = np.linalg.norm(hi_action - ag)
                        print("distance", distance)
                        hi_action_vec.append(hi_action)
                    action = self.test_policy(act_obs[:, :self.low_dim], hi_action_tensor)
                observation_new, rew, done, info = env.step(action)
                if self.animate:
                    env.render()
                obs = observation_new['observation']
                obs_vec.append(obs)
                if done:
                    if info['is_success']:
                        print("success !!!")
                    break
        else:
            obs_vec = load_obs[0]

        plt.figure(figsize=(12, 6))
        obs_vec = np.array(obs_vec)
        with open('fig/final/' + "img_push_hard.pkl", 'wb') as output:
            pickle.dump(obs_vec, output)
        self.plot_rollout(obs_vec, "XY_{}".format(epoch * self.env_params['max_timesteps']), 121, goal=g)

        if not self.learn_goal_space:
            features = obs_vec[:, :2]
            feature_goal = g[:2]
        else:
            obs_tensor = torch.Tensor(obs_vec[:, :self.hi_dim]).to(self.device)
            features = self.representation(obs_tensor).detach().cpu().numpy()
            # rest = (self.env_params['obs'] - self.env_params['goal']) * [0.]
            # g = np.concatenate((g, np.array(rest)))
            # g = torch.tensor(g, dtype=torch.float32).unsqueeze(0).to(self.device)
            # feature_goal = self.representation(g).detach().cpu().numpy()[0]
            feature_goal = None
        hi_action_vec = np.array(hi_action_vec)
        self.plot_rollout(features, "Feature_{}".format(epoch * self.env_params['max_timesteps']), 122, feature_goal, color_map="Blues",
                          hi_action_vec = hi_action_vec)
        if load_obs is not None and len(load_obs) > 1:
            obs_vec = load_obs[1]
            obs_tensor = torch.Tensor(obs_vec[:, :self.hi_dim]).to(self.device)
            features = self.representation(obs_tensor).detach().cpu().numpy()
            self.plot_rollout(features, "Feature_{}".format(epoch * self.env_params['max_timesteps']), 122, feature_goal,
                              color_map="Wistia")

        file_name = 'fig/rebuttal/rollout' + str(epoch) + '.png'
        plt.savefig(file_name, bbox_inches='tight', transparent=True)
        # plt.show()
        plt.close()

    def plot_rollout(self, obs_vec, name, num, goal=None, hi_action_vec=None, no_axis=True, color_map='RdYlBu'):
        plt.subplot(num)
        cm = plt.cm.get_cmap(color_map)
        num = np.arange(obs_vec.shape[0])
        plt.scatter(obs_vec[:, 0], obs_vec[:, 1], c=num, cmap=cm)
        if goal is not None:
            plt.scatter([goal[0]], [goal[1]], marker='*',
                        color='green', s=200, label='goal')
        if hi_action_vec is not None:
            plt.scatter(hi_action_vec[:, 0], hi_action_vec[:, 1], c="k")
        plt.title(name, fontsize=24)
        if no_axis:
            plt.axis('off')
        if not no_axis:
            plt.scatter([obs_vec[0, 0]], [obs_vec[0, 1]], marker='+',
                        color='green', s=200, label='start')
            plt.scatter([obs_vec[-1, 0]], [obs_vec[-1, 1]], marker='+',
                        color='red', s=200, label='end')
            plt.legend(loc=2, bbox_to_anchor=(1.05, 1.0), fontsize=14, borderaxespad=0.)
        # plt.show()





















