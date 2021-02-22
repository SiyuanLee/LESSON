import numpy as np
from gym import utils
from gym.envs.mujoco import mujoco_env

class SwimmerEnv(mujoco_env.MujocoEnv, utils.EzPickle):
    ORI_IND = 2
    FILE = "swimmer.xml"
    def __init__(self, file_path=None, expose_all_qpos=True):
        self._expose_all_qpos = expose_all_qpos
        self.add_noise = False

        mujoco_env.MujocoEnv.__init__(self, file_path, 4)
        utils.EzPickle.__init__(self)

    def _step(self, a):
        return self.step(a)

    def step(self, a):
        ctrl_cost_coeff = 0.0001
        xposbefore = self.sim.data.qpos[0]
        self.do_simulation(a, self.frame_skip)
        xposafter = self.sim.data.qpos[0]
        reward_fwd = (xposafter - xposbefore) / self.dt
        reward_ctrl = - ctrl_cost_coeff * np.square(a).sum()
        reward = reward_fwd + reward_ctrl
        ob = self._get_obs()
        return ob, reward, False, dict(reward_fwd=reward_fwd, reward_ctrl=reward_ctrl)

    def _get_obs(self):
        qpos = self.sim.data.qpos
        qvel = self.sim.data.qvel
        # print("qpos", qpos)
        # print("qvel", qvel)
        return np.concatenate([qpos.flat, qvel.flat])

    def reset_model(self):
        self.set_state(
            self.init_qpos + self.np_random.uniform(low=-.1, high=.1, size=self.model.nq),
            self.init_qvel + self.np_random.uniform(low=-.1, high=.1, size=self.model.nv)
        )
        return self._get_obs()

    def get_ori(self):
        return self.data.qpos[self.__class__.ORI_IND]

    def set_xy(self, xy):
        qpos = np.copy(self.data.qpos)
        qpos[0] = xy[0]
        qpos[1] = xy[1]

        qvel = self.data.qvel
        self.set_state(qpos, qvel)

    def get_xy(self):
        qpos = np.copy(self.data.qpos)
        return qpos[:2]

    def viewer_setup(self):
        # self.viewer.cam.trackbodyid = 1
        # self.viewer.cam.distance = self.model.stat.extent * 0.7
        # self.viewer.cam.lookat[2] = 0.8925
        # self.viewer.cam.elevation = 0

        self.viewer.cam.trackbodyid = -1
        self.viewer.cam.distance = 60
        self.viewer.cam.elevation = -90
