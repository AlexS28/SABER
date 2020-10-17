import gym
from gym import error, spaces, utils
from gym.utils import seeding
import numpy as np
from scipy.spatial import distance
import cv2
from PIL import Image
from sklearn.utils.extmath import cartesian
import math as m
import matplotlib.pyplot as plt

class dqnEnv(gym.Env):
  metadata = {'render.modes': ['human']}

  # reward parameters

  def init(self, curr_posUGV, goal_posUGV, curr_posUAV, goal_posUAV, obstacles, SMPC_UGV, SMPC_UAV):
    self.curr_posUGV = curr_posUGV
    self.curr_posUAV = curr_posUAV
    self.curr_posUGV_start = curr_posUGV
    self.curr_posUAV_start = curr_posUAV
    self.xr = curr_posUGV[0]
    self.yr = curr_posUGV[1]
    self.gxr = goal_posUGV[0]
    self.gyr = goal_posUGV[1]
    self.xd = curr_posUAV[0]
    self.yd = curr_posUAV[4]
    self.zd = curr_posUAV[8]
    self.gxd = goal_posUAV[0]
    self.gyd = goal_posUAV[4]
    self.obs = obstacles
    self.SMPC_UGV = SMPC_UGV
    self.SMPC_UAV = SMPC_UAV
    self.SMPC_UGV.opti.set_value(self.SMPC_UGV.r1_goal, curr_posUGV)
    self.SMPC_UAV.opti.set_value(self.SMPC_UAV.r_goal, curr_posUAV)
    num_obs_const = 0
    for i in range(1, len(self.obs) + 1):
        num_obs_const += self.obs[i]['polygon_type']
    self.NUM_OBSTACLES = num_obs_const
    self.next_state = np.zeros(((self.NUM_OBSTACLES*2)+3,))
    self.next_state[0] = np.round(distance.euclidean((self.xr, self.yr), (self.gxr, self.gyr)), 2)
    self.next_state[1] = np.round(distance.euclidean((self.xd, self.yd), (self.gxd, self.gyd)), 2)
    self.next_state[2] = np.round(distance.euclidean((self.xd, self.yd, self.zd), (self.xr, self.yr, 0)), 2)
    self.max_r = self.next_state[0]
    self.max_d = self.next_state[1]
    # each robot has 9 possible actions, two robots simultaneously --> 9x9 = 81 actions, control up/down for drone = 83 actions
    self.actions = cartesian(([-1,0,1],[-1,0,1],[-1,0,1],[-1,0,1]))
    self.actions = np.concatenate((self.actions, [[0,0,2,2]]), axis=0)
    self.actions = np.concatenate((self.actions, [[0,0,-2,-2]]), axis=0)
    # termination state for ground robot r and drone robot d
    self.done = False
    # reward for ground robot r and drone robot d
    self.reward_step = 0
    self.reward = 0
    self.step_number = 0
    self.max_steps = 5
    self.dist_factor = 5
    self.solver_failure = 0
    self.ROBOT_VISION_DISTANCE = 15
    self.OBSTACLE_COLLISION_PENALTY = 100
    self.GOAL_REWARD = 100
    self.SOLVER_FAIL_PENALTY = 100
    self.COMMUNICATION_RANGE_PENALTY = 100
    self.episode_steps = 0

  def step(self, action):
      # termination state for ground robot r and drone robot d
      self.done = False
      self.reward = 0
      self.step_number = 0

      # choose action
      chosen_action = self.actions[action]

      # set local target positions for both robots
      self.target(xr = chosen_action[0], yr = chosen_action[1], xd = chosen_action[2], yd = chosen_action[3])

      # run SMPC for both robots for max_steps, then collect rewards
      while self.step_number < self.max_steps:
          try:
            solUGV = self.SMPC_UGV.opti.solve()
            x = solUGV.value(self.SMPC_UGV.X)[:, 1]
            self.curr_posUGV = np.array(x).reshape(3, 1)
            self.SMPC_UGV.check_obstacles(np.concatenate((self.curr_posUGV[0], self.curr_posUGV[1], [0])))
          except:
            u = np.zeros((2,))
            x = self.SMPC_UGV.next_state_nominal(self.curr_posUGV, u)
            self.curr_posUGV = np.array(x).reshape(3, 1)
            self.reward -= self.SOLVER_FAIL_PENALTY
            self.solver_failure += 1

          self.SMPC_UGV.opti.set_value(self.SMPC_UGV.r1_pos, x)

          try:
              solUAV = self.SMPC_UAV.opti.solve()
              x = solUAV.value(self.SMPC_UAV.X)[:, 1]
              self.curr_posUAV = np.array(x).reshape(10, 1)
              self.SMPC_UAV.check_obstacles(np.concatenate((self.curr_posUAV[0], self.curr_posUAV[4], self.curr_posUAV[8])))
          except:
              u = np.zeros((3,))
              x = self.SMPC_UAV.next_state_nominal(self.curr_posUAV, u)
              self.curr_posUAV = np.array(x).reshape(10, 1)
              self.reward -= self.SOLVER_FAIL_PENALTY
              self.solver_failure += 1

          self.SMPC_UAV.opti.set_value(self.SMPC_UAV.r_pos, x)
          self.step_number += 1

      # update current position to be used by the DQN
      self.xr = self.curr_posUGV[0][0]
      self.yr = self.curr_posUGV[1][0]
      self.xd = self.curr_posUAV[0][0]
      self.yd = self.curr_posUAV[4][0]
      self.zd = self.curr_posUAV[8][0]

      # reward if either one or both of the robots are at the goal
      if not self.done and (m.sqrt((self.xr - self.gxr)**2 + (self.yr-self.gyr)**2) < 0.5 or m.sqrt((
            self.xd-self.gxd)**2 + (self.yd-self.gyd)**2) < 0.5):
            self.reward += self.GOAL_REWARD
      if not self.done and (m.sqrt((self.xr - self.gxr)**2 + (self.yr - self.gyr)**2) < 0.5 or m.sqrt((
            self.xd - self.gxd)**2 + (self.yd - self.gyd)**2) < 0.5):
            self.reward += self.GOAL_REWARD*4
            self.done = True

      # punish if the UAV and UAV are too far apart
      if np.round(distance.euclidean((self.xd, self.yd, self.zd), (self.xr, self.yr, 0)), 2) < 2 or np.round(
              distance.euclidean((self.xd, self.yd, self.zd), (self.xr, self.yr, 0)), 2) > 6:
          self.reward -= self.COMMUNICATION_RANGE_PENALTY
          #self.done = True

      # punish if there are too many solver failures
      if self.solver_failure > 10:
          self.done = True
          self.reward -= 1000

      # update the DQN states
      self.next_state[0] = np.round(distance.euclidean((self.xr, self.yr), (self.gxr, self.gyr)), 2)
      self.next_state[1] = np.round(distance.euclidean((self.xd, self.yd), (self.gxd, self.gyd)), 2)
      self.next_state[2] = np.round(distance.euclidean((self.xd, self.yd, self.zd), (self.xr, self.yr, 0)), 2)
      self.next_state[3:3+self.NUM_OBSTACLES] = self.SMPC_UGV.dqn_states
      self.next_state[3+self.NUM_OBSTACLES:] = self.SMPC_UAV.dqn_states

      self.reward_step -= 1
      if self.episode_steps > 40:
          self.reward += self.reward_step

      return self.next_state, self.reward, self.done, {}

  def target(self, xr, yr, xd, yd):

        if np.abs(xd) != 2:
            self.xr_target = (self.xr + self.dist_factor*xr)
            self.yr_target = (self.yr + self.dist_factor*yr)
            self.xd_target = (self.xd + self.dist_factor*xd)
            self.yd_target = (self.yd + self.dist_factor*yd)
            goal_posUGV = np.array([self.xr_target, self.yr_target, 0])
            goal_posUAV = np.array([self.xd_target, 0, 0, 0, self.yd_target, 0, 0, 0, self.zd, 0])
            self.SMPC_UGV.opti.set_value(self.SMPC_UGV.r1_goal, goal_posUGV)
            self.SMPC_UAV.opti.set_value(self.SMPC_UAV.r_goal, goal_posUAV)

        else:
            self.zd = self.zd*xd/2
            goal_posUAV = np.array([self.xd_target, 0, 0, 0, self.yd_target, 0, 0, 0, self.zd, 0])
            self.SMPC_UAV.opti.set_value(self.SMPC_UAV.r_goal, goal_posUAV)


  def reset(self):

      self.curr_posUGV = self.curr_posUGV_start
      self.curr_posUAV = self.curr_posUAV_start
      self.SMPC_UGV.opti.set_value(self.SMPC_UGV.r1_pos, self.curr_posUGV)
      self.SMPC_UAV.opti.set_value(self.SMPC_UAV.r_pos, self.curr_posUAV)
      self.xr = self.curr_posUGV[0][0]
      self.yr = self.curr_posUGV[1][0]
      self.xd = self.curr_posUAV[0][0]
      self.yd = self.curr_posUAV[4][0]
      self.zd = self.curr_posUAV[8][0]
      self.SMPC_UGV.opti.set_value(self.SMPC_UGV.r1_goal, self.curr_posUGV)
      self.SMPC_UAV.opti.set_value(self.SMPC_UAV.r_goal, self.curr_posUAV)
      self.done = False
      # reward for ground robot r and drone robot d
      self.reward_step = 0
      self.reward = 0
      self.step_number = 0
      self.solver_failure = 0
      self.next_state = np.zeros(((self.NUM_OBSTACLES * 2) + 3,))
      self.next_state[0] = np.round(distance.euclidean((self.xr, self.yr), (self.gxr, self.gyr)), 2)
      self.next_state[1] = np.round(distance.euclidean((self.xd, self.yd), (self.gxd, self.gyd)), 2)
      self.next_state[2] = np.round(distance.euclidean((self.xd, self.yd, self.zd), (self.xr, self.yr, 0)), 2)

      return self.next_state

  def render(self, mode='human', close=False):
    self.SMPC_UGV.animate(self.curr_posUGV)
    self.SMPC_UAV.animate_multi_agents(self.SMPC_UGV.ax, self.curr_posUAV)
    plt.show()
    plt.pause(0.001)