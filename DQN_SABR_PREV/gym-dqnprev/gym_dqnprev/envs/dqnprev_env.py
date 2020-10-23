import gym
from gym import error, spaces, utils
from gym.utils import seeding
import numpy as np
from scipy.spatial import distance
import cv2
from PIL import Image
from sklearn.utils.extmath import cartesian

class dqnprevEnv(gym.Env):
  metadata = {'render.modes': ['human']}

  # reward parameters
  #MOVE_REWARD = 1
  ROBOT_VISION_DISTANCE = 15
  OBSTACLE_COLLISION_PENALTY = 100
  GOAL_REWARD = 100
  def __init__(self):
    self.colors = {"robot": (255, 175, 0),
              "goal": (0, 255, 0),
              "unseen_obstacle": (0, 0, 75),
              "seen_obstacle": (0, 0, 255),
              "drone": (255, 0, 255)}

  def init(self, xr, yr, gxr, gyr, xd, yd, gxd, gyd, map_size, obstacles_x, obstacles_y, randomize):
    self.xr_start = xr
    self.yr_start = yr
    self.xr = xr
    self.yr = yr
    self.gxr = gxr
    self.gyr = gyr
    self.xd_start = xd
    self.yd_start = yd
    self.xd = xd
    self.yd = yd
    self.gxd = gxd
    self.gyd = gyd
    self.OBSTACLE_X = obstacles_x
    self.OBSTACLE_Y = obstacles_y
    self.NUM_OBSTACLES = len(self.OBSTACLE_X)
    self.next_state = np.zeros(((self.NUM_OBSTACLES + 1)*2,))
    self.next_state[0] = round(distance.euclidean((self.xr, self.yr), (self.gxr, self.gyr)),2)
    self.next_state[1] = round(distance.euclidean((self.xd, self.yd), (self.gxd, self.gyd)),2)
    self.max_r = self.next_state[0]
    self.max_d = self.next_state[1]
    self.size = map_size
    self.randomize = randomize
    # each robot has 9 possible actions, two robots simultaneously --> 9x9 = 81 actions
    self.actions = cartesian(([-1,0,1],[-1,0,1],[-1,0,1],[-1,0,1]))
    # termination state for ground robot r and drone robot d
    self.done = False
    self.done_r = False
    self.done_d = False
    # reward for ground robot r and drone robot d
    self.reward_step = 0
    self.reward = 0
    self.step_number = 0

  class Blob:
    def __init__(self, x, y):
      self.x = x
      self.y = y

  def step(self, action):
      # termination state for ground robot r and drone robot d
      self.done_r = False
      self.done_d = False
      self.done = False
      self.reward = 0

      # choose action
      chosen_action = self.actions[action]

      # move both robots
      self.move(xr = chosen_action[0], yr = chosen_action[1], xd = chosen_action[2], yd = chosen_action[3])

      # check for new seen obstacles within (robot vision range) - This only serves for visualization purposes
      for obstacle in self.unseen_obstacles:
          ugv_to_obstacle = round(distance.euclidean((self.xr, self.yr), (obstacle.x, obstacle.y)),1)
          uav_to_obstacle = round(distance.euclidean((self.xd, self.yd), (obstacle.x, obstacle.y)),1)
          if self.ROBOT_VISION_DISTANCE >= ugv_to_obstacle or self.ROBOT_VISION_DISTANCE >= uav_to_obstacle:
              self.seen_obstacles.append(obstacle)
              self.unseen_obstacles.remove(obstacle)

      for i in range(0, len(self.obstacles)):
            self.next_state[i + 2] = round(
                distance.euclidean((self.xr, self.yr), (self.obstacles[i].x, self.obstacles[i].y)),2)
            self.next_state[i + 2 + self.NUM_OBSTACLES] = round(
                distance.euclidean((self.xd, self.yd), (self.obstacles[i].x, self.obstacles[i].y)),2)

      self.next_state[0] = round(distance.euclidean((self.xr, self.yr), (self.gxr, self.gyr)),2)
      self.next_state[1] = round(distance.euclidean((self.xd, self.yd), (self.gxd, self.gyd)),2)

      # punish either robot or drone if colliding into obstacle
      for obstacle in self.obstacles:
        if self.xr == obstacle.x and self.yr == obstacle.y:
            self.reward -= self.OBSTACLE_COLLISION_PENALTY
            self.done = True
        if self.xd == obstacle.x and self.yd == obstacle.y:
            self.reward -= self.OBSTACLE_COLLISION_PENALTY
            self.done = True

      if not self.done and ((self.xr == self.gxr and self.yr == self.gyr) or (self.xd == self.gxd and self.yd == self.gyd)):
            self.reward += self.GOAL_REWARD
      if not self.done and ((self.xr == self.gxr and self.yr == self.gyr) and (self.xd == self.gxd and self.yd == self.gyd)):
            self.reward += 4*self.GOAL_REWARD

      self.reward_step -= 1
      if self.step_number > 50:
          self.reward += self.reward_step

      return self.next_state, self.reward, self.done, {}

  def move(self, xr, yr, xd, yd):

      self.xr += xr
      self.yr += yr
      self.xd += xd
      self.yd += yd

      # If we are out of bounds, fix!
      if self.xr < 0:
        self.xr = 0
      elif self.xr > self.size-1:
        self.xr = self.size-1
      if self.yr < 0:
        self.yr = 0
      elif self.yr > self.size-1:
        self.yr = self.size-1

      if self.xd < 0:
        self.xd = 0
      elif self.xd > self.size-1:
        self.xd = self.size-1
      if self.yd < 0:
        self.yd = 0
      elif self.yd > self.size-1:
        self.yd = self.size-1

  def reset(self):
      self.xr, self.yr = self.xr_start, self.yr_start
      self.xd, self.yd = self.xd_start, self.yd_start
      self.obstacles = []
      self.unseen_obstacles = []
      for i in range(self.NUM_OBSTACLES):
          if self.randomize:
            x = np.random.randint(2, self.size-2)
            y = np.random.randint(2, self.size-2)
            new_obstacle = self.Blob(x, y)
            self.obstacles.append(new_obstacle)
            self.unseen_obstacles.append(new_obstacle)
          else:
            new_obstacle = self.Blob(self.OBSTACLE_X[i], self.OBSTACLE_Y[i])
            self.obstacles.append(new_obstacle)
            self.unseen_obstacles.append(new_obstacle)
      self.seen_obstacles = []
      self.episode_step = 0
      self.next_state = np.zeros(((self.NUM_OBSTACLES+1)*2, ))
      self.next_state[0] = round(distance.euclidean((self.xr, self.yr), (self.gxr, self.gyr)),2)
      self.next_state[1] = round(distance.euclidean((self.xd, self.yd), (self.gxd, self.gyd)),2)
      # reward for ground robot r and drone robot d
      self.reward = 0
      self.step_number = 0
      self.reward_step = 0
      return self.next_state

  def render(self, mode='human', close=False):
    img = self.get_image()
    img = img.resize((300, 300))  # resizing so we can see our agent in all its glory.
    cv2.imshow("image", np.array(img))  # show it!
    cv2.waitKey(1)

  def get_image(self):
    env = np.zeros((self.size, self.size, 3), dtype=np.uint8)  # starts an rbg of our size
    env[self.xr][self.yr] = self.colors["robot"]  # sets the robot tile to blue
    env[self.gxr][self.gyr] = self.colors["goal"]  # sets the goal location tile to green color
    env[self.xd][self.yd] = self.colors["drone"]  # sets the robot tile to blue
    env[self.gxd][self.gyd] = self.colors["goal"]
    # for obstacle in self.unseen_obstacles:
    # env[obstacle.x][obstacle.y] = self.colors["unseen_obstacle"]  # sets the obstacle locations to dark red
    for obstacle in self.seen_obstacles:
      env[obstacle.x][obstacle.y] = self.colors["seen_obstacle"]  # sets the seen obstacle locations to bright red

    img = Image.fromarray(env, 'RGB')  # reading to rgb. Apparently. Even tho color definitions are bgr. ???
    return img