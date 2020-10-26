import torch
from DQN_SABR.dqn_agent import Agent
import numpy as np
from DQN_SABR.model import QNetwork
import time
from scipy.spatial import distance
import gym
import gym_dqn
from DQN_SABR.dqn_agent import Agent
import random
import torch
from collections import deque
import matplotlib.pyplot as plt
import torch.optim as optim
import numpy as np
from SMPC_uav import *
from SMPC_ugv import *

# SMPC setup for UAV and UGV

# initialize obstacles
obs = {1: {'vertices': [[-3.01, -1,0], [-3.02, 1.03,0], [3,1,0], [3.02, -1.05,0]], 'a': [], 'slopes': [], 'intercepts': [],
               'polygon_type': 4, 'risk': 0.1}}

# initialize prediction horizon and discretized time, and whether to animate
dT = 1
mpc_horizon = 5
animate = True

# initialize SMPC parameters for the UGV
curr_posUGV = np.array([0, -6, 0]).reshape(3,1)
goal_posUGV = np.array([0, 6, 0])
robot_size = 0.5
lb_state = np.array([[-10], [-10], [-2*np.pi]], dtype=float)
ub_state = np.array([[10], [10], [2*np.pi]], dtype=float)
lb_control = np.array([[-1.5], [-np.pi/2]], dtype=float)
ub_control = np.array([[1.5], [np.pi/2]], dtype=float)
Q = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
R_init = np.array([[1, 0, 0], [0, 1, 0] ,[0, 0, 0.001]])
angle_noise_r1 = 0.0
angle_noise_r2 = 0.0
relative_measurement_noise_cov = np.array([[0.0,0], [0,0.0]])
maxComm_distance = -10


SMPC_UGV = SMPC_UGV_Planner(dT, mpc_horizon, curr_posUGV, robot_size, lb_state,
                            ub_state, lb_control, ub_control, Q, R_init, angle_noise_r1, angle_noise_r2,
                            relative_measurement_noise_cov, maxComm_distance, obs, animate)

# initialize SMPC parameters for the UAV
curr_posUAV = np.array([0,0,0,0,-6,0,0,0,4,0]).reshape(10,1)
goal_posUAV = np.array([0,0,0,0,6,0,0,0,4,0])
robot_size = 0.5
vel_limit = 2
lb_state = np.array(
        [[-10], [-vel_limit], [-10**10], [-10**10], [-10], [-vel_limit], [-10**10], [-10**10], [1],
         [-vel_limit]], dtype=float)
ub_state = np.array(
        [[10], [vel_limit], [10**10], [10**10], [10], [vel_limit], [10**10], [10**10], [10], [vel_limit]],
        dtype=float)
lb_control = np.array([[-1], [-1], [-1]], dtype=float)
ub_control = np.array([[1], [1], [1]], dtype=float)

Q = np.array([[1,0,0,0,0,0,0,0,0,0],
                          [0,1,0,0,0,0,0,0,0,0],
                          [0,0,1,0,0,0,0,0,0,0],
                           [0,0,0,1,0,0,0,0,0,0],
                           [0,0,0,0,1,0,0,0,0,0],
                           [0,0,0,0,0,1,0,0,0,0],
                           [0,0,0,0,0,0,1,0,0,0],
                           [0,0,0,0,0,0,0,1,0,0],
                           [0,0,0,0,0,0,0,0,1,0],
                           [0,0,0,0,0,0,0,0,0,1]])

R = np.array([[0.001, 0, 0], [0, 0.001, 0], [0, 0, 0.001]])

# if the UAV and UGV are to be animated in the same screen, multi_agent must be set to true
SMPC_UAV = SMPC_UAV_Planner(dT, 10, curr_posUAV, lb_state,
                            ub_state, lb_control, ub_control, Q, R, robot_size, obs, animate, multi_agent=True)

num_obs_const = 0
for i in range(1, len(obs) + 1):
    num_obs_const += obs[i]['polygon_type']
NUM_OBSTACLES = num_obs_const

max_t = 50
num_models = 11

env = gym.make('dqn-v0')
env.seed(0)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
env.init(curr_posUGV, goal_posUGV, curr_posUAV, goal_posUAV, obs, SMPC_UGV, SMPC_UAV)

index = num_models

index = 0
model = QNetwork(state_size=(NUM_OBSTACLES * 2) + 3+4, action_size=83, seed=0).to(device)
model.load_state_dict(torch.load('dqn_models/checkpoint1.pth'))
state = env.reset()

for t in range(max_t):
    state = torch.from_numpy(state).float().unsqueeze(0).to(device)
    action_values = model(state)
    action = np.argmax(action_values.cpu().data.numpy())
    next_state, _, done, _ = env.step(action)
    state = next_state
    env.render()
    time.sleep(.1)

"""
for i in range(0, num_models):



    model = QNetwork(state_size=(len(OBSTACLE_X)+1)*2, action_size=81, seed=0).to(device)
    model.load_state_dict(torch.load('dqn_models{}checkpoint{}.pth'.format('/', index)))
    state = env.reset()

    for t in range(max_t):
        state = torch.from_numpy(state).float().unsqueeze(0).to(device)
        action_values = model(state)
        action = np.argmax(action_values.cpu().data.numpy())
        next_state, _, done, _ = env.step(action)
        state = next_state
        if (env.xr==env.gxr and env.yr==env.gyr) and (env.xd==env.gxd and env.yd==env.gyd):
            break
        env.render()
        time.sleep(.1)
    index -= 1
"""

