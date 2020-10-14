import torch
from DQN_SABR.dqn_agent import Agent
import numpy as np
from DQN_SABR.model import QNetwork
import gym
import gym_dqn
import time
from scipy.spatial import distance

max_t = 100
num_models = 11
MAP_SIZE=10
OBSTACLE_X = [5, 7, 3, 4, 6, 3, 5, 4, 2, 5, 4, 3]
OBSTACLE_Y = [5, 7, 4, 3, 8, 5, 4, 5, 3, 3, 4, 3]
OBSTACLE_X = [5, 7, 4]
OBSTACLE_Y = [5, 7, 4]

robot_start = [0, 0]
robot_goal = [9, 9]
drone_start = [0, 0]
drone_goal = [9, 9]

env = gym.make('dqn-v0')
env.seed(0)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
env.init(robot_start[0], robot_start[1], robot_goal[0], robot_goal[1], drone_start[0], drone_start[1], drone_goal[0],
         drone_goal[1], MAP_SIZE, OBSTACLE_X, OBSTACLE_Y, True)
index = num_models

index = 19
model = QNetwork(state_size=(len(OBSTACLE_X)+1)*2, action_size=81, seed=0).to(device)
model.load_state_dict(torch.load('dqn_models{}checkpoint{}.pth'.format('/', index)))
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

