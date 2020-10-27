import gym_dqnprev
import gym
from dqn_agent import Agent
from model import QNetwork
import torch
import numpy as np
import matplotlib.pyplot as plt
import time

max_t = 500
num_models = 11
MAP_SIZE=20
# Optimality:
robot_start = [16, 9]
robot_goal = [2, 9]
drone_start = [16, 10]
drone_goal = [2, 10]

OBSTACLE_X = [9, 9, 9 ,9 ,9 ,9, 10, 10, 10, 10, 10, 10]
OBSTACLE_Y = [7, 8, 9, 10, 11, 12, 7, 8, 9, 10, 11, 12]

# Safety
OBSTACLE_X = [8,9,10,11,12,13,12,13,6]
OBSTACLE_Y = [3,3,3,3,3,3,2,2,14]

env = gym.make('dqnprev-v0')
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
env.init(robot_start[0], robot_start[1], robot_goal[0], robot_goal[1], drone_start[0], drone_start[1], drone_goal[0], drone_goal[1], MAP_SIZE, OBSTACLE_X, OBSTACLE_Y, False)
env.seed(0)
index = num_models

index = 0
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

