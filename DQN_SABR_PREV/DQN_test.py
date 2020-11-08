import gym_dqnprev
import gym
from dqn_agent import Agent
from model import QNetwork
import torch
import numpy as np
import matplotlib.pyplot as plt
import time

import gym
from model import QNetwork
import torch
import numpy as np
import matplotlib.pyplot as plt
import random

max_t = 50
num_models_ugv = 6
num_models_uav = 16
# MAP_SIZE=10
# robot_start = [8, 5]
# robot_goal = [0, 5]
# drone_start = [8, 5]
# drone_goal = [0, 6]
# OBSTACLE_X = [4, 4, 4, 4, 5, 5, 5,5]
# OBSTACLE_Y = [3, 4, 5, 6, 3, 4, 5,6]

MAP_SIZE = 20
# Optimality:
# robot_start = [16, 10]
# robot_goal = [2, 10]
# OBSTACLE_X = [9,9, 9, 9 ,9 ,9 ,9, 9,10, 10,10, 10, 10, 10, 10,10]
# OBSTACLE_Y = [6,7, 8, 9, 10, 11, 12,13, 6,7, 8, 9, 10, 11, 12,13]

# Complex
robot_start = [16, 9]
robot_goal = [2, 13]
# drone_start = [16, 10]
# drone_goal = [2, 6]
OBSTACLE_X = [6, 9, 9, 9, 9, 12, 9, 12, 5, 12, 12, 12, 12, 7, 7, 7, 8, 8, 8]
OBSTACLE_Y = [5, 2, 3, 4, 5, 6, 8, 6, 17, 11, 12, 13, 14, 12, 13, 14, 12, 13, 14]

env = gym.make('dqnprev-v0')
#device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#env.init(robot_start[0], robot_start[1], robot_goal[0], robot_goal[1], MAP_SIZE, OBSTACLE_X, OBSTACLE_Y, False)
env = gym.make('dqnprev-v0')
MAP_SIZE=10
robot_start = [0, 0]
robot_goal = [9, 0]
drone_start = [0, 1]
drone_goal = [9, 9]
#OBSTACLE_X = [5, 7, 3, 4, 6, 3, 5, 4, 2, 5, 4, 3]
#OBSTACLE_Y = [5, 7, 4, 3, 8, 5, 4, 5, 3, 3, 4, 3]

OBSTACLE_X = [5, 7, 4, 7, 2, 5]
OBSTACLE_Y = [5, 4, 4, 6, 3, 6]

env.init(robot_start[0], robot_start[1], robot_goal[0], robot_goal[1], drone_start[0], drone_start[1], drone_goal[0], drone_goal[1], MAP_SIZE, OBSTACLE_X, OBSTACLE_Y, False)
env.seed(0)

for i in range(0, len(env.OBSTACLE_X)):
    plt.plot(env.OBSTACLE_X[i], env.OBSTACLE_Y[i], marker="s", color="red", markersize=22)

index = 0
eps = 0
for i in range(3, 5):

    model = QNetwork(state_size=(len(OBSTACLE_X)+1)*2, action_size=81, seed=0)

    model.load_state_dict(torch.load('dqn_models{}checkpoint{}.pth'.format('/', index)))
    state = env.reset()
    env.render()
    time.sleep(5)
    for t in range(max_t):

        state = torch.from_numpy(state).float().unsqueeze(0)
        action_values = model(state)
        # Epsilon-greedy action selection
        if random.random() > eps:
            action = np.argmax(action_values.cpu().data.numpy())
        else:
            action = random.choice(np.arange(model.action_size))

        next_state, _, done, _ = env.step(action)
        state = next_state
        if done:
            break

        env.render()
        time.sleep(0.5)
        plt.plot([env.xd], [env.yd], marker="o", color="purple")
        plt.plot([env.xr], [env.yr], marker="o", color="orange")
        plt.plot([env.gxr], [env.gyr], marker="o", color="green")
        plt.plot([env.gxd], [env.gyd], marker="o", color="green")
        plt.grid(color='dimgrey', linestyle='-', linewidth=2)

    index =i

"""
index = 0
for i in range(10, num_models_uav):

    model = QNetwork(state_size=(len(OBSTACLE_X) + 5), action_size=9, seed=0).to(device)
    model.load_state_dict(torch.load('dqn_models_uav{}checkpoint{}.pth'.format('/', index)))
    state = env.reset()

    for t in range(max_t):

        state = torch.from_numpy(state).float().unsqueeze(0).to(device)
        action_values = model(state)
        action = np.argmax(action_values.cpu().data.numpy())
        next_state, _, done, _ = env.step(action)
        state = next_state
        if ((next_state[0] < 2)):
            break
        env.render()

        plt.plot([env.xr], [env.yr], marker="o", color="purple")
        plt.grid(color='dimgrey', linestyle='-', linewidth=2)
    index += 1
"""
plt.show()