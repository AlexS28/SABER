import gym
import gym_dqn
from dqn_agent import Agent
import random
import torch
import numpy as np
from collections import deque
import matplotlib.pyplot as plt
import torch.optim as optim
import pandas as pd
from SMPC_uav import *
from SMPC_ugv import *

# save datasets that track controller failure
if not os.path.isdir("dqn_models"):
    os.makedirs("dqn_models")

if not os.path.isdir("data_collection"):
    os.makedirs("data_collection")

# SMPC setup for UAV and UGV
# initialize obstacles
obs = {1: {'vertices': [[-1.01, -1,0], [-3.02, 1.03,0], [3,1,0], [3.02, -1.05,0]], 'a': [], 'slopes': [], 'intercepts': [],
               'polygon_type': 4, 'risk': 0.1}}
obs.update({2: {'vertices': [[4, 3,0], [5, 5,0], [6, 3.2,0]], 'a': [], 'slopes': [], 'intercepts': [], 'polygon_type': 3,
             'risk': 0.4}})
obs.update(
        {3: {'vertices': [[-4, 4.1,0]], 'size': 0.7, 'polygon_type': 1, 'risk': 0.4}})
obs.update(
        {4: {'vertices': [[0, 4.1,0]], 'size': 0.7, 'polygon_type': 1, 'risk': 0.4}})


# initialize prediction horizon and discretized time, and whether to animate
dT = 1
mpc_horizon = 5
animate = True

# initialize SMPC parameters for the UGV
curr_posUGV = np.array([0, -6, 0]).reshape(3,1)
goal_posUGV = np.array([0, 7, 0])
robot_size = 0.5
lb_state = np.array([[-10], [-10], [-2*np.pi]], dtype=float)
ub_state = np.array([[10], [10], [2*np.pi]], dtype=float)
lb_control = np.array([[-1.5], [-np.pi/2]], dtype=float)
ub_control = np.array([[1.5], [np.pi/2]], dtype=float)
Q = np.array([[1, 0, 0], [0, 2, 0], [0, 0, 0.05]])
R_init = np.array([[0.5, 0, 0], [0, 0.5, 0] ,[0, 0, 0.5]])
angle_noise_r1 = 0.0
angle_noise_r2 = 0.0
relative_measurement_noise_cov = np.array([[0.0,0], [0,0.0]])
maxComm_distance = -10

SMPC_UGV = SMPC_UGV_Planner(dT, mpc_horizon, curr_posUGV, robot_size, lb_state,
                            ub_state, lb_control, ub_control, Q, R_init, angle_noise_r1, angle_noise_r2,
                            relative_measurement_noise_cov, maxComm_distance, obs, animate)

# initialize SMPC parameters for the UAV
curr_posUAV = np.array([0,0,0,0,-6,0,0,0,4,0]).reshape(10,1)
goal_posUAV = np.array([0,0,0,0,7,0,0,0,4,0])
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

env = gym.make('dqn-v0')
env.init(curr_posUGV, goal_posUGV, curr_posUAV, goal_posUAV, obs, SMPC_UGV, SMPC_UAV)
env.seed(0)
agent = Agent(state_size=(NUM_OBSTACLES * 2) + 3 + 4, action_size=83, seed=0)

# max_t = 200, eps_.999, eps_end 0.1 (For random obstacles, eps_decay = 0.99995 seems good), otherwise use 0.9995
def dqn(n_episodes=25000, max_t=100, eps_start=1, eps_end=0.05, eps_decay=0.99995):

    """Deep Q-Learning.
    Params
    ======
        n_episodes (int): maximum number of training episodes
        max_t (int): maximum number of timesteps per episode
        eps_start (float): starting value of epsilon, for epsilon-greedy action selection
        eps_end (float): minimum value of epsilon
        eps_decay (float): multiplicative factor (per episode) for decreasing epsilon
    """

    #LR = agent.LR
    scores = []  # list containing scores from each episode
    scores_graph = []
    # specify the number of models that will be saved, per training run
    number_models = 100
    scores_window = []  # last 100 scores
    eps = eps_start  # initialize epsilon
    index_model = 0

    #model_dict = {0: agent.qnetwork_local.state_dict()}
    #for i in range(1, number_scores):
        #model_dict.update({i: agent.qnetwork_local.state_dict()})

    for i_episode in range(1, n_episodes + 1):
        state = env.reset()
        score = 0
        for t in range(max_t):
            action = agent.act(state, eps)
            next_state, reward, done, _ = env.step(action)
            agent.step(state, action, reward, next_state, done)
            state = next_state
            score += reward
            env.episode_steps = t
            env.render()
            if done:
                break

        scores_window.append(score)  # save most recent score
        scores.append(score)  # save most recent score
        eps = max(eps_end, eps_decay * eps)  # decrease epsilon

        print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_window)), end="")
        if i_episode % 10 == 0:
            print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_window)))
            print(eps)
            scores_graph.append(np.mean(scores_window))
            max_avg = np.mean(scores_window)

            if max_avg == np.max(scores_graph):
                if index_model == number_models:
                    index_model = 0
                print("Model saved! maximum average reward was: {:.2f}".format(np.mean(scores_window)))

                #index_min = min(range(len(scores_window_model)), key=scores_window_model.__getitem__)
                # saving all models from best run in a folder
                torch.save(agent.qnetwork_local.state_dict(), 'dqn_models/checkpoint{}.pth'.format(index_model))
                scores_window = []
                #agent.save('dqn_models/model{}.h5'.format(index_model))
                index_model += 1

    torch.save(agent.qnetwork_local.state_dict(), 'checkpoint_last.pth')
    return scores_graph


scores_graph = dqn()
dataset_r = np.delete(env.dataset_r, 0, 1)
dataset_d = np.delete(env.dataset_d, 0, 1)
np.savetxt("data_collection/dataset_r.csv", dataset_r, delimiter=",")
np.savetxt("data_collection/dataset_d.csv", dataset_d, delimiter=",")

# plot the scores
fig = plt.figure()
ax = fig.add_subplot(111)
plt.plot(np.arange(len(scores_graph)), scores_graph)
plt.plot(pd.Series(scores_graph).rolling(100).mean())

plt.title('Deep Q-Learning - Average Rewards During Training')
plt.ylabel('Average Reward')
plt.xlabel('Per 100 Episodes')
plt.show()
plt.savefig("dqn_results", dpi=None, facecolor='w', edgecolor='w',
        orientation='portrait', papertype=None, format=None,
        transparent=False, bbox_inches=None, pad_inches=0.1,
        frameon=None, metadata=None)

