import gym
import gym_dqn
from DQN_SABR.dqn_agent import Agent
import random
import torch
import numpy as np
from collections import deque
import matplotlib.pyplot as plt

env = gym.make('dqn-v0')
MAP_SIZE=10
robot_start = [0, 0]
robot_goal = [9, 9]
drone_start = [9, 0]
drone_goal = [0, 9]
OBSTACLE_X = [5, 7, 3, 4, 6, 3, 5, 4, 2, 5, 4, 3]
OBSTACLE_Y = [5, 7, 4, 3, 8, 5, 4, 5, 3, 3, 4, 3]

env.init(robot_start[0], robot_start[1], robot_goal[0], robot_goal[1], drone_start[0], drone_start[1], drone_goal[0], drone_goal[1], MAP_SIZE, OBSTACLE_X, OBSTACLE_Y)
env.seed(0)
agent = Agent(state_size=(len(OBSTACLE_X)+1)*2+1, action_size=81, seed=0)




def dqn(n_episodes=5000, max_t=100, eps_start=1, eps_end=0.01, eps_decay=0.995):

    """Deep Q-Learning.
    Params
    ======
        n_episodes (int): maximum number of training episodes
        max_t (int): maximum number of timesteps per episode
        eps_start (float): starting value of epsilon, for epsilon-greedy action selection
        eps_end (float): minimum value of epsilon
        eps_decay (float): multiplicative factor (per episode) for decreasing epsilon
    """

    scores = []  # list containing scores from each episode
    scores_graph = []
    scores_window = deque(maxlen=100)  # last 100 scores
    max_avg = 0
    eps = eps_start  # initialize epsilon
    for i_episode in range(1, n_episodes + 1):
        state = env.reset()
        score = 0
        for t in range(max_t):
            action = agent.act(state, eps)
            next_state, reward, done = env.step(action)
            agent.step(state, action, reward, next_state, done)
            state = next_state
            score += reward
            env.render()
            if done:
                break
        scores_window.append(score)  # save most recent score
        scores.append(score)  # save most recent score
        eps = max(eps_end, eps_decay * eps)  # decrease epsilon
        print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_window)), end="")
        if i_episode % 100 == 0:
            print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_window)))
            scores_graph.append(np.mean(scores_window))
            max_avg = np.mean(scores_window)


        if np.mean(scores_window) >= 1.9:
            print('\nEnvironment solved in {:d} episodes!\tAverage Score: {:.2f}'.format(i_episode - 100,
                                                                        np.mean(scores_window)))
            torch.save(agent.qnetwork_local.state_dict(), 'checkpoint.pth')
            #break

    #torch.save(agent.qnetwork_local.state_dict(), 'checkpoint.pth')
    return scores_graph

scores_graph = dqn()


# plot the scores
fig = plt.figure()
ax = fig.add_subplot(111)
plt.plot(np.arange(len(scores_graph)), scores_graph)
plt.ylabel('Average Reward')
plt.xlabel('Per 100 Episodes')
plt.show()