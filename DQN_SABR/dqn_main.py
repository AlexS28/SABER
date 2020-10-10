import gym
import gym_dqn
from DQN_SABR.dqn_agent import Agent
import random
import torch
import numpy as np
from collections import deque
import matplotlib.pyplot as plt
import torch.optim as optim

env = gym.make('dqn-v0')
MAP_SIZE=10
robot_start = [0, 0]
robot_goal = [0, 9]
drone_start = [0, 0]
drone_goal = [9, 9]
OBSTACLE_X = [5, 7, 3, 4, 6, 3, 5, 4, 2, 5, 4, 3]
OBSTACLE_Y = [5, 7, 4, 3, 8, 5, 4, 5, 3, 3, 4, 3]

env.init(robot_start[0], robot_start[1], robot_goal[0], robot_goal[1], drone_start[0], drone_start[1], drone_goal[0], drone_goal[1], MAP_SIZE, OBSTACLE_X, OBSTACLE_Y)
env.seed(0)
agent = Agent(state_size=(len(OBSTACLE_X)+1)*2, action_size=81, seed=0)


# max_t = 200, eps_.999, eps_end 0.1
def dqn(n_episodes=8000, max_t=100, eps_start=1, eps_end=0.05, eps_decay=0.9995):

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
    number_scores = 100
    scores_window = []  # last 100 scores
    eps = eps_start  # initialize epsilon
    index_model = 0

    for i_episode in range(1, n_episodes + 1):
        state = env.reset()
        score = 0
        for t in range(max_t):
            action = agent.act(state, eps)
            next_state, reward, done = env.step(action)
            agent.step(state, action, reward, next_state, done)
            state = next_state
            score += reward
            env.step_number = t
            env.render()
            if done:
                break
        scores_window.append(score)  # save most recent score
        scores.append(score)  # save most recent score
        eps = max(eps_end, eps_decay * eps)  # decrease epsilon

        print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_window)), end="")
        if i_episode % 100 == 0:
            scores_window_model = scores_window
            print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_window)))
            scores_graph.append(np.mean(scores_window))
            max_avg = np.mean(scores_window)
            scores_window = []

            if max_avg == np.max(scores_graph):
                if index_model == number_scores:
                    index_model = 0
                print("Model saved! maximum average reward was: {:.2f}".format(np.max(scores_window_model)))
                # saving all models from best run in a folder
                torch.save(agent.qnetwork_local.state_dict(), 'dqn_models/checkpoint{}.pth'.format(index_model))
                index_model += 1

    torch.save(agent.qnetwork_local.state_dict(), 'checkpoint_last.pth')
    return scores_graph

scores_graph = dqn()

# plot the scores
fig = plt.figure()
ax = fig.add_subplot(111)
plt.plot(np.arange(len(scores_graph)), scores_graph)
plt.ylabel('Average Reward')
plt.xlabel('Every 100 Episodes')
plt.show()