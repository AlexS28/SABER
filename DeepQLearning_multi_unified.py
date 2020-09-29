"""
Trains two models, a ground robot and a scouting drone. It simulates the robot being blind to obstacles,
encouraging the drone to do the majority of the exploring by penalizing movements harder when simulated
uncertainty is higher. Actions which would promote uncertainty in real life will punish the system harder.

The drone is encouraged to explore the environment quickly and return to the ground robot, and the ground
robot uses this information to stay close to the obstacles without hitting them. It is also encouraged to
get to the goal as quickly as possible, as the punishment for waiting is exponentially increasing. Hopefully,
the Network is able to find the optimal combination of waiting for the environment to be explored, and moving
toward the goal in a quick but certain way.
"""


############################## IMPORTS ##############################
import numpy as np
import math
import tensorflow.keras.backend as backend
from keras.models import Sequential, load_model
from keras.layers import Dense, Dropout, Conv2D, MaxPooling2D, Activation, Flatten
from keras.optimizers import Adam
from keras.callbacks import TensorBoard
import tensorflow as tf
from collections import deque
import time
import random
from tqdm import tqdm
import os
from PIL import Image
import cv2
import datetime
from scipy.spatial import distance

############################## SETTINGS ##############################

DISCOUNT = 0.99
REPLAY_MEMORY_SIZE = 50_000  # How many last steps to keep for model training
MIN_REPLAY_MEMORY_SIZE = 1_000  # Minimum number of steps in a memory to start training
MINIBATCH_SIZE = 64  # How many steps (samples) to use for training
UPDATE_TARGET_EVERY = 5  # Terminal states (end of episodes)x
MIN_REWARD = -200  # For model save
MEMORY_FRACTION = 0.20

# SETTINGS TO CHECK BEFORE RUNNING
TRAINING = True
MODEL_NUMBER = '5'
MAP_NUMBER = '5' # model N is trained on map N, but can be tested on different maps

# This is to protect silly programmers
if TRAINING:
    MAP_NUMBER = MODEL_NUMBER

# Environment settings
if TRAINING:
    EPISODES = 5000
    AGGREGATE_STATS_EVERY = 1  # show preview every episodes
else:
    EPISODES = 100
    AGGREGATE_STATS_EVERY = 1

# Exploration settings
robot_epsilon = 1  # limit robot random exploration
drone_epsilon = 1 # we want drone to explore more
EPSILON_DECAY = 0.99997
MIN_EPSILON = 0.001

SHOW_PREVIEW = True

# For stats
ep_rewards = [0]
ep_d_rewards = [0]


# For more repetitive results
random.seed(1)
np.random.seed(1)
tf.random.set_seed(1)


def translate(value, leftMin, leftMax, rightMin, rightMax):
    # Figure out how 'wide' each range is
    leftSpan = leftMax - leftMin
    rightSpan = rightMax - rightMin

    # Convert the left range into a 0-1 range (float)
    valueScaled = float(value - leftMin) / float(leftSpan)

    # Convert the 0-1 range into a value in the right range.
    return rightMin + (valueScaled * rightSpan)

############################## MAPS ##############################
MAP_SIZE = 10

if MAP_NUMBER == '4': # Two obstacles lined up parallel to goal path
    R1X = 0
    R1Y = 0
    D1X = 1
    D1Y = 0
    G1X = 9
    G1Y = 9
    OBSTACLE_X = [5,7]
    OBSTACLE_Y = [5,7]
    NUM_OBSTACLES = len(OBSTACLE_X)
elif MAP_NUMBER == '5':
    R1X = 0
    R1Y = 0
    D1X = 1
    D1Y = 0
    G1X = 9
    G1Y = 9
    OBSTACLE_X = [5,7,3,4,6,3,5,4,2,5,4,3]
    OBSTACLE_Y = [5,7,4,3,8,5,4,5,3,3,4,3]
    NUM_OBSTACLES = len(OBSTACLE_X)
elif MAP_NUMBER == '6': # Two obstacles lined up perpendicular to goal path
    R1X = 0
    R1Y = 0
    O1X = 3
    O1Y = 4
    O2X = 4
    O2Y = 3
    G1X = 9
    G1Y = 9
elif MAP_NUMBER == '7' or MAP_NUMBER == '10' or MAP_NUMBER == '11': # Fixed 4 obstacles
    R1X = 0
    R1Y = 5
    O1X = 3
    O1Y = 4
    O2X = 5
    O2Y = 8
    O3X = 2
    O3Y = 6
    O4X = 7
    O4Y = 7
    G1X = 9
    G1Y = 7
elif MAP_NUMBER == '8': # Random 4 obstacles each episode
    R1X = 0
    R1Y = 0
    O1X = np.random.randint(0, MAP_SIZE)
    O1Y = np.random.randint(0, MAP_SIZE)
    O2X = np.random.randint(0, MAP_SIZE)
    O2Y = np.random.randint(0, MAP_SIZE)
    O3X = np.random.randint(0, MAP_SIZE)
    O3Y = np.random.randint(0, MAP_SIZE)
    O4X = np.random.randint(0, MAP_SIZE)
    O4Y = np.random.randint(0, MAP_SIZE)
    G1X = 9
    G1Y = 9
elif MAP_NUMBER == '9': # Clumped obstacles away from goal (we want it to still go near obstacles)
    R1X = 5
    R1Y = 0
    O1X = 3
    O1Y = 6
    O2X = 3
    O2Y = 7
    O3X = 3
    O3Y = 4
    O4X = 3
    O4Y = 5
    G1X = 7
    G1Y = 9
elif MAP_NUMBER == '12': # Random 4 obstacles each episode
    R1X = 0
    R1Y = 0
    O1X = np.random.randint(0, MAP_SIZE)
    O1Y = np.random.randint(0, MAP_SIZE)
    O2X = np.random.randint(0, MAP_SIZE)
    O2Y = np.random.randint(0, MAP_SIZE)
    O3X = np.random.randint(0, MAP_SIZE)
    O3Y = np.random.randint(0, MAP_SIZE)
    O4X = np.random.randint(0, MAP_SIZE)
    O4Y = np.random.randint(0, MAP_SIZE)
    G1X = np.random.randint(0, MAP_SIZE)
    G1Y = np.random.randint(0, MAP_SIZE)
elif MAP_NUMBER == '13': # Random 4 obstacles each episode
    R1X = np.random.randint(0, MAP_SIZE)
    R1Y = np.random.randint(0, MAP_SIZE)
    O1X = np.random.randint(0, MAP_SIZE)
    O1Y = np.random.randint(0, MAP_SIZE)
    O2X = np.random.randint(0, MAP_SIZE)
    O2Y = np.random.randint(0, MAP_SIZE)
    O3X = np.random.randint(0, MAP_SIZE)
    O3Y = np.random.randint(0, MAP_SIZE)
    O4X = np.random.randint(0, MAP_SIZE)
    O4Y = np.random.randint(0, MAP_SIZE)
    G1X = np.random.randint(0, MAP_SIZE)
    G1Y = np.random.randint(0, MAP_SIZE)
#maze

############################## CLASSES ##############################


class Blob:
    def __init__(self, size, x ,y):
        self.size = size
        self.x = x
        self.y = y
        
    def __str__(self):
        return f"Blob ({self.x}, {self.y})"

    def __sub__(self, other):
        return (self.x-other.x, self.y-other.y)

    def __eq__(self, other):
        return self.x == other.x and self.y == other.y

    def action(self, choice):
        '''
        Gives us 9 total movement options. (0,1,2,3,4,5,6,7,8)
        '''
        if choice == 0:
            self.move(x=1, y=1)
        elif choice == 1:
            self.move(x=-1, y=-1)
        elif choice == 2:
            self.move(x=-1, y=1)
        elif choice == 3:
            self.move(x=1, y=-1)

        elif choice == 4:
            self.move(x=1, y=0)
        elif choice == 5:
            self.move(x=-1, y=0)

        elif choice == 6:
            self.move(x=0, y=1)
        elif choice == 7:
            self.move(x=0, y=-1)

        elif choice == 8:
            self.move(x=0, y=0)

    def move(self, x=None, y=None):

        # If no value for x, move randomly
        if x == None:
            self.x += np.random.randint(-1, 2)
        else:
            self.x += x

        # If no value for y, move randomly
        if y == None:
            self.y += np.random.randint(-1, 2)
        else:
            self.y += y

        # If we are out of bounds, fix!
        if self.x < 0:
            self.x = 0
        elif self.x > self.size-1:
            self.x = self.size-1
        if self.y < 0:
            self.y = 0
        elif self.y > self.size-1:
            self.y = self.size-1


class BlobEnv:
    SIZE = MAP_SIZE
    OBSERVATION_SPACE_VALUES = (SIZE, SIZE, 3)  # 4
    ACTION_SPACE_SIZE = 9
    
    MOVE_PENALTY = 1
    MOVE_PENALTY_EXP_CONSTANT = 0.01

    ROBOT_VISION_DISTANCE = 2
    DRONE_VISION_DISTANCE = 3
    
    OBSTACLE_COLLISION_PENALTY = 2000
    OBSTACLE_PROXIMITY_THRESHOLD = 2
    OBSTACLE_PROXIMITY_LIMIT = 5
    OBSTACLE_PROXIMITY_PENALTY = 1
    
    GOAL_REWARD = 1000
    GOAL_PROXIMITY_THRESHOLD = 10
    GOAL_PROXIMITY_MAX_REWARD = 1
    GOAL_REWARD_EXP_CONSTANT = 0.5

    DRONE_PROXIMITY_THRESHOLD = 5
    DRONE_PROXIMITY_REWARD = 1
    DRONE_DISCOVERY_REWARD = 50
    DRONE_MOVE_DISCOUNT = 0.5
    
    colors = {"robot"               : (255, 175, 0),
              "goal"                : (0, 255, 0),
              "unseen_obstacle"     : (0, 0, 75),
              "seen_obstacle"       : (0, 0, 255),
              "drone"               : (255, 0, 255)}

    def reset(self):
        self.robot = Blob(self.SIZE, R1X, R1Y)
        self.drone = Blob(self.SIZE, D1X, D1Y)
        self.goal = Blob(self.SIZE, G1X, G1Y)
        self.obstacles = []
        for i in range (NUM_OBSTACLES):
            new_obstacle = Blob(self.SIZE, OBSTACLE_X[i], OBSTACLE_Y[i])
            self.obstacles.append(new_obstacle)
        self.unseen_obstacles = self.obstacles
        self.seen_obstacles = []
        self.episode_step = 0

        observation = np.array(self.get_image())
        return observation

    def step(self, robot_action, drone_action):
        self.episode_step += 1
        
        self.robot.action(robot_action)
        self.drone.action(drone_action)
        
        done = False
        uncertainty = 0
        new_observation = np.array(self.get_image())

        robot_reward = 0
        for obstacle in self.seen_obstacles:
            if self.robot == obstacle:
                robot_reward -= self.OBSTACLE_COLLISION_PENALTY
                done = True
        if not done and self.robot == self.goal:
            robot_reward += self.GOAL_REWARD
            done = True
        elif not done:
            # punish more as time goes on for taking a long time (moving and staying still), and punish more for moving (but not staying still) in underexplored environment, but punish harder for staying still in explored evironment
            if (NUM_OBSTACLES == len(self.seen_obstacles)) and (robot_action == 8):
                robot_reward -= 2*(self.MOVE_PENALTY * math.exp(self.MOVE_PENALTY_EXP_CONSTANT*self.episode_step))
            elif robot_action == 8:
                robot_reward -= (self.MOVE_PENALTY * math.exp(self.MOVE_PENALTY_EXP_CONSTANT*self.episode_step))
            else:
                robot_reward -= (self.MOVE_PENALTY * math.exp((NUM_OBSTACLES - len(self.seen_obstacles) + 1)*self.MOVE_PENALTY_EXP_CONSTANT*self.episode_step))
            
            # check for new seen obstacles within (robot vision range)
            for obstacle in self.unseen_obstacles:
                robot_to_obstacle = distance.euclidean((self.robot.x,self.robot.y) ,(obstacle.x,obstacle.y))
                if self.ROBOT_VISION_DISTANCE >= robot_to_obstacle:
                    self.seen_obstacles.append(obstacle)
                    self.unseen_obstacles.remove(obstacle)
            # if the distance from any seen obstacles that are (reasonably close) is (too great) punish robot because we want uncertainty lower and being closer to obstacles decreases uncertainty
            for obstacle in self.seen_obstacles:
                robot_to_seen_obstacle = distance.euclidean((self.robot.x,self.robot.y) ,(obstacle.x,obstacle.y))
                if self.OBSTACLE_PROXIMITY_LIMIT >= robot_to_seen_obstacle >= self.OBSTACLE_PROXIMITY_THRESHOLD:
                    robot_reward -= self.OBSTACLE_PROXIMITY_PENALTY
            # constantly rewarded based on distance from the goal if (close enough)
            robot_to_goal = distance.euclidean((self.robot.x,self.robot.y) ,(self.goal.x,self.goal.y))
            if robot_to_goal <= self.GOAL_PROXIMITY_THRESHOLD:
                robot_reward += (self.GOAL_PROXIMITY_MAX_REWARD * math.exp(self.GOAL_REWARD_EXP_CONSTANT*(self.SIZE - robot_to_goal)))
            # if distance from any other agent is (close enough) reward it because we want uncertainty lower and being closer to another agent decreases uncertainty
            robot_to_drone = distance.euclidean((self.robot.x,self.robot.y) ,(self.drone.x,self.drone.y))
            if robot_to_drone <= self.DRONE_PROXIMITY_THRESHOLD:
                robot_reward += self.DRONE_PROXIMITY_REWARD
            # reward based on obstacles seen
            robot_reward += (len(self.seen_obstacles)/NUM_OBSTACLES)
            
        drone_reward = 0
        # check for new seen obstacles within (drone vision range)
        for obstacle in self.unseen_obstacles:
            drone_to_obstacle = distance.euclidean((self.drone.x,self.drone.y) ,(obstacle.x,obstacle.y))
            if self.DRONE_VISION_DISTANCE >= drone_to_obstacle:
                self.seen_obstacles.append(obstacle)
                self.unseen_obstacles.remove(obstacle)
                drone_reward += self.DRONE_DISCOVERY_REWARD
        # if distance from any other agent is (close enough) reward it because we want uncertainty lower and being closer to another agent decreases uncertainty, but if (too far) we punish it
        robot_to_drone = distance.euclidean((self.robot.x,self.robot.y) ,(self.drone.x,self.drone.y))
        if robot_to_drone <= self.DRONE_PROXIMITY_THRESHOLD:
            drone_reward += self.DRONE_PROXIMITY_REWARD
        elif robot_to_drone > self.DRONE_PROXIMITY_THRESHOLD:
            drone_reward -= self.DRONE_PROXIMITY_REWARD
        # reward based on obstacles seen (encourages exploration) but diminishes overtime (encourages fast exploration)
        if len(self.seen_obstacles) != len(self.unseen_obstacles):
            drone_reward += math.log(abs((len(self.seen_obstacles) - len(self.unseen_obstacles))/NUM_OBSTACLES))
        # punish based on obstacles left to see (encourages exploration) and increases overtime (encourages fast exploration)
        if self.unseen_obstacles:
            drone_reward -= math.exp(abs((NUM_OBSTACLES - len(self.seen_obstacles))/NUM_OBSTACLES))
        # punish (some fraction of the robot movement punishment) for moving unless all obstacles are seen
        if len(self.seen_obstacles) != NUM_OBSTACLES:
            drone_reward -= self.DRONE_MOVE_DISCOUNT*(self.MOVE_PENALTY * math.exp(self.MOVE_PENALTY_EXP_CONSTANT*self.episode_step))
            
        if self.episode_step >= 200:
            done = True
        reward = drone_reward + robot_reward
        return new_observation, reward, done

    def render(self):
        img = self.get_image()
        img = img.resize((300, 300))  # resizing so we can see our agent in all its glory.
        cv2.imshow("image", np.array(img))  # show it!
        cv2.waitKey(1)

    def get_image(self):
        env = np.zeros((self.SIZE, self.SIZE, 3), dtype=np.uint8)  # starts an rbg of our size
        env[self.robot.x][self.robot.y] = self.colors["robot"]  # sets the robot tile to blue
        env[self.drone.x][self.drone.y] = self.colors["drone"]  # sets the drone tile to purple
        env[self.goal.x][self.goal.y] = self.colors["goal"]  # sets the goal location tile to green color
        #for obstacle in self.unseen_obstacles:
            #env[obstacle.x][obstacle.y] = self.colors["unseen_obstacle"]  # sets the obstacle locations to dark red
        for obstacle in self.seen_obstacles:
            env[obstacle.x][obstacle.y] = self.colors["seen_obstacle"]  # sets the seen obstacle locations to bright red

        img = Image.fromarray(env, 'RGB')  # reading to rgb. Apparently. Even tho color definitions are bgr. ???
        return img

env = BlobEnv()


# Create models folder
if not os.path.isdir('models'):
    os.makedirs('models')


# Agent class
class DQNAgent:
    def __init__(self):

        # Main model
        self.model = self.create_model()

        # Target network
        self.target_model = self.create_model()
        self.target_model.set_weights(self.model.get_weights())

        # An array with last n steps for training
        self.replay_memory = deque(maxlen=REPLAY_MEMORY_SIZE)

        # Custom tensorboard object
         
        log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        self.tensorboard = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

        # Used to count when to update target network with main network's weights
        self.target_update_counter = 0

    def create_model(self):
        model = Sequential()

        model.add(Conv2D(256, (3, 3), input_shape=env.OBSERVATION_SPACE_VALUES))  # OBSERVATION_SPACE_VALUES = (10, 10, 3) a 10x10 RGB image.
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.2))

        model.add(Conv2D(256, (3, 3)))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.2))

        model.add(Flatten())  # this converts our 3D feature maps to 1D feature vectors
        model.add(Dense(64))

        model.add(Dense(2*env.ACTION_SPACE_SIZE, activation='linear'))  # ACTION_SPACE_SIZE = how many choices (9)
        model.compile(loss="mse", optimizer=Adam(lr=0.1), metrics=['accuracy'])
        return model

    # Adds step's data to a memory replay array
    # (observation space, action, reward, new observation space, done)
    def update_replay_memory(self, transition):
        self.replay_memory.append(transition)

    # Trains main network every step during episode
    def train(self, terminal_state, step):

        # Start training only if certain number of samples is already saved
        if len(self.replay_memory) < MIN_REPLAY_MEMORY_SIZE:
            return

        # Get a minibatch of random samples from memory replay table
        minibatch = random.sample(self.replay_memory, MINIBATCH_SIZE)

        # Get current states from minibatch, then query NN model for Q values
        current_states = np.array([transition[0] for transition in minibatch])/255
        current_qs_list = self.model.predict(current_states)

        # Get future states from minibatch, then query NN model for Q values
        # When using target network, query it, otherwise main network should be queried
        new_current_states = np.array([transition[3] for transition in minibatch])/255
        future_qs_list = self.target_model.predict(new_current_states)

        X = []
        y = []

        # Now we need to enumerate our batches
        for index, (current_state, action, reward, new_current_state, done) in enumerate(minibatch):

            # If not a terminal state, get new q from future states, otherwise set it to 0
            # almost like with Q Learning, but we use just part of equation here
            if not done:
                max_future_q = np.max(future_qs_list[index])
                new_q = reward + DISCOUNT * max_future_q
            else:
                new_q = reward

            # Update Q value for given state
            current_qs = current_qs_list[index]
            current_qs[action] = new_q

            # And append to our training data
            X.append(current_state)
            y.append(current_qs)

        # Fit on all samples as one batch, log only on terminal state
        self.model.fit(np.array(X)/255, np.array(y), batch_size=MINIBATCH_SIZE, verbose=0, shuffle=False, callbacks=[self.tensorboard] if terminal_state else None)

        # Update target network counter every episode
        if terminal_state:
            self.target_update_counter += 1

        # If counter reaches set value, update target network with weights of main network
        if self.target_update_counter > UPDATE_TARGET_EVERY:
            self.target_model.set_weights(self.model.get_weights())
            self.target_update_counter = 0

    # Queries main network for Q values given current observation space (environment state)
    def get_qs(self, state):

        actions = self.model.predict(np.array(state).reshape(-1, *state.shape)/255)[0]

        robot_actions = actions[0:8]
        drone_actions = actions[9:18]
        
        return robot_actions, drone_actions


########## TRAINING ##########
if TRAINING:
    robot_agent = DQNAgent()


    # Iterate over episodes
    for episode in tqdm(range(1, EPISODES + 1), ascii=True, unit='episodes'):

        # Update tensorboard step every episode
        robot_agent.tensorboard.step = episode

        # Restarting episode - reset episode reward and step number
        episode_robot_reward = 0
        step = 1

        # Reset environment and get initial state
        current_state = env.reset()

        # Reset flag and start iterating until episode ends
        done = False
        while not done:
            if np.random.random() > robot_epsilon:
                #Get action from Q table
                robot_actions, drone_actions = robot_agent.get_qs(current_state)
                robot_action = np.argmax(robot_actions)
                drone_action = np.argmax(drone_actions)
                print("calc")
            else:
                # Get random action
                print("rand")
                robot_action = np.random.randint(0, env.ACTION_SPACE_SIZE)
                drone_action = np.random.randint(0, env.ACTION_SPACE_SIZE)


            new_state, robot_reward, done = env.step(robot_action, drone_action)

            # Transform new continuous state to new discrete state and count reward
            episode_robot_reward += robot_reward
##            episode_drone_reward += drone_reward

            if SHOW_PREVIEW and not episode % AGGREGATE_STATS_EVERY:
                env.render()

            # Every step we update replay memory and train the networks
            robot_agent.update_replay_memory((current_state, robot_action, robot_reward, new_state, done))
            robot_agent.train(done, step)
##            drone_agent.update_replay_memory((current_state, drone_action, drone_reward, new_state, done))
##            drone_agent.train(done, step)

            current_state = new_state
            step += 1

        # Append episode reward to a list and log stats (every given number of episodes)
        ep_rewards.append(episode_robot_reward)
##        ep_d_rewards.append(episode_drone_reward)
        if not episode % AGGREGATE_STATS_EVERY or episode == 1:
            #average_reward = sum(ep_rewards[-AGGREGATE_STATS_EVERY:])/len(ep_rewards[-AGGREGATE_STATS_EVERY:])
            #min_reward = min(ep_rewards[-AGGREGATE_STATS_EVERY:])
            #max_reward = max(ep_rewards[-AGGREGATE_STATS_EVERY:])
            #agent.tensorboard.update_stats(reward_avg=average_reward, reward_min=min_reward, reward_max=max_reward, epsilon=epsilon)

            # Save model
            robot_agent.model.save(f'models/model' + MODEL_NUMBER + '_multi_unified.model')
##            drone_agent.model.save(f'models/model' + MODEL_NUMBER + '_multi_drone.model')

        # Decay epsilons
        if robot_epsilon > MIN_EPSILON:
            robot_epsilon *= EPSILON_DECAY
            robot_epsilon = max(MIN_EPSILON, robot_epsilon)
##        if drone_epsilon > MIN_EPSILON:
##            drone_epsilon *= EPSILON_DECAY
##            drone_epsilon = max(MIN_EPSILON, drone_epsilon)
        print(episode_robot_reward)
        print(robot_epsilon)
##        print(episode_drone_reward)
        
    # Create model results folder
    if not os.path.isdir('results/model' + MODEL_NUMBER + '_multi'):
        os.makedirs('results/model' + MODEL_NUMBER + '_multi')
    np.savetxt(f'results/model' + MODEL_NUMBER + '_multi/model' + MODEL_NUMBER + '_multi_training_rewards_unified.csv', ep_rewards)
##    np.savetxt(f'results/model' + MODEL_NUMBER + '_multi/model' + MODEL_NUMBER + '_multi_training_rewards_drone.csv', ep_d_rewards)

########## TESTING ##########
if not TRAINING:
    agent = DQNAgent()
    agent.model = load_model(f'models/model' + MODEL_NUMBER + '_multi.model')


    # Iterate over episodes
    for episode in tqdm(range(1, EPISODES + 1), ascii=True, unit='episodes'):

        # Update tensorboard step every episode
        agent.tensorboard.step = episode

        # Restarting episode - reset episode reward and step number
        episode_reward = 0
        step = 1

        # Reset environment and get initial state
        current_state = env.reset()

        # Reset flag and start iterating until episode ends
        done = False
        while not done:

            # This part stays mostly the same, the change is to query a model for Q values
            if np.random.random() > epsilon:
                # Get action from Q table
                action = np.argmax(agent.get_qs(current_state))
            else:
                # Get random action
                action = np.random.randint(0, env.ACTION_SPACE_SIZE)

            new_state, reward, done = env.step(action)

            # Transform new continous state to new discrete state and count reward
            episode_reward += reward

            if SHOW_PREVIEW and not episode % AGGREGATE_STATS_EVERY:
                env.render()

            # Every step we update replay memory and train main network
            agent.update_replay_memory((current_state, action, reward, new_state, done))

            current_state = new_state
            step += 1

        # Append episode reward to a list and log stats (every given number of episodes)
        ep_rewards.append(episode_reward)
        if not episode % AGGREGATE_STATS_EVERY or episode == 1:
            average_reward = sum(ep_rewards[-AGGREGATE_STATS_EVERY:])/len(ep_rewards[-AGGREGATE_STATS_EVERY:])
            min_reward = min(ep_rewards[-AGGREGATE_STATS_EVERY:])
            max_reward = max(ep_rewards[-AGGREGATE_STATS_EVERY:])
        
        # Decay epsilon
        if epsilon > MIN_EPSILON:
            epsilon *= EPSILON_DECAY
            epsilon = max(MIN_EPSILON, epsilon)
        #print(episode_reward)

    # Create model results folder
    if not os.path.isdir('results/model' + MODEL_NUMBER + '_multi'):
        os.makedirs('results/model' + MODEL_NUMBER + '_multi')
    np.savetxt(f'results/model' + MODEL_NUMBER + '_multi/model' + MODEL_NUMBER + '_multi_testing_rewards.csv', ep_rewards)
    
############################



