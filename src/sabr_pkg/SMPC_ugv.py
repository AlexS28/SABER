"""FILE CREATED BY: Alexander Schperberg, aschperb@gmail.com
Copyright by RoMeLa (Robotics and Mechanisms Laboratory, University of California, Los Angeles)"""

# This file provides a stochastic and robust model predictive controller for a simple unmanned ground vehicle that
# moves a ground vehicle to any desired goal location, while considering obstacles (represented as polygons and circles)
# and with cross communication consideration with another robot (using cooperative localization algorithms)

from casadi import *
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import mpl_toolkits.mplot3d.art3d as art3d
from matplotlib.patches import Circle
from scipy.spatial import distance
import matplotlib.pyplot as plt
import math as m
import control
from scipy.stats import linregress
#from ROS_interface import *

class SMPC_UGV_Planner():

    def __init__(self, dT, mpc_horizon, curr_pos, robot_size, lb_state,
                 ub_state, lb_control, ub_control, Q, R, angle_noise_r1, angle_noise_r2,
                 relative_measurement_noise_cov, maxComm_distance, obs, animate):

        # initialize Optistack class
        self.opti = casadi.Opti()
        # dt = discretized time difference
        self.dT = dT
        # mpc_horizon = number of time steps for the mpc to look ahead
        self.N = mpc_horizon
        # robot_size = input a radius value, where the corresponding circle represents the size of the robot
        self.robot_size = robot_size
        # lower_bound_state = numpy array corresponding to the lower limit of the robot states, e.g.
        # lb_state = np.array([[-20], [-20], [-pi], dtype=float), the same for the upper limit (ub). Similar symbolic
        # representation for the controls (lb_control and ub_control) as well
        self.lb_state = lb_state
        self.ub_state = ub_state
        self.lb_control = lb_control
        self.ub_control = ub_control
        # Q and R diagonal matrices, used for the MPC objective function, Q is 3x3, R is 4x4 (first 2 diagonals
        # represent the cost on linear and angular velocity, the next 2 diagonals represent cost on state slack,
        # and terminal slack respectively. The P diagonal matrix represents the cost on the terminal constraint.
        self.Q = Q
        self.R_dare = R
        self.R = np.array([[R[0,0], 0], [0, R[2,2]]])
        # initialize discretized state matrices A and B (note, A is constant, but B will change as it is a function of
        # state theta)
        self.A = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
        self.B = np.array([[self.dT, 0, 0], [0, self.dT, 0], [0, 0, self.dT]])
        # initialize the P matrix, which is the cost matrix that defines the optimal state feedback controller
        self.P, _, _ = control.dare(self.A, self.B, self.Q, self.R_dare)
        # initalize cost on slack
        self.slack_cost = 1000
        # initialize measurement noise (in our calculation, measurement noise is set by the user and is gaussian,
        # zero-mean). It largely represents the noise due to communication transmitters, or other sensor devices. It
        # is assumed to be a 3x3 matrix (x, y, and theta) for both robots
        self.relative_measurement_noise_cov = relative_measurement_noise_cov
        # we assume that there is constant noise in angle (while x and y are dynamically updated) - should be a variance
        # value
        self.angle_noise_r1 = angle_noise_r1
        self.angle_noise_r2 = angle_noise_r2
        # initialize the maximum distance that robot 1 and 2 are allowed to have for cross communication
        self.maxComm_distance = maxComm_distance
        # distance to obstacle to be used as constraints
        self.max_obs_distance = 20
        # initialize obstacles
        self.obs = obs
        # initialize robot's current position
        self.curr_pos = curr_pos
        # self.change_goal_point(goal_pos)
        # initialize the current positional uncertainty (and add the robot size to it)
        # TODO: this is a temporary fix for testing
        self.r1_cov_curr = np.array([[0.1 + self.robot_size, 0], [0, 0.1 + self.robot_size]])
        # initialize cross diagonal system noise covariance matrix
        self.P12 = np.array([[0, 0], [0, 0]])
        # bool variable to indicate whether the robot has made first contact with the uav
        self.first_contact = False
        # initialize state, control, and slack variables
        self.initVariables()
        # initialize states for DQN (relative distance between robot-goal, and robot-obstacles
        num_obs_const = 0
        for i in range(1, len(self.obs)+1):
            num_obs_const += self.obs[i]['polygon_type']
        self.dqn_states = np.zeros((num_obs_const,))
        # initialize parameters for animation
        if animate:
            plt.ion()
            fig = plt.figure()
            fig.canvas.mpl_connect('key_release_event',
                                   lambda event: [exit(0) if event.key == 'escape' else None])
            self.ax = fig.add_subplot(111, projection='3d')
            self.ax = Axes3D(fig)
            u = np.linspace(0, 2 * np.pi, 100)
            v = np.linspace(0, np.pi, 100)
            self.x_fig = np.outer(self.robot_size * np.cos(u), np.sin(v))
            self.y_fig = np.outer(self.robot_size * np.sin(u), np.sin(v))
            self.z_fig = np.outer(self.robot_size * np.ones(np.size(u)), np.cos(v))

    def initVariables(self):

        # initialize x, y, and theta state variables
        self.X = self.opti.variable(3, self.N+1)
        self.x_pos = self.X[0,:]
        self.y_pos = self.X[1,:]
        self.th = self.X[2, :]
        self.opti.set_initial(self.X, 0)

        # initialize, linear and angular velocity control variables (v, and w), and repeat above procedure
        self.U = self.opti.variable(2, self.N)
        self.v = self.U[0, :]
        self.w = self.U[1, :]
        self.opti.set_initial(self.U, 0)

        # initialize the current robot pos (x, y and th current position)
        self.r1_pos = self.opti.parameter(3, 1)
        self.opti.set_value(self.r1_pos, self.curr_pos)

        # initialize the slack variables
        self.slack = self.opti.variable(1, self.N + 1)

        # initialize the goal robot pos (x, y, and th goal position)
        self.r1_goal = self.opti.parameter(3, 1)

        # initialize the angle noise for robot 1
        self.angle_noise = self.opti.parameter(1, 1)
        self.opti.set_value(self.angle_noise, self.angle_noise_r1)

        # initialize the uncertainty covariances from the RNN, provided by robot 1 (4 x 1 vector per covariance matrix)
        # must be positive semi-definite, from t+1 to N
        self.r1_pos_cov = self.opti.parameter(4, self.N+1)
        self.opti.set_value(self.r1_pos_cov, 0)

        # initialize the uncertainty covariances from the RNN, provided by robot 2 (4 x 1 vector per covariance matrix)
        # must be positive semi-definite, from t+1 to N
        self.r2_pos_cov = self.opti.parameter(4, self.N+1)
        self.opti.set_value(self.r2_pos_cov, 0)

        # initialize robot 2, future positions (x, y, and th), from t+1 to N
        self.r2_traj = self.opti.parameter(3, self.N+1)
        self.opti.set_value(self.r2_traj, 0)

        # initialize the objective function
        self.obj()

    def obj(self):

        self.objFunc = 0

        for k in range(0, self.N-1):
            con = self.U[:, k]
            st = self.X[:, k + 1]

            self.objFunc = self.objFunc + mtimes(mtimes((st - self.r1_goal).T, self.Q), st - self.r1_goal) + \
                           0.5*mtimes(mtimes(con.T, self.R), con) + self.slack[:, k+1]*self.slack_cost

        st = self.X[:, self.N]
        self.objFunc = self.objFunc + mtimes(mtimes((st - self.r1_goal).T, self.P), st - self.r1_goal) + self.slack[:,self.N]*self.slack_cost

        # initialize the constraints for the objective function
        self.init_constraints()

    def init_constraints(self):

        # constrain the current state, and bound current and future states by their limits
        self.opti.subject_to(self.X[:, 0] == self.r1_pos)
        self.opti.subject_to(self.opti.bounded(self.lb_state, self.X, self.ub_state))
        self.opti.subject_to(self.opti.bounded(self.lb_control, self.U, self.ub_control))

        # constrain slack variable
        self.opti.subject_to(self.slack >= 0)

        # initiate multiple shooting constraints
        for k in range(0, self.N):
             next_state = if_else((sqrt((self.X[0, k] - self.r2_traj[0, k])**2 +
                                      (self.X[1, k] - self.r2_traj[1, k]**2)) >= self.maxComm_distance),
                                 self.update_1(self.X[:,k], self.U[0:2,k]), self.update_2(self.X[:,k], self.U[0:2,k], k))

             self.opti.subject_to(self.X[:,k + 1] == next_state)

        if self.obs:
            # initialize obstacles, animate them, and also constrain them for the MPC
            self.init_obstacles(self.obs, self.animate)

        # initialize the objective function into the solver
        self.opti.minimize(self.objFunc)

        # initiate the solver
        self.pre_solve()

    def init_obstacles(self, obstacles, animate):
        # receive the slope, intercepts, of the obstacles for chance constraints, and plot
        for i in range(1, len(obstacles)+1):
            it = 0
            slopes = []
            intercepts = []
            a_vectors = np.empty((2, obstacles[i]['polygon_type']))

            if obstacles[i]['polygon_type'] != 1:

                for j in range(0, obstacles[i]['polygon_type']):

                    if it == obstacles[i]['polygon_type']-1:
                        point_1 = obstacles[i]['vertices'][-1]
                        point_2 = obstacles[i]['vertices'][0]

                    else:
                        point_1 = obstacles[i]['vertices'][it]
                        point_2 = obstacles[i]['vertices'][it+1]

                    it += 1

                    x = [point_1[0], point_2[0]]
                    y = [point_1[1], point_2[1]]
                    _, intercept, _, _, _ = linregress(x, y)

                    a_x = x[1] - x[0]
                    a_y = y[1] - y[0]
                    slope = a_y / a_x
                    distance = np.sqrt(a_x**2 + a_y**2)
                    slopes.append(slope)
                    a_norm = np.array([a_x / distance, a_y / distance], dtype=float).reshape(2, 1)

                    # rotate the a_norm counter clockwise
                    a_norm = np.array([a_norm[1]*-1, a_norm[0]], dtype=float).reshape(1, 2)
                    a_vectors[:, j] = a_norm
                    intercepts = np.append(intercepts, intercept)

                obstacles[i]['a'] = a_vectors
                obstacles[i]['slopes'] = slopes
                obstacles[i]['intercepts'] = intercepts
            self.obs = obstacles

        if animate:
            self.x_list = []
            self.y_list = []
            self.z_list = []
            for i in range(1, len(obstacles)+1):
                if obstacles[i]['polygon_type'] != 0:
                    x_ani = []
                    y_ani = []
                    z_ani = []
                    vertices = self.obs[i]['vertices']
                    for j in range(0, len(vertices)):
                        x_ani.append(vertices[j][0])
                        y_ani.append(vertices[j][1])
                        z_ani.append(0.1)

                    self.x_list.append(x_ani)
                    self.y_list.append(y_ani)
                    self.z_list.append(z_ani)

        # initialize chance constraints for obstacle avoidance
        self.chance_constraints()

    def chance_constraints(self):

        # Using chance constraints on polygon obstacles
        # create integer variable for chance constraints
        self.obs_indexL = []
        for i in range(1, len(self.obs)+1):
            if self.obs[i]['polygon_type'] != 1:
                self.obs_indexL.append(self.obs[i]['polygon_type'])

        self.I = self.opti.variable(sum(self.obs_indexL), 1)
        # provide constraints on the integer variable
        self.opti.subject_to(self.opti.bounded(0, self.I, 1))

        # set chance constraints for obstacles
        # initialize c parameter for chance constraint equation, this value will change for each time step
        self.cl = self.opti.parameter(sum(self.obs_indexL), self.N+1)
        self.opti.set_value(self.cl, 1)

        # initialize a switch variable, to turn off or on obstacle constraints if the obstacle is not in a desired range
        self.switch_obsL = self.opti.parameter(len(self.obs_indexL), 1)
        self.opti.set_value(self.switch_obsL, 0)

        # initialize integer constraints
        iter_2 = 0
        for i in range(0, len(self.obs_indexL)):
            sum_I = 0
            iter_1 = iter_2
            iter_2 = iter_2 + self.obs_indexL[i]
            for j in range(iter_1, iter_2):
                sum_I = sum_I + self.I[j]
            self.opti.subject_to(sum_I >= 1)

        iter_2 = 0
        for i in range(0, len(self.obs_indexL)):
            iter_1 = iter_2
            iter_2 = iter_2 + self.obs_indexL[i]
            index_slope_intercept = 0
            r = self.obs[i + 1]['risk']

            for j in range(iter_1, iter_2):
                a = self.obs[i + 1]['a'][:, index_slope_intercept]
                b = self.obs[i + 1]['intercepts'][index_slope_intercept]
                m = self.obs[i + 1]['slopes'][index_slope_intercept]
                self.opti.set_value(self.cl[j,:], np.sqrt(np.dot(np.dot(2 * np.transpose(a), self.r1_cov_curr), a)) * erfinv((1 - 2 * r)))
                index_slope_intercept += 1

                for k in range(0, self.N+1):
                    dist = self.distance_pt_line(m, a, b, self.X[0:2,k])
                    self.opti.subject_to(self.switch_obsL[i]*dist * self.I[j] >= self.cl[j,k] * self.I[j] * self.switch_obsL[i] - self.slack[0,k])

        # Using chance constraints on circular obstacles
        self.obs_indexC = []
        for i in range(1, len(self.obs)+1):
            if self.obs[i]['polygon_type'] == 1:
                self.obs_indexC.append(self.obs[i]['polygon_type'])

        self.cc = self.opti.parameter(sum(self.obs_indexC), self.N+1)
        self.opti.set_value(self.cc, 1)

        self.switch_obsC = self.opti.parameter(len(self.obs_indexC), 1)
        self.opti.set_value(self.switch_obsC, 0)

        for i in range(1, len(self.obs)+1):
            iter = 0
            if self.obs[i]['polygon_type'] == 1:
                a = np.array([1,1]).reshape(2,1)
                r = self.obs[i]['risk']
                center = self.obs[i]['vertices'][0]
                size = self.obs[i]['size']

                self.opti.set_value(self.cc[iter,:],
                                np.sqrt(np.dot(np.dot(2 * np.transpose(a), self.r1_cov_curr), a)) * erfinv((1 - 2 * r)))

                for k in range(0, self.N+1):
                    dist = -1 * self.distance_pt_circle(center, self.X[0:2,k],size, self.robot_size) + self.cc[iter] - self.slack[0, k]
                    self.opti.subject_to(self.switch_obsC[iter]*dist <= 0)
                iter += 1

    def check_obstacles(self, curr_pos):
        # this function is run to update obstacle constraints for all timesteps of the MPC prediction horizon

        iter = 0
        iter2 = 0
        obs_iter2 = 0
        ind_dqn = 0

        #self.dqn_states[0] = distance.euclidean((curr_pos[0], curr_pos[1]), (goal_pos[0], goal_pos[1]))
        for i in range(1, len(self.obs)+1):

            if self.obs[i]['polygon_type'] != 1:
                break_now = False
                ind = 0
                for j in range(0, self.obs[i]['polygon_type']):

                    if ind == self.obs[i]['polygon_type']-1:
                        a, b = np.asarray(self.obs[i]['vertices'][0]), np.asarray(self.obs[i]['vertices'][-1])
                        dist = self.distance_pt_line_check(curr_pos, a, b)
                        self.dqn_states[ind_dqn] = np.round(dist,2)
                        ind_dqn +=1
                    else:
                        a, b = np.asarray(self.obs[i]['vertices'][ind]), np.asarray(self.obs[i]['vertices'][ind+1])
                        dist = self.distance_pt_line_check(curr_pos, a, b)
                        self.dqn_states[ind_dqn] = np.round(dist,2)
                        ind_dqn +=1
                        ind += 1

                    if dist <= self.max_obs_distance and not break_now:

                        obs_iter1 = obs_iter2
                        obs_iter2 = obs_iter2 + self.obs_indexL[iter]

                        index_slope_intercept = 0
                        r = self.obs[i]['risk']

                        for l in range(obs_iter1, obs_iter2):
                            a = self.obs[i]['a'][:, index_slope_intercept]

                            for k in range(0, self.N+1):
                                # self.r1_cov_curr[0,:] = self.r1_pos_cov[0:2,k]
                                # self.r1_cov_curr[1, :] = self.r1_pos_cov[2:4,k]
                                self.opti.set_value(self.cl[l, k], np.sqrt(
                                        np.dot(np.dot(2 * np.transpose(a), self.r1_cov_curr), a)) * erfinv((1 - 2 * r)))

                            index_slope_intercept += 1

                        self.opti.set_value(self.switch_obsL[iter], 1)
                        break_now = True
                    elif dist > self.max_obs_distance and not break_now:
                        self.opti.set_value(self.switch_obsL[iter], 0)

                iter += 1

            else:
                center = self.obs[i]['vertices'][0]
                size = self.obs[i]['size']
                dist = self.distance_pt_circle(center, curr_pos, size, self.robot_size)
                self.dqn_states[ind_dqn] = np.round(dist,2)
                ind_dqn += 1
                if dist <= self.max_obs_distance:
                    a = np.array([1, 1]).reshape(2, 1)
                    r = self.obs[i]['risk']

                    for k in range(0, self.N+1):
                        #self.r1_cov_curr[0,:] = self.r1_pos_cov[0:2,k]
                        #self.r1_cov_curr[1, :] = self.r1_pos_cov[2:4,k]
                        self.opti.set_value(self.cc[iter2, k],
                                        np.sqrt(np.dot(np.dot(2 * np.transpose(a), self.r1_cov_curr), a)) * erfinv(
                                            (1 - 2 * r)))
                    self.opti.set_value(self.switch_obsC[iter2], 1)
                    iter2 += 1
                    break
                else:
                    self.opti.set_value(self.switch_obsC[iter2], 0)
                    iter2 += 1
                    break
    """
    def rotation_constraints(self):
        # rotation constraints can be used to ensure that the robot is directed along the path it is moving
        gRotx = []
        gRoty = []
        for k in range(0, self.N):
            rhsx = (cos(self.X[2, k]) * (self.U[0, k]) + sin(self.X[2, k]) * (self.U[1, k]))
            gRotx = vertcat(gRotx, rhsx)
        for k in range(0, self.N):
            rhsy = (-sin(self.X[2, k]) * (self.U[0, k]) + cos(self.X[2, k]) * (self.U[1, k]))
            gRoty = vertcat(gRoty, rhsy)
        self.opti.subject_to(self.opti.bounded(-1.8, gRotx, 1.8))
        self.opti.subject_to(self.opti.bounded(0, gRoty, 0))
    """

    def pre_solve(self):
        # initiate the solver called bonmin - performs Mixed-Integer Nonlinear Programming (MINLP)

        # ensure states X, and controls U are continuous, while I variables are integers
        OT_Boolvector_X = [0]*self.X.size()[0]*self.X.size()[1]
        OT_Boolvector_U = [0]*self.U.size()[0]*self.U.size()[1]
        OT_Boolvector_Slack = [0]*self.slack.size()[0]*self.slack.size()[1]
        if self.obs:
            OT_Boolvector_Int = [1] * self.I.size()[0] * self.I.size()[1]
        else:
            OT_Boolvector_Int = []
        OT_Boolvector = OT_Boolvector_X + OT_Boolvector_U + OT_Boolvector_Slack + OT_Boolvector_Int

        opts = {'bonmin.warm_start': 'interior_point', 'discrete': OT_Boolvector, 'error_on_fail': True, 'bonmin.time_limit': 1.0,
                'bonmin.acceptable_obj_change_tol': 1e40, 'bonmin.acceptable_tol': 1e-1, 'bonmin.sb': 'yes', 'bonmin.bb_log_level':0}

        # create the solver
        self.opti.solver('bonmin', opts)

    # the nominal next state is calculated for use as a terminal constraint in the objective function
    def next_state_nominal(self, x, u):
        next_state = mtimes(self.A, x) +  mtimes(self.dT,vertcat(u[0]*cos(x[2]), u[0]*sin(x[2]), u[1]))
        return next_state

    # the next state is calculated with consideration of system noise, also considered the true state
    def next_state_withSystemNoise(self, x, u, system_noise_cov):
        # the system_noise_covariance will be a flattened 1x4 array, provided by the output of an RNN. We need to
        # convert it into a 3x3 matrix. We will assume a constant noise in theta however.
        system_noise_cov_converted = np.array([[self.opti.value(system_noise_cov[0]),
                                                    self.opti.value(system_noise_cov[1])],
                                                   [self.opti.value(system_noise_cov[2]),
                                                    self.opti.value(system_noise_cov[3])]])

        # sample a gaussian distribution of the system_noise covariance (for x and y)
        system_noise_xy = np.random.multivariate_normal([0, 0], system_noise_cov_converted,
                                                            check_valid='warn').reshape(2, 1)
        # sample a gaussian distribution of theta
        # system_noise_th = np.sqrt(angle_noise_r1)
        system_noise_th = np.random.normal(0, self.opti.value(self.angle_noise))
        system_noise = np.append(system_noise_xy, system_noise_th)

        next_state = mtimes(self.A, x) + mtimes(self.dT,vertcat(u[0]*cos(x[2]), u[0]*sin(x[2]), u[1])) + system_noise
        return next_state

    def update_1(self, x, u):
        return self.next_state_nominal(x, u)

    def update_2(self, x, u, k):

        if self.first_contact == False:
            return self.update_3(x, u, k)
        else:

            # obtain the current system noise covariance matrix of robot 1
            system_noise_cov = self.r1_pos_cov[:, k]

            # obtain the current robot 1 position
            x_prev_r1 = x

            # propagate robot 1 position
            xHat_next_r1_noUpdate = self.next_state_nominal(x_prev_r1, u)

            # propagate the system noise covariance matrix of robot 1 from the RNN
            system_noise_cov_next_r1 = self.r1_pos_cov[:, k+1]
            P11_noUpdate = np.array([[self.opti.value(system_noise_cov_next_r1[0]),
                                      self.opti.value(system_noise_cov_next_r1[1])],
                                     [self.opti.value(system_noise_cov_next_r1[2]),
                                      self.opti.value(system_noise_cov_next_r1[3])]])

            # obtain robot 2 position and its covariance matrix from the RNN, note robot 2 position, covariance will not
            # be updated, the update for robot 2 will occur in the MPC script for robot 2 in the next time step
            xHat_next_r2_noUpdate = self.r2_traj[:, k+1]
            system_noise_cov_next_r2 = self.r2_pos_cov[:, k+1]
            P22_noUpdate = np.array([[self.opti.value(system_noise_cov_next_r2[0]),
                                      self.opti.value(system_noise_cov_next_r2[1])],
                                     [self.opti.value(system_noise_cov_next_r2[2]),
                                      self.opti.value(system_noise_cov_next_r2[3])]])

            # TODO: x_next_r1 needs to equal the received measurements from the sensors
            # calculate x_next_r1 (this is used for calculating our measurements)
            x_next_r1 = self.next_state_withSystemNoise(x_prev_r1, u, system_noise_cov)

            # TODO: x_next_r2 needs to equal the received measurements from the sensors
            # calculate x_next_r2
            x_next_r2 = xHat_next_r2_noUpdate

            # take measurement
            z = x_next_r1 - x_next_r2

            # obtain the relative measurement uncertainty (based on communication uncertainty)
            R12 = self.relative_measurement_noise_cov

            # TODO: the self.P21 term must come from robot 2 (CHANGE in the future)
            # calculate the S matrix
            P21 = self.P12.T
            S = P11_noUpdate - self.P12 - P21 + P22_noUpdate + R12

            # calculate the inverse S matrix, if not possible, assume zeros
            try:
                S_inv = np.linalg.inv(S)

            except np.linalg.LinAlgError as err:
                if 'Singular matrix' in str(err):
                    S_inv = np.zeros((2, 2))
                else:
                    S_inv = np.zeros((2, 2))

            # calculate the kalman gain K
            K = mtimes(P11_noUpdate - self.P12, S_inv)

            # update x_hat of robot 1
            xHat_next_r1_update = xHat_next_r1_noUpdate[0:2] + mtimes(K, (
                        z[0:2] - (xHat_next_r1_noUpdate[0:2] - xHat_next_r2_noUpdate[0:2])))
            xHat_next_r1_update = vertcat(xHat_next_r1_update, x_next_r1[2])

            # update the covariance system noise matrix of robot 1 with the updated matrix
            P11_update = P11_noUpdate - mtimes(mtimes((P11_noUpdate - self.P12), S_inv), P11_noUpdate - P21)

            # update the covariance system noise matrix for robot 1 and 2
            self.P12 = mtimes(mtimes(P11_noUpdate, S_inv), P22_noUpdate)

            self.opti.set_value(self.r1_pos_cov[0, k + 1], P11_update[0])
            self.opti.set_value(self.r1_pos_cov[1, k + 1], P11_update[1])
            self.opti.set_value(self.r1_pos_cov[2, k + 1], P11_update[2])
            self.opti.set_value(self.r1_pos_cov[3, k + 1], P11_update[3])

        return xHat_next_r1_update

    def update_3(self, x, u, k):

        # obtain the current system noise covariance matrix of robot 1
        system_noise_cov = self.r1_pos_cov[:, k]

        # obtain the current robot 1 position
        x_prev_r1 = x

        # propagate robot 1 position, considering the effects of noise
        xHat_next_r1_noUpdate = self.next_state_nominal(x_prev_r1, u)

        # propagate the system noise covariance matrix of robot 1 from the RNN
        system_noise_cov_next_r1 = self.r1_pos_cov[:, k+1]
        P11_noUpdate = np.array([[self.opti.value(system_noise_cov_next_r1[0]),
                                  self.opti.value(system_noise_cov_next_r1[1])],
                                 [self.opti.value(system_noise_cov_next_r1[2]),
                                  self.opti.value(system_noise_cov_next_r1[3])]])

        # obtain robot 2 position and its covariance matrix from the RNN, note robot 2 position, covariance will not
        # be updated, the update for robot 2 will occur in the MPC script for robot 2 in the next time step
        xHat_next_r2_noUpdate = self.r2_traj[:, k+1]
        system_noise_cov_next_r2 = self.r2_pos_cov[:, k+1]
        P22_noUpdate = np.array([[self.opti.value(system_noise_cov_next_r2[0]),
                                  self.opti.value(system_noise_cov_next_r2[1])],
                                 [self.opti.value(system_noise_cov_next_r2[2]),
                                  self.opti.value(system_noise_cov_next_r2[3])]])

        # calculate x_next_r1 (this is used for calculating our measurements)
        # TODO: x_next_r1 needs to equal the received measurements from the sensors
        x_next_r1 = self.next_state_withSystemNoise(x_prev_r1, u, system_noise_cov)

        # TODO: x_next_r2 needs to equal the received measurements from the sensors
        # calculate x_next_r2
        x_next_r2 = xHat_next_r2_noUpdate

        # take measurement
        z = x_next_r1 - x_next_r2

        # obtain the relative measurement uncertainty (based on communication uncertainty)
        R12 =  self.relative_measurement_noise_cov

        # calculate the S matrix
        S = P11_noUpdate + P22_noUpdate + R12

        # calculate the inverse S matrix, if not possible, assume zeros
        try:
            S_inv = np.linalg.inv(S)

        except np.linalg.LinAlgError as err:
            if 'Singular matrix' in str(err):
                S_inv = np.zeros((2, 2))
            else:
                S_inv = np.zeros((2, 2))

        # calculate the kalman gain K
        K = mtimes(P11_noUpdate, S_inv)

        # update x_hat of robot 1
        xHat_next_r1_update = xHat_next_r1_noUpdate[0:2] + mtimes(K, (z[0:2] - (xHat_next_r1_noUpdate[0:2] -
                                                                                xHat_next_r2_noUpdate[0:2])))
        xHat_next_r1_update = vertcat(xHat_next_r1_update, x_next_r1[2])

        # update the covariance system noise matrix of robot 1 with the updated matrix
        P11_update = P11_noUpdate - mtimes((mtimes(P11_noUpdate, S_inv), P11_noUpdate))

        # update the covariance system noise matrix for robot 1 and 2
        self.P12 = mtimes(mtimes(P11_noUpdate, S_inv), P22_noUpdate)

        self.opti.set_value(self.r1_pos_cov[0, k+1], P11_update[0])
        self.opti.set_value(self.r1_pos_cov[1, k+1], P11_update[1])
        self.opti.set_value(self.r1_pos_cov[2, k+1], P11_update[2])
        self.opti.set_value(self.r1_pos_cov[3, k+1], P11_update[3])

        self.first_contact = True

        return xHat_next_r1_update

    def distance_pt_line(self, slope, a, intercept, point):
        A = -slope
        B = 1
        C = -intercept
        d = fabs(A*point[0] + B*point[1] + C) / sqrt(A**2 + B**2)
        a_slope = a[1]/a[0]

        x = (a_slope * point[0] - point[1] + intercept) / (a_slope - slope)
        y = slope * x + intercept
        dist = if_else(logic_and(sign(y - point[1]) == sign(a[1]), sign(x - point[0]) == sign(a[0])), -1*d, d)

        return dist
    """
    def distance_pt_line_check(self, slope, intercept, point):
        A = -slope
        B = 1
        C = -intercept
        dist = np.abs(A*point[0] + B*point[1] + C) / np.sqrt(A**2 + B**2)
        return dist
    """
    def distance_pt_line_check(self, p, a, b):
        # normalized tangent vector
        d = np.divide(b - a, np.linalg.norm(b - a))

        # signed parallel distance components
        s = np.dot(a - p, d)
        t = np.dot(p - b, d)

        # clamped parallel distance
        h = np.maximum.reduce([s, t, 0])

        # perpendicular distance component
        c = np.cross(p - a, d)

        return np.hypot(h, np.linalg.norm(c))

    def distance_pt_circle(self, center, point, obs_size, robot_size):
        # calculates the distance between the outerboundary of an obstacle and the outerboundary of the robot
        dist = np.sqrt((point[0] - center[0]) ** 2 + (point[1] - center[1]) ** 2) - robot_size - obs_size
        return dist

    def animate(self, curr_pos):
        plt.cla()
        plt.xlim(-8, 8)
        plt.ylim(-8, 8)
        self.ax.set_zlim(0, 10)
        # graph robot as a round sphere for simplicity
        self.ax.plot_surface(self.x_fig + curr_pos[0], self.y_fig + curr_pos[1], self.z_fig,
                             rstride=4, cstride=4, color='b')
        x_togo = 2 * np.cos(curr_pos[2])
        y_togo = 2 * np.sin(curr_pos[2])

        # graph direction of the robot heading
        self.ax.quiver(curr_pos[0], curr_pos[1], 0, x_togo, y_togo, 0, color='red', alpha=.8, lw=3)

        if self.obs:
            # graph polygon obstacles
            for i in range(0, len(self.x_list)):
                verts = [list(zip(self.x_list[i], self.y_list[i], self.z_list[i]))]
                self.ax.add_collection3d(Poly3DCollection(verts))

        # graph circle obstacles
        # Draw a circle on the z=0
            for i in range(1, len(self.obs) + 1):
                if self.obs[i]['polygon_type'] == 1:
                    center = self.obs[i]['vertices'][0]
                    size = self.obs[i]['size']
                    q = Circle((center[0], center[1]), size, color='green')
                    self.ax.add_patch(q)
                    art3d.pathpatch_2d_to_3d(q, z=0, zdir="z")
                    #height = np.linspace(0, 8, num=100)
                    #for j in range(0, len(height)):
                    #    q = Circle((center[0], center[1]), size, color='green')
                    #    self.ax.add_patch(q)
                    #    art3d.pathpatch_2d_to_3d(q, z=height[j], zdir="z")

"""
if __name__ == '__main__':
    # initialize all required variables for the SMPC solver
    dT = 0.5
    mpc_horizon = 2
    curr_pos = np.array([0, -5, 0]).reshape(3,1)
    goal_points = [[7, 0, 0]]
    robot_size = 0.5
    lb_state = np.array([[-8], [-8], [-2*pi]], dtype=float)
    ub_state = np.array([[8], [8], [2*pi]], dtype=float)
    lb_control = np.array([[-1.5], [-np.pi/2]], dtype=float)
    ub_control = np.array([[1.5], [np.pi/2]], dtype=float)
    Q = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
    R_init = np.array([[1, 0, 0], [0, 1, 0] ,[0, 0, 0.001]])
    angle_noise_r1 = 0.0
    angle_noise_r2 = 0.0
    relative_measurement_noise_cov = np.array([[0.0,0], [0,0.0]])
    maxComm_distance = -10
    animate = True
    failure_count = 0
    # initialize obstacles to be seen in a dictionary format. If obstacle should be represented as a circle, the
    # 'vertices' is should be a single [[x,y]] point representing the center of the circle, with 'size' equal to the
    # radius of the circle, and polygon_type: 1.
    obs = {1: {'vertices': [[-3.01, -1,0], [-3.02, 1.03,0], [3,1,0], [3.02, -1.05,0]], 'a': [], 'slopes': [], 'intercepts': [],
               'polygon_type': 4, 'risk': 0.1}}
    #obs.update(
    #    {2: {'vertices': [[6, 5,0], [7, 7,0], [8, 5.2,0]], 'a': [], 'slopes': [], 'intercepts': [], 'polygon_type': 3,
    #         'risk': 0.4}})
    #obs.update(
    #    {3: {'vertices': [[4, 4.1,0]], 'size': 0.7, 'polygon_type': 1, 'risk': 0.4}})

    SMPC = SMPC_UGV_Planner(dT, mpc_horizon, curr_pos, robot_size, lb_state,
                            ub_state, lb_control, ub_control, Q, R_init, angle_noise_r1, angle_noise_r2,
                            relative_measurement_noise_cov, maxComm_distance, obs, animate)

    ROS = ROSInterface(True)
    rospy.init_node('ros_interface')
    rate = rospy.Rate(10)

    for i in range(0, len(goal_points)):
        goal_pos = np.array(goal_points[i])
        SMPC.opti.set_value(SMPC.r1_goal, goal_pos)
        while m.sqrt((curr_pos[0] - goal_pos[0]) ** 2 + (curr_pos[1] - goal_pos[1]) ** 2) > 0.5:
            try:
                sol = SMPC.opti.solve()
                u = sol.value(SMPC.U[:, SMPC.N-1])
                ROS.send_velocity(u)
                curr_pos = ROS.get_current_pose()
                curr_pos = np.array(curr_pos).reshape(3, 1)
                SMPC.check_obstacles(np.concatenate((curr_pos[0], curr_pos[1], [0])))
            except:
                failure_count += 1
                u = sol.value(SMPC.U[:, 0])
                u[1] = 0
                ROS.send_velocity(u)
                curr_pos = ROS.get_current_pose()
                curr_pos = np.array(curr_pos).reshape(3,1)
                print('WARNING: Solver has failed, using previous control value for next input')
                SMPC.check_obstacles(np.concatenate((curr_pos[0], curr_pos[1], [0])))
            SMPC.opti.set_value(SMPC.r1_pos, curr_pos)
            SMPC.animate(curr_pos)
            rate.sleep()
"""