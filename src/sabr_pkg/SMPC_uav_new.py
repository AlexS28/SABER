
"""FILE CREATED BY: Alexander Schperberg, aschperb@gmail.com
Copyright by RoMeLa (Robotics and Mechanisms Laboratory, University of California, Los Angeles)"""

# This file provides a stochastic model predictive controller for a unmanned aerial vehicle for data collection
# purposes that moves a aerial to several user-specified goal locations. Currently, for the UAV, only circular
# obstacles are considered. For the UGV, both circular and convex polygon obstacles are considered.

from casadi import *
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import mpl_toolkits.mplot3d.art3d as art3d
from matplotlib.patches import Circle
import matplotlib.pyplot as plt
import math as m
import control
#from ROS_interface import *

class SMPC_UAV_Planner():

    def __init__(self, dT, mpc_horizon, curr_pos, robot_size, lb_state,
                 ub_state, lb_control, ub_control, Q, R, relative_measurement_noise_cov,
                 maxComm_distance, obs, animate, multi_agent):

        # initialize Optistack class
        self.opti = casadi.Opti()
        # dt = discretized time difference
        self.dT = dT
        # mpc_horizon = number of time steps for the mpc to look ahead
        self.N = mpc_horizon
        # robot_size = input a radius value, where the corresponding circle represents the size of the robot
        self.robot_size = robot_size
        # if the uav is to be animated along the same graph as the ugv, then put multi_agent to true
        self.multi_agent = multi_agent
        # lower_bound_state = numpy array corresponding to the lower limit of the robot states, e.g.
        # lb_state = np.array([[-20], [-20], [-pi], dtype=float), the same for the upper limit (ub). Similar symbolic
        # representation for the controls (lb_control and ub_control) as well
        self.lb_state = lb_state
        self.ub_state = ub_state
        self.lb_control = lb_control
        self.ub_control = ub_control
        # initialize obstacles
        self.obs = obs
        # distance to obstacle to be used as constraints
        self.max_obs_distance = 10
        # Q and R diagonal matrices, used for the MPC objective function, Q is 3x3, R is 4x4 (first 2 diagonals
        # represent the cost on linear and angular velocity, the next 2 diagonals represent cost on state slack,
        # and terminal slack respectively. The P diagonal matrix represents the cost on the terminal constraint.
        self.relative_measurement_noise_cov = relative_measurement_noise_cov
        # initialize the maximum distance that robot 1 and 2 are allowed to have for cross communication
        self.maxComm_distance = maxComm_distance
        # initialize cross diagonal system noise covariance matrix
        self.P12 = np.array([[0, 0], [0, 0]])
        # bool variable to indicate whether the robot has made first contact with the uav
        self.first_contact = False
        self.Q = Q
        self.R = R
        # initialize cost on slack
        self.slack_cost = 10000
        # initialize discretized state matrices A and B
        A1 = 0.7969
        A2 = 0.02247
        A3 = -1.7976
        A4 = 0.9767
        B1 = 0.01166
        B2 = 0.9921
        g = 9.81
        KT = 0.91
        m = .001 #originally 1.42

        self.A = np.array([[1, self.dT, (g*self.dT**2)/2, 0, 0, 0, 0, 0, 0, 0],
                          [0, 1, g*self.dT, 0, 0, 0, 0, 0, 0, 0],
                          [0, 0, A1, A2, 0, 0, 0, 0, 0, 0],
                          [0, 0, A3, A4, 0, 0, 0, 0, 0, 0],
                           [0, 0, 0, 0, 1, self.dT, (g*self.dT**2)/2, 0, 0, 0],
                           [0,0,0,0,0,1,g*self.dT,0,0,0],
                           [0,0,0,0,0,0,A1,A2,0,0],
                           [0,0,0,0,0,0,A3,A4,0,0],
                           [0,0,0,0,0,0,0,0,1,self.dT],
                           [0,0,0,0,0,0,0,0,0,1]])

        self.B = np.array([[0,0,0], [0,0,0], [B1,0,0], [B2,0,0], [0,0,0],
                           [0,0,0], [0,B1,0],[0,B2,0],[0,0,(-KT*self.dT**2)/(m*2)],
                           [0,0,(-KT*self.dT)/m]])
        self.P, _, _ = control.dare(self.A, self.B, self.Q, self.R)
        # TODO: adding the C matrix does not work, potentially a problem with equations
        #self.C = np.array([[0],[0],[0],[0],[0],[0],[0],[0],[0.0031], [0.2453]])
        # Current covariance of robot
        self.r1_cov_curr = np.array([[0.01 + self.robot_size, 0], [0, 0.01 + self.robot_size]])
        # initialize robot's current position
        self.curr_pos = curr_pos
        # initialize measurement noise (in our calculation, measurement noise is set by the user and is gaussian,
        # zero-mean). It largely represents the noise due to communication transmitters, or other sensor devices. It
        # is assumed to be a 3x3 matrix (x, y, and theta) for both robots
        self.relative_measurement_noise_cov = relative_measurement_noise_cov

        self.initVariables()
        # initialize parameters for animation
        if animate:
            if not self.multi_agent:
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
        self.X = self.opti.variable(10, self.N+1)
        self.x_pos = self.X[0,:]
        self.x_vel = self.X[1,:]
        self.th1 = self.X[2,:]
        self.th1_vel = self.X[3,:]
        self.y_pos = self.X[4,:]
        self.y_vel = self.X[5,:]
        self.th2 = self.X[6,:]
        self.th2_vel = self.X[7,:]
        self.z_pos = self.X[8,:]
        self.z_vel = self.X[9,:]

        # initialize, linear and angular velocity control variables (v, and w), and repeat above procedure
        self.U = self.opti.variable(3, self.N)
        self.u1 = self.U[0, :]
        self.u2 = self.U[1, :]
        self.u3 = self.U[2, :]

        # initalize slack variable
        self.slack = self.opti.variable(1, self.N+1)

        # initialize the current robot pos
        self.r_pos = self.opti.parameter(10, 1)
        self.opti.set_value(self.r_pos, self.curr_pos)

        # initialize the goal point
        self.r_goal = self.opti.parameter(10, 1)

        # initialize the uncertainty covariances from the RNN, provided by robot 1 (4 x 1 vector per covariance matrix)
        # must be positive semi-definite, from t+1 to N
        self.r1_pos_cov = self.opti.parameter(4, self.N + 1)
        self.opti.set_value(self.r1_pos_cov, 0)

        # initialize the uncertainty covariances from the RNN, provided by robot 2 (4 x 1 vector per covariance matrix)
        # must be positive semi-definite, from t+1 to N
        self.r2_pos_cov = self.opti.parameter(4, self.N + 1)
        self.opti.set_value(self.r2_pos_cov, 0)

        # initialize robot 2, future positions (x, y), from t+1 to N
        self.r2_traj = self.opti.parameter(2, self.N + 1)
        self.opti.set_value(self.r2_traj, 0)

        # initialize the objective function
        self.obj()

    def obj(self):
        self.objFunc = 0
        for k in range(0, self.N-1):
            con = self.U[:, k]
            st = self.X[:, k + 1]

            self.objFunc = self.objFunc + mtimes(mtimes((st - self.r_goal).T, self.Q), st - self.r_goal) + \
                           mtimes(mtimes(con.T, self.R), con) + self.slack[:,k+1]*self.slack_cost

            st = self.X[:, self.N]

            self.objFunc = self.objFunc + mtimes(mtimes((st - self.r_goal).T, self.P), st - self.r_goal) + self.slack[:, self.N] * self.slack_cost

        self.init_constraints()

        # initialize the objective function into the solver
        self.opti.minimize(self.objFunc)

    def init_constraints(self):
        # constrain the current state, and bound current and future states by their limits
        self.opti.subject_to(self.X[:, 0] == self.r_pos)
        self.opti.subject_to(self.opti.bounded(self.lb_state, self.X, self.ub_state))
        self.opti.subject_to(self.opti.bounded(self.lb_control, self.U, self.ub_control))

        # constrain the slack variable
        self.opti.subject_to(self.slack >= 0)

        # initiate multiple shooting constraints
        for k in range(0, self.N):
            #next_state = self.next_state_nominal(self.X[:, k], self.U[:, k])
            #self.opti.subject_to(self.X[:, k + 1] == next_state)
            next_state = if_else((sqrt((self.X[0, k] - self.r2_traj[0, k])**2 +
                                      (self.X[4, k] - self.r2_traj[1, k]**2)) >= self.maxComm_distance),
                                 self.update_1(self.X[:,k], self.U[0:3,k]), self.update_2(self.X[:,k], self.U[0:3,k], k))
            self.opti.subject_to(self.X[:, k+1]==next_state)

        # initialize chance constraints
        if self.obs:
            self.chance_constraints()
        # set solver
        self.pre_solve()

    def chance_constraints(self):

        # Using chance constraints on circular obstacles
        self.obs_index = []
        for i in range(1, len(self.obs) + 1):
            if self.obs[i]['polygon_type'] == 1:
                self.obs_index.append(self.obs[i]['polygon_type'])

        self.cc = self.opti.parameter(sum(self.obs_index), self.N + 1)
        self.opti.set_value(self.cc, 1)

        self.switch_obs = self.opti.parameter(len(self.obs_index), 1)
        self.opti.set_value(self.switch_obs, 0)

        for i in range(1, len(self.obs) + 1):
            iter = 0
            if self.obs[i]['polygon_type'] == 1:
                a = np.array([1, 1]).reshape(2, 1)
                r = self.obs[i]['risk']
                center = self.obs[i]['vertices'][0]
                size = self.obs[i]['size']
                self.opti.set_value(self.cc[iter, :],
                                    np.sqrt(np.dot(np.dot(2 * np.transpose(a), self.r1_cov_curr), a)) * erfinv(
                                        (1 - 2 * r)))
                for k in range(0, self.N + 1):
                    dist = -1 * self.distance_pt_circle(center, [self.X[0, k], self.X[4, k]], size, self.robot_size)+self.cc[iter] - self.slack[0,k]
                    self.opti.subject_to(dist*self.switch_obs[iter] <= 0)
                iter += 1

    def check_obstacles(self, curr_pos):
        iter = 0
        for i in range(1, len(self.obs)+1):

            if self.obs[i]['polygon_type'] == 1:
                center = self.obs[i]['vertices'][0]
                size = self.obs[i]['size']
                dist = self.distance_pt_circle(center, curr_pos, size, self.robot_size)
                if dist <= self.max_obs_distance:
                    a = np.array([1, 1]).reshape(2, 1)
                    r = self.obs[i]['risk']

                    for k in range(0, self.N + 1):
                        # self.r1_cov_curr[0,:] = self.r1_pos_cov[0:2,k]
                        # self.r1_cov_curr[1, :] = self.r1_pos_cov[2:4,k]
                        self.opti.set_value(self.cc[iter, k],
                                    np.sqrt(np.dot(np.dot(2 * np.transpose(a), self.r1_cov_curr), a)) * erfinv(
                                        (1 - 2 * r)))
                        self.opti.set_value(self.cc[iter,k], 0)
                        self.opti.set_value(self.switch_obs[iter], 1)
                        iter += 1
                        break
                else:
                    self.opti.set_value(self.switch_obs[iter], 0)
                    iter += 1
                    break

    def next_state_nominal(self, x, u):
        next_state = mtimes(self.A, x) + mtimes(self.B, u) #+ self.C
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
        #system_noise_th = np.random.normal(0, self.opti.value(self.angle_noise))
        #system_noise = np.append(system_noise_xy, system_noise_th)
        system_noise = system_noise_xy
        next_state = mtimes(self.A, x) + mtimes(self.B, u)
        next_state[0] = next_state[0] + system_noise[0]
        next_state[4] = next_state[4] + system_noise[1]
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

            # only x and y are updated for simplicity (eventually may want to include z and also angles)
            xHat_next_r1_noUpdate_f = []
            xHat_next_r1_noUpdate_f = vertcat(xHat_next_r1_noUpdate_f, xHat_next_r1_noUpdate[0])
            xHat_next_r1_noUpdate_f = vertcat(xHat_next_r1_noUpdate_f, xHat_next_r1_noUpdate[1])

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
            x_next_r1_f = []
            x_next_r1_f = vertcat(x_next_r1_f,x_next_r1[0])
            x_next_r1_f = vertcat(x_next_r1_f,x_next_r1[4])

            # TODO: x_next_r2 needs to equal the received measurements from the sensors
            # calculate x_next_r2
            x_next_r2 = xHat_next_r2_noUpdate

            # take measurement
            z = x_next_r1_f - x_next_r2

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
            xHat_next_r1_update_f = xHat_next_r1_noUpdate_f[0:2] + mtimes(K, (
                        z[0:2] - (xHat_next_r1_noUpdate_f[0:2] - xHat_next_r2_noUpdate[0:2])))


            # update the covariance system noise matrix of robot 1 with the updated matrix
            P11_update = P11_noUpdate - mtimes(mtimes((P11_noUpdate - self.P12), S_inv), P11_noUpdate - P21)

            # update the covariance system noise matrix for robot 1 and 2
            self.P12 = mtimes(mtimes(P11_noUpdate, S_inv), P22_noUpdate)

            self.opti.set_value(self.r1_pos_cov[0, k + 1], P11_update[0])
            self.opti.set_value(self.r1_pos_cov[1, k + 1], P11_update[1])
            self.opti.set_value(self.r1_pos_cov[2, k + 1], P11_update[2])
            self.opti.set_value(self.r1_pos_cov[3, k + 1], P11_update[3])

            xHat_next_r1_update = xHat_next_r1_noUpdate
            xHat_next_r1_update[0] = xHat_next_r1_update_f[0]
            xHat_next_r1_update[4] = xHat_next_r1_update_f[1]

        return xHat_next_r1_update

    def update_3(self, x, u, k):

        # obtain the current system noise covariance matrix of robot 1
        system_noise_cov = self.r1_pos_cov[:, k]

        # obtain the current robot 1 position
        x_prev_r1 = x

        # propagate robot 1 position, considering the effects of noise
        xHat_next_r1_noUpdate = self.next_state_nominal(x_prev_r1, u)

        # only x and y are updated for simplicity (eventually may want to include z and also angles)
        xHat_next_r1_noUpdate_f = []
        xHat_next_r1_noUpdate_f = vertcat(xHat_next_r1_noUpdate_f, xHat_next_r1_noUpdate[0])
        xHat_next_r1_noUpdate_f = vertcat(xHat_next_r1_noUpdate_f, xHat_next_r1_noUpdate[1])

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
        x_next_r1_f = []
        x_next_r1_f = vertcat(x_next_r1_f, x_next_r1[0])
        x_next_r1_f = vertcat(x_next_r1_f, x_next_r1[4])

        # TODO: x_next_r2 needs to equal the received measurements from the sensors
        # calculate x_next_r2
        x_next_r2 = xHat_next_r2_noUpdate

        # take measurement
        z = x_next_r1_f - x_next_r2

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
        xHat_next_r1_update_f = xHat_next_r1_noUpdate_f[0:2] + mtimes(K, (
                z[0:2] - (xHat_next_r1_noUpdate_f[0:2] - xHat_next_r2_noUpdate[0:2])))

        # update the covariance system noise matrix of robot 1 with the updated matrix
        P11_update = P11_noUpdate - mtimes((mtimes(P11_noUpdate, S_inv), P11_noUpdate))

        # update the covariance system noise matrix for robot 1 and 2
        self.P12 = mtimes(mtimes(P11_noUpdate, S_inv), P22_noUpdate)

        self.opti.set_value(self.r1_pos_cov[0, k+1], P11_update[0])
        self.opti.set_value(self.r1_pos_cov[1, k+1], P11_update[1])
        self.opti.set_value(self.r1_pos_cov[2, k+1], P11_update[2])
        self.opti.set_value(self.r1_pos_cov[3, k+1], P11_update[3])

        self.first_contact = True
        xHat_next_r1_update = xHat_next_r1_noUpdate
        xHat_next_r1_update[0] = xHat_next_r1_update_f[0]
        xHat_next_r1_update[4] = xHat_next_r1_update_f[1]

        return xHat_next_r1_update

    def pre_solve(self):
        # For the UAV only circular obstacles are considered, see SMPC_ugv.py code for how polygons can be considered

        opts = {'bonmin.warm_start': 'interior_point',
                'bonmin.acceptable_obj_change_tol': 1, 'bonmin.acceptable_tol': 1}
        # create the solver
        self.opti.solver('bonmin', opts)


        # set solver options for ipopt (nonlinear programming)
        #opts = {'ipopt': {'max_iter': 1000, 'print_level': 0, 'acceptable_tol': 1,
        #              'acceptable_obj_change_tol': 1}}
        #opts.update({'print_time': 0})
        # create solver
        #self.opti.solver('ipopt', opts)

    def distance_pt_circle(self, center, point, obs_size, robot_size):
        # calculates the distance between the outerboundary of an obstacle and the outerboundary of the robot
        dist = np.sqrt((point[0] - center[0]) ** 2 + (point[1] - center[1]) ** 2) - robot_size - obs_size
        return dist

    def animate(self, curr_pos):
        plt.cla()
        plt.xlim(0, 10)
        plt.ylim(0, 10)
        self.ax.set_zlim(0, 10)
        # graph robot as a round sphere for simplicity
        self.ax.plot_surface(self.x_fig + curr_pos[0], self.y_fig + curr_pos[4], self.z_fig + curr_pos[8],
                             rstride=4, cstride=4, color='b')

        for i in range(1, len(self.obs) + 1):
            if self.obs[i]['polygon_type'] == 1:
                center = self.obs[i]['vertices'][0]
                size = self.obs[i]['size']
                q = Circle((center[0], center[1]), size, color='red')
                self.ax.add_patch(q)
                art3d.pathpatch_2d_to_3d(q, z=0, zdir="z")
        if self.animate:
            plt.pause(0.001)

    def animate_multi_agents(self, ax, curr_pos):
        ax.set_zlim(0, 10)
        # graph robot as a round sphere for simplicity
        ax.plot_surface(self.x_fig + curr_pos[0], self.y_fig + curr_pos[4], self.z_fig + curr_pos[8],
                             rstride=4, cstride=4, color='b')
        x_togo = 2 * np.cos(0)
        y_togo = 2 * np.sin(0)

        x_togo2 = 2 * np.cos(pi/2)
        y_togo2 = 2 * np.sin(pi/2)

        x_togo3 = 2 * np.cos(pi)
        y_togo3 = 2 * np.sin(pi)

        x_togo4 = 2 * np.cos(3*pi/2)
        y_togo4 = 2 * np.sin(3*pi/2)

        # graph direction of the robot heading
        ax.quiver(curr_pos[0], curr_pos[4], curr_pos[8], x_togo, y_togo, 0, color='red', alpha=.8, lw=2)
        ax.quiver(curr_pos[0], curr_pos[4], curr_pos[8], x_togo2, y_togo2, 0, color='red', alpha=.8, lw=2)
        ax.quiver(curr_pos[0], curr_pos[4], curr_pos[8], x_togo3, y_togo3, 0, color='red', alpha=.8, lw=2)
        ax.quiver(curr_pos[0], curr_pos[4], curr_pos[8], x_togo4, y_togo4, 0, color='red', alpha=.8, lw=2)



if __name__ == '__main__':
    # initialize all required variables for the SMPC solver
    dT = 0.5
    mpc_horizon = 10
    # x, vel_x, th1, th1_vel, y, vel_y, th2, th2_vel, z, z_vel
    curr_pos = np.array([0,0,0,0,0,0,0,0,0,0]).reshape(10,1)
    goal_points = [[0,0,0,0,0,0,0,0,5,0], [10,0,0,0,10,0,0,0,5,0], [0,0,0,0,10,0,0,0,5,0], [0, 0,0, 0,0,0 ,0,0,5,0]]
    robot_size = 0.5
    #lb_state = np.array([[-20], [-5], [-10*(pi/180)], [-50*pi/180],[-20], [-5], [-10*(pi/180)], [-50*pi/180],[-20],[-5]], dtype=float)
    #ub_state = np.array([[20], [5], [10*(pi/180)], [50*pi/180],[20], [5], [10*(pi/180)], [50*pi/180],[20],[5]], dtype=float)
    #lb_control = np.array([[-10*pi/180], [-10*pi/180], [-10*pi/180]], dtype=float)
    #ub_control = np.array([[10*pi/180], [10*pi/180], [10*pi/180]], dtype=float)
    # only constraints on the states will be the velocity (this provided the most stability)
    vel_limit = 1.5
    lb_state = np.array(
        [[-10**10], [-vel_limit], [-10**10], [-10**10], [-10**10], [-vel_limit], [-10**10], [-10**10], [-10**10],
         [-vel_limit]], dtype=float)
    ub_state = np.array(
        [[10**10], [vel_limit], [10**10], [10**10], [10**10], [vel_limit], [10**10], [10**10], [10**10], [vel_limit]],
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

    obs = {1: {'vertices': [[3.9, 4], [4, 6], [6, 6.1], [5.9, 4.1]], 'a': [], 'slopes': [], 'intercepts': [],
               'polygon_type': 4, 'risk': 0.1}}
    obs.update(
        {2: {'vertices': [[6, 5], [7, 7], [8, 5.2]], 'a': [], 'slopes': [], 'intercepts': [], 'polygon_type': 3,
             'risk': 0.4}})
    obs.update(
        {3: {'vertices': [[4, 1]], 'size': 0.9, 'polygon_type': 1, 'risk': 0.3}})
    obs.update(
        {4: {'vertices': [[3, 4]], 'size': 0.9, 'polygon_type': 1, 'risk': 0.3}})
    obs.update(
        {5: {'vertices': [[8, 4]], 'size': 0.9, 'polygon_type': 1, 'risk': 0.3}})
    obs.update(
        {6: {'vertices': [[3, 9]], 'size': 0.9, 'polygon_type': 1, 'risk': 0.3}})

    obs.update(
        {7: {'vertices': [[3, 10]], 'size': 0.9, 'polygon_type': 1, 'risk': 0.3}})

    obs.update(
        {8: {'vertices': [[7, 7]], 'size': 0.9, 'polygon_type': 1, 'risk': 0.3}})
    #SMPC = SMPC_UAV_Planner(dT, mpc_horizon, curr_pos, lb_state,
    #                        ub_state, lb_control, ub_control, Q, R, robot_size, obs, animate, multi_agent=False)
    #ROS = ROSInterface()
    #rospy.init_node('ros_interface')
    #rate = rospy.Rate(10)
    #ROS.send_velocityUAV([0, 0, 0, 0, 0])
    #curr_posROS = ROS.get_current_poseUAV()
    #curr_pos[0] = curr_posROS[0]
    #curr_pos[4] = curr_posROS[1]
    #curr_pos[8] = curr_posROS[2]
    animate = True
    relative_measurement_noise_cov = np.array([[0.0,0], [0,0.0]])
    maxComm_distance = -10

    SMPC = SMPC_UAV_Planner(dT,mpc_horizon,curr_pos,robot_size,lb_state,ub_state,lb_control,ub_control,Q,R,relative_measurement_noise_cov,maxComm_distance,obs,animate,multi_agent=False)

    for i in range(0, len(goal_points)):
        goal_pos = np.array(goal_points[i])
        SMPC.opti.set_value(SMPC.r_goal, goal_pos)

        while m.sqrt((curr_pos[0] - goal_pos[0]) ** 2 + (curr_pos[4] - goal_pos[4]) ** 2 + (curr_pos[8] - goal_pos[8])**2) > 0.5: #and not rospy.is_shutdown():
            sol = SMPC.opti.solve()
            x = sol.value(SMPC.X)[:, 1]
            x_vel = sol.value(SMPC.X[:, SMPC.N])
            curr_pos = np.array(x).reshape(10, 1)
            curr_pos2 = np.array(x_vel).reshape(10, 1)

            #ROS.send_velocityUAV([curr_pos[1], curr_pos[5], curr_pos[9], curr_pos[3], curr_pos[7]])
            #curr_posROS = ROS.get_current_poseUAV()
            #curr_pos[0] = curr_posROS[0]
            #curr_pos[4] = curr_posROS[1]
            #curr_pos[8] = curr_posROS[2]

            SMPC.opti.set_value(SMPC.r_pos, curr_pos)
            SMPC.check_obstacles([curr_pos[0], curr_pos[4]])
            SMPC.animate(curr_pos)
            #rate.sleep()
