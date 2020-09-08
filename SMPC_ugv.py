"""FILE CREATED BY: Alexander Schperberg, aschperb@gmail.com
Copyright by RoMeLa (Robotics and Mechanisms Laboratory, University of California, Los Angeles)"""

# This file provides a stochastic and robust model predictive controller for a simple unmanned ground vehicle that
# moves a ground vehicle to any desired goal location, while considering obstacles (represented as 2D polygons) and
# cross communication ability with another robot


from casadi import *
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import matplotlib.pyplot as plt
import math as m
import control
from scipy.stats import linregress
from scipy import special

class SMPC_UGV_Planner():

    def __init__(self, dT, mpc_horizon, curr_pos, goal_pos, robot_size, max_nObstacles, field_of_view, lb_state,
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
        # max_nObstacles = total number of obstacles the mpc constraints are allowed to use in its calculations
        self.max_nObstacles = max_nObstacles
        # view_distance = how far and wide the robot's sensor is allowed to see its surrounding obstacles,
        # an example input is the following: field_of_view = {'max_distance': 10.0, 'angle_range': [45, 135]}
        self.field_of_view = field_of_view
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
        self.R = R
        # initialize discretized state matrices A and B (note, A is constant, but B will change as it is a function of
        # state theta)
        self.A = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
        self.B = np.array([[self.dT, 0, 0], [0, self.dT, 0], [0, 0, self.dT]])
        # initialize the P matrix, which is the cost matrix that defines the optimal state feedback controller
        _, self.P, _ = control.lqr(self.A, self.B, self.Q, self.R)

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
        # initialize obstacles
        self.obs = obs
        # initialize robot's current position
        self.curr_pos = curr_pos
        # initialize robot's goal position
        self.goal_pos = goal_pos
        #self.change_goal_point(goal_pos)
        # initialize the current positional uncertainty (and add the robot size to it)
        # TODO: this is a temporary fix for testing
        self.r1_cov_curr = np.array([[0.1+self.robot_size, 0], [0, 0.1+self.robot_size]])
        # initialize cross diagonal system noise covariance matrix
        self.P12 = np.array([[0,0], [0,0]])
        # bool variable to indicate whether the robot has made first contact with the uav
        self.first_contact = False
        # initialize state, control, and slack variables
        self.initVariables()
        # initialize objective function
        self.obj()
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
        self.U = self.opti.variable(3, self.N)
        self.vx = self.U[0,:]
        self.vy = self.U[1, :]
        self.w = self.U[2,:]
        self.opti.set_initial(self.U, 0)

        # initialize the slack variables, one for the chance constraints, and the second rotation
        self.slack = self.opti.variable(1, self.N + 1)

        # initialize the current robot pos (x, y and th current position)
        self.r1_pos = self.opti.parameter(3, 1)
        self.opti.set_value(self.r1_pos, self.curr_pos)

        # initialize the goal robot pos (x, y, and th goal position)
        self.r1_goal = self.opti.parameter(3, 1)
        self.opti.set_value(self.r1_goal, self.goal_pos)

        # initialize the angle noise for warm start MPC
        self.angle_noise = self.opti.parameter(1, 1)
        self.opti.set_value(self.angle_noise, 0)

        # initialize the uncertainty covariances from the RNN, provided by robot 1 (4 x 1 vector per covariance matrix)
        # must be positive semi-definite, from t to N
        self.r1_pos_cov = self.opti.parameter(4, self.N)
        self.opti.set_value(self.r1_pos_cov, 0)

        # initialize robot 2, future positions (x, y, and th), from t+1 to N
        self.r2_traj = self.opti.parameter(3, self.N)
        self.opti.set_value(self.r2_traj, 0)

        # initialize the uncertainty covariances from the RNN, provided by robot 2 (4 x 1 vector per covariance matrix)
        # must be positive semi-definite, from t to N
        self.r2_pos_cov = self.opti.parameter(4, self.N)
        self.opti.set_value(self.r2_pos_cov, 0)

    # objective function, with consideration of a final terminal state
    def obj(self):
        self.objFunc = 0
        V = np.array([4000])

        for k in range(0, self.N-1):

            con = self.U[:, k]
            st = self.X[:, k+1]

            self.objFunc = self.objFunc + mtimes(mtimes((st - self.r1_goal).T, self.Q), st - self.r1_goal) + mtimes(
                mtimes(con.T, self.R), con) + mtimes(mtimes(self.slack[:,k+1].T, V), self.slack[:,k+1])

        st = self.X[:, self.N]
        self.objFunc = self.objFunc + mtimes(mtimes((st - self.r1_goal).T, self.P), st - self.r1_goal) + \
                       mtimes(mtimes(self.slack[:,self.N].T, V), self.slack[:,self.N])

        # initialize the constraints for the objective function
        self.init_constraints()

        # initialize the objective function into the solver
        self.opti.minimize(self.objFunc)

        # initiate the solver
        self.pre_solve()

    def pre_solve(self):
        # initiate the solver called bonmin - performs Mixed-Integer Nonlinear Programming (MINLP)

        # ensure states X, and controls U are continues, while I variables are integers
        OT_Boolvector_X = [0]*self.X.size()[0]*self.X.size()[1]
        OT_Boolvector_U = [0]*self.U.size()[0]*self.U.size()[1]
        OT_Boolvector_slack = [0] * self.slack.size()[0] * self.slack.size()[1]
        OT_Boolvector_Int = [1]*self.I.size()[0]*self.I.size()[1]
        OT_Boolvector = OT_Boolvector_X + OT_Boolvector_U + OT_Boolvector_slack + OT_Boolvector_Int

        opts = {'bonmin.warm_start': 'interior_point', 'discrete': OT_Boolvector, 'error_on_fail': True,
            }

        # create the solver
        self.opti.solver('bonmin', opts)

    # the nominal next state is calculated for use as a terminal constraint in the objective function
    def next_state_nominal(self, x, u):
        next_state = mtimes(self.A, x) + mtimes(self.B, u)
        return next_state

    # the next state is calculated with consideration of system noise, also considered the true state
    def next_state_withSystemNoise(self, x, u, system_noise_cov):
        # the system_noise_covariance will be a flattened 1x4 array, provided by the output of an RNN. We need to
        # convert it into a 3x3 matrix. We will assume a constant noise in theta however.

        system_noise_cov_converted = np.array([[self.opti.value(system_noise_cov[0]),
                self.opti.value(system_noise_cov[1])], [self.opti.value(system_noise_cov[2]),
                                                        self.opti.value(system_noise_cov[3])]])


        # sample a gaussian distribution of the system_noise covariance (for x and y)
        system_noise_xy = np.random.multivariate_normal([0,0], system_noise_cov_converted, check_valid='warn').reshape(2,1)
        # sample a gaussian distribution of theta
        #system_noise_th = np.sqrt(angle_noise_r1)
        system_noise_th = np.random.normal(0, self.opti.value(self.angle_noise))
        system_noise = np.append(system_noise_xy, system_noise_th)

        next_state = mtimes(self.A, x) + mtimes(self.B, u) + system_noise
        return next_state

    # initialize all required constraints for the objective function
    def init_constraints(self):
        # constrain the current state

        self.opti.subject_to(self.opti.bounded(self.lb_state, self.X, self.ub_state))
        self.opti.subject_to(self.opti.bounded(self.lb_control, self.U, self.ub_control))
        self.opti.subject_to(self.opti.bounded([-inf], self.slack, [inf]))
        self.opti.subject_to(self.X[:, 0] == self.r1_pos)


        for k in range(0, self.N-1):
             next_state = if_else((sqrt((self.X[0, k] - self.r2_traj[0, k])**2 +
                                      (self.X[1, k] - self.r2_traj[1, k]**2)) >= self.maxComm_distance),
                                 self.update_1(self.X[:,k], self.U[0:3,k], k), self.update_2(self.X[:,k], self.U[0:3,k], k))

             self.opti.subject_to(self.X[:,k+1] == next_state)

        # include the terminal constraint
        next_state = self.next_state_nominal(self.X[:, self.N-1], self.U[0:3, self.N-1])
        self.opti.subject_to(self.X[:, self.N]  == next_state)

        # provide rotational constraints
        self.rotation_constraints()

        # initialize the obstacles to be used by chance constraints
        self.init_obstacles(self.obs, animate)

        # provide chance constraints
        #self.chance_constraints()

    def rotation_constraints(self):

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


    def update_1(self, x, u, k):
        system_noise_cov = self.r1_pos_cov[:, k]

        return self.next_state_withSystemNoise(x, u, system_noise_cov)

    def update_2(self, x, u, k):

        if self.first_contact == False:
            return self.update_3(x, u, k)
        else:


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
            x_next_r1 = self.next_state_withSystemNoise(x_prev_r1, u, system_noise_cov)

            # TODO: x_next_r2 needs to equal the dynamic equations of the quadrotor (CHANGE in the future)
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
        x_next_r1 = self.next_state_withSystemNoise(x_prev_r1, u, system_noise_cov)

        # TODO: x_next_r2 needs to equal the dynamic equations of the quadrotor (CHANGE in the future)
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


    def chance_constraints(self):
        """
        r = self.obs[1]['risk']
        a1 = self.obs[1]['a'][:, 0].reshape(2,1)
        b1 = self.obs[1]['intercepts'][0]

        a2 = self.obs[1]['a'][:, 1].reshape(2, 1)
        b2 = self.obs[1]['intercepts'][1]

        a3 = self.obs[1]['a'][:, 2].reshape(2, 1)
        b3 = self.obs[1]['intercepts'][2]


        # TODO: Replace self.r1_cov_curr with a the general covariance matrix in position from the RNN

        c1 = np.sqrt(np.dot(np.dot(2*np.transpose(a1), self.r1_cov_curr), a1)) * special.erfinv((1 - 2 * r))
        c2 = np.sqrt(np.dot(np.dot(2*np.transpose(a2), self.r1_cov_curr), a2)) * special.erfinv((1 - 2 * r))
        c3 = np.sqrt(np.dot(np.dot(2*np.transpose(a3), self.r1_cov_curr), a3)) * special.erfinv((1 - 2 * r))


        for i in range(0, self.N+1):

            self.opti.subject_to(((mtimes(np.transpose(a1), self.X[0:2, i]) - b1))*self.I[0] + self.slack[0, i] >= c1*self.I[0] - self.slack[0, i])
            self.opti.subject_to(((mtimes(np.transpose(a2), self.X[0:2, i]) - b2))*self.I[1] + self.slack[0, i] >= c2*self.I[1] - self.slack[0, i])
            self.opti.subject_to(((mtimes(np.transpose(a3), self.X[0:2, i]) - b3))*self.I[2] + self.slack[0, i] >= c3*self.I[2] - self.slack[0, i])
        """
        return 0

    def value_function(self):
        # get value from the objective function
        value_fun = self.opti.value(self.opti.f)
        return value_fun

    def init_obstacles(self, obstacles, animate):

        # receive the slope, intercepts, of the obstacles for chance constraints, and plot
        for i in range(1, len(obstacles)+1):
            it = 0
            slopes = []
            intercepts = []
            a_vectors = np.empty((2, len(obstacles[i]['vertices'])))

            for j in range(0, len(obstacles[i]['vertices'])-1):
                point_1 = obstacles[i]['vertices'][it]
                point_2 = obstacles[i]['vertices'][it+1]
                it += 1

                x = [point_1[0], point_2[0]]
                y = [point_1[1], point_2[1]]
                slope, intercept, _, _, _ = linregress(x, y)

                slopeX = x[1] - x[0]
                slopeY = y[1] - y[0]
                slopes = np.append(slopes, slope)

                if abs(slope) < 1:
                    slope_norm = np.array([slopeX, slopeY], dtype=float).reshape(2, 1)
                    slope_norm_sum = np.sqrt(np.sum(slope_norm ** 2))
                    slope_norm[0] = slope_norm[0]  / slope_norm_sum
                    slope_norm[1] = slope_norm[1]  / slope_norm_sum
                    intercept = intercept*-1
                    if (slopeX < 0 and slopeY < 0) or (slopeY < 0 and slopeX > 0):
                        rot = np.array([[0, -1], [1, 0]])
                        a = np.dot(rot, slope_norm)
                        a_vectors[:, j] = a.reshape(1, 2)
                    elif (slopeX > 0 and slopeY > 0) or (slopeY > 0 and slopeX < 0):
                        rot = np.array([[0, 1], [-1, 0]])
                        a = np.dot(rot, slope_norm)
                        a_vectors[:, j] = a.reshape(1, 2)
                else:
                    slope_norm = np.array([slopeX, slopeY], dtype=float).reshape(2, 1)


                    if (slopeX < 0 and slopeY < 0) or (slopeY > 0 and slopeX < 0):
                        rot = np.array([[0, 1], [-1, 0]])
                        a = np.dot(rot, slope_norm)
                        a_vectors[:, j] = a.reshape(1, 2)
                    elif (slopeX > 0 and slopeY > 0) or (slopeY < 0 and slopeX > 0):
                        rot = np.array([[0, -1], [1, 0]])
                        a = np.dot(rot, slope_norm)
                        a_vectors[:, j] = a.reshape(1, 2)

                intercepts = np.append(intercepts, intercept)

                if it == len(obstacles[i]['vertices'])-1:
                    point_1 = obstacles[i]['vertices'][-1]
                    point_2 = obstacles[i]['vertices'][0]

                    x = [point_1[0], point_2[0]]
                    y = [point_1[1], point_2[1]]
                    slope, intercept, _, _, _ = linregress(x, y)

                    slopeX = x[1] - x[0]
                    slopeY = y[1] - y[0]
                    slopes = np.append(slopes, slope)

                    if abs(slope) < 1:
                        slope_norm = np.array([slopeX, slopeY], dtype=float).reshape(2, 1)
                        slope_norm_sum = np.sqrt(np.sum(slope_norm ** 2))
                        slope_norm[0] = slope_norm[0] / slope_norm_sum
                        slope_norm[1] = slope_norm[1] / slope_norm_sum
                        intercept = intercept * -1
                        if (slopeX < 0 and slopeY < 0) or (slopeY < 0 and slopeX > 0):
                            rot = np.array([[0, -1], [1, 0]])
                            a = np.dot(rot, slope_norm)
                            a_vectors[:, -1] = a.reshape(1, 2)
                        elif (slopeX > 0 and slopeY > 0) or (slopeY > 0 and slopeX < 0):
                            rot = np.array([[0, 1], [-1, 0]])
                            a = np.dot(rot, slope_norm)
                            a_vectors[:, -1] = a.reshape(1, 2)
                    else:
                        slope_norm = np.array([slopeX, slopeY], dtype=float).reshape(2, 1)
                        if (slopeX < 0 and slopeY < 0) or (slopeY < 0 and slopeX > 0):
                            rot = np.array([[0, -1], [1, 0]])
                            a = np.dot(rot, slope_norm)
                            a_vectors[:, -1] = a.reshape(1, 2)
                        elif (slopeX > 0 and slopeY > 0) or (slopeY > 0 and slopeX < 0):
                            rot = np.array([[0, 1], [-1, 0]])
                            a = np.dot(rot, slope_norm)
                            a_vectors[:, -1] = a.reshape(1, 2)

                    intercepts = np.append(intercepts, intercept)

            obstacles[i]['a'] = a_vectors
            obstacles[i]['slopes'] = slopes
            obstacles[i]['intercepts'] = intercepts
        self.obs = obstacles

        # initialize obstacles for chance constraints
        self.init_obs_constraints(obstacles)

        if animate:
            self.x_list = []
            self.y_list = []
            self.z_list = []
            for i in range(1, len(obstacles)+1):
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

    def init_obs_constraints(self, obstacles):


        # create integer variable for chance constraints
        self.obs_index = []

        for i in range(1, len(obstacles)+1):
             self.obs_index.append(obstacles[i]['polygon_type'])

        self.I = self.opti.variable(sum(self.obs_index), 1)

        # set chance constraints for obstacles
        # initialize c parameter for chance constraint equation, this value will change for each time step
        self.c = self.opti.parameter(sum(self.obs_index), 1)
        self.opti.set_value(self.c, 0)

        # initialize integer constraints
        iter_2 = 0
        for i in range(0, len(self.obs_index)):
            sum_I = 0
            iter_1 = iter_2
            iter_2 = iter_2 + self.obs_index[i]
            for j in range(iter_1, iter_2):
                sum_I = sum_I + self.I[j]
            self.opti.subject_to(sum_I >= 1)

        # provide constraints on the integer variable
        self.opti.subject_to(self.opti.bounded(0, self.I, 1))
        #self.chance_constraints()



        #self.test = self.opti.parameter(sum(self.obs_index, 1))
        #self.opti.set_value(self.test, 1)


        iter_2 = 0
        for i in range(0, len(self.obs_index)):
            iter_1 = iter_2
            iter_2 = iter_2 + self.obs_index[i]
            index_slope_intercept = 0
            r = self.obs[i + 1]['risk']

            for j in range(iter_1, iter_2):
                a = self.obs[i + 1]['a'][:, index_slope_intercept].reshape(2, 1)
                b = self.obs[i + 1]['intercepts'][index_slope_intercept]

                self.opti.set_value(self.c[j], np.sqrt(np.dot(np.dot(2 * np.transpose(a), self.r1_cov_curr), a)) * erfinv((1 - 2 * r)))

                #self.opti.set_value(self.c[j], 1)
                index_slope_intercept = index_slope_intercept + 1

                for k in range(0, self.N + 1):
                    self.opti.subject_to(
                        (((mtimes(np.transpose(a), self.X[0:2, k]) - b)) * self.I[j] + self.slack[0, k]) >= self.c[j] * self.I[
                            j] - self.slack[0, k])

    def check_obstacles(self):

        return 0

    def change_goal_point(self, goal_pos):

        return 0

    def distance_pt_line(self, slope, intercept, point):
        A = -slope
        B = 1
        C = -intercept
        d = np.abs(A*point[0] + B*point[1] + C) / np.sqrt(A**2 + B**2)
        return d

    def animate(self, curr_pos):
        plt.cla()
        plt.xlim(0, 10)
        plt.ylim(0, 10)
        self.ax.set_zlim(0, 10)
        # graph robot as a round sphere for simplicity
        self.ax.plot_surface(self.x_fig + curr_pos[0], self.y_fig + curr_pos[1], self.z_fig,
                             rstride=4, cstride=4, color='b')
        x_togo = 2 * np.cos(curr_pos[2])
        y_togo = 2 * np.sin(curr_pos[2])

        # graph direction of the robot heading
        self.ax.quiver(curr_pos[0], curr_pos[1], 0, x_togo, y_togo, 0, color='red', alpha=.8, lw=3)

        # graph obstacles
        for i in range(0, len(self.x_list)):
            verts = [list(zip(self.x_list[i], self.y_list[i], self.z_list[i]))]
            self.ax.add_collection3d(Poly3DCollection(verts))

        #self.abline(self.slope_line, self.intercept)

        plt.show()
        plt.pause(0.001)

    def abline(self, slope, intercept):
        """Plot a line from slope and intercept"""
        axes = plt.gca()
        x_vals = np.array(axes.get_xlim())
        y_vals = intercept + slope * x_vals

        plt.plot(x_vals, y_vals, '--')

        return x_vals, y_vals

if __name__ == '__main__':

    # initialize all required variables for the SMPC solver
    dT = 0.1
    mpc_horizon = 5
    curr_pos = np.array([0, 4, 0]).reshape(3, 1)
    goal_pos = np.array([10, 5, 0]).reshape(3, 1)
    goal_position = np.array([10, 6, 0]).reshape(3,1)
    robot_size = 0.5
    max_nObstacles = 2
    field_of_view = {'max_distance': 10.0, 'angle_range': [45, 135]}
    lb_state = np.array([[-20], [-20], [-2*pi]], dtype=float)
    ub_state = np.array([[20], [20], [2*pi]], dtype=float)
    lb_control = np.array([[-1.5], [-1.5], [-np.pi/2]], dtype=float)
    ub_control = np.array([[1.5], [1.5], [np.pi/2]], dtype=float)

    Q = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
    R = np.array([[1, 0, 0], [0, 0.1, 0], [0, 0, 0.0001]])
    angle_noise_r1 = 0.0
    angle_noise_r2 = 0.0
    relative_measurement_noise_cov = np.array([[0.0,0], [0,0.0]])
    maxComm_distance = -10
    animate = True
    # initialize obstacles to be seen
    #obs = ({1: {'vertices': [[5, 5], [8, 5.1], [8.1, 7]], 'a': [], 'slopes': [], 'intercepts': [], 'polygon_type': 3, 'risk': 0.1}})
    obs = {1: {'vertices': [[3.9, 4], [4, 6], [6, 6.1], [5.9, 4.1]], 'a': [], 'slopes': [], 'intercepts': [], 'polygon_type': 4, 'risk': 0.1}}
    obs.update(
        {2: {'vertices': [[6, 5], [7, 7], [8, 5.2]], 'a': [], 'slopes': [], 'intercepts': [], 'polygon_type': 3, 'risk': 0.1}})

    SMPC = SMPC_UGV_Planner(dT, mpc_horizon, curr_pos, goal_pos, robot_size, max_nObstacles, field_of_view, lb_state,
                            ub_state, lb_control, ub_control, Q, R, angle_noise_r1, angle_noise_r2,
                            relative_measurement_noise_cov, maxComm_distance, obs, animate)


    while m.sqrt((curr_pos[0] - goal_position[0]) ** 2 + (curr_pos[1] - goal_position[1]) ** 2) > 1.0:


        print(SMPC.goal_pos)

        sol = SMPC.opti.solve()
        x = sol.value(SMPC.X)[:,1]
        print(sol.value(SMPC.I))
        curr_pos = np.array(x).reshape(3,1)
        SMPC.opti.set_value(SMPC.r1_pos, x)
        SMPC.animate(curr_pos)



