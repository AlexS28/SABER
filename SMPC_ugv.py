"""FILE CREATED BY: Alexander Schperberg, aschperb@gmail.com
Copyright by RoMeLa (Robotics and Mechanisms Laboratory, University of California, Los Angeles)"""

# This file provides a stochastic and robust model predictive controller for a simple unmanned ground vehicle that
# moves a ground vehicle to any desired goal location, while considering obstacles (represented as 2D polygons) and
# cross communication ability with another robot


from casadi import *
import numpy as np



class SMPC_UGV_Planner():

    def __init__(self, dT, mpc_horizon, robot_size, max_nObstacles, field_of_view, lb_state, ub_state,
                 lb_control, ub_control, Q, R, angle_noise_r1, angle_noise_r2,  measurement_noise_cov_r1,
                 measurement_noise_cov_r2):

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
        # and terminal slack respectively
        self.Q = Q
        self.R = R

        # initialize discretized state matrices A and B (note, A is constant, but B will change as it is a function of
        # state theta)
        self.A = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
        self.B = np.array([[0, 0], [0, 0], [0, 0]])

        # initialize measurement noise (in our calculation, measurement noise is set by the user and is gaussian,
        # zero-mean). It largely represents the noise due to communication transmitters, or other sensor devices. It
        # is assumed to be a 3x3 matrix (x, y, and theta) for both robots
        self.measurement_noise_cov_r1 = measurement_noise_cov_r1
        self.measurement_noise_cov_r2 = measurement_noise_cov_r2

        # we assume that there is constant noise in angle (while x and y are dynamically updated) - should be a variance
        # value
        self.angle_noise_r1 = angle_noise_r1
        self.angle_noise_r2 = angle_noise_r2

        # to produce our state and covariance updates due to cooperative localization, we need the future states,
        # system noise and measurement noise from the second robot
        #TODO: Still need to incorporate future states and system noise covariance from robot 2

        #self.robot2_futureStates = np.zeros((3, self.N + 1))
        #self.robot2_futureSystemNoise_cov = np.zeros((self.N + 1, 3, 3))
        #self.robot2_futureMeasurementNoise_cov = np.zeros((self.N + 1, 3, 3))

        # initialize cross diagonal system noise covariance matrix
        self.P12 = np.array([[0,0,0], [0,0,0], [0,0,0]])

        # bool variable to indicate whether the robot has made first contact with the uav
        self.first_contact = False

        # initialize state, control, and slack variables
        self.initVariables()
        # initialize objective function
        self.obj()

    def initVariables(self):

        # initialize x, y, and theta state variables
        x = SX.sym('x')
        y = SX.sym('y')
        th = SX.sym('th')
        self.states = vertcat(x, y, th)
        # calculate total number of states
        self.nStates = self.states.numel()
        # initialize states for entire prediction horizon
        self.X = SX.sym('X', self.nStates, (self.N + 1))

        # initialize, linear and angular velocity control variables (v, and w), and repeat above procedure
        v = SX.sym('v')
        w = SX.sym('w')
        self.controls = vertcat(v, w)
        self.nControls = self.controls.numel()
        self.U = SX.sym('U', self.nControls, self.N)

        # initialize slack variables for states for prediction horizon N
        self.S1 = SX.sym('S', 1, self.N)
        # initialize slack variable for the terminal state, N+1
        self.ST = SX.sym('ST', 1, 1)

        # initialize the parameter variables, where the first 6 entries constitute the current and goal states
        # while the next N * 4 entries constitutes the current and future covariances matrices (representing the
        # system noise predicted by the RNN) which are flattened into a 1 x 4 array.
        self.P = SX.sym('P', self.nStates * 2 + self.N * 4)

    # the nominal next state is calculated for use as a terminal constraint in the objective function
    def next_state_nominal(self, x, u):
        self.B = np.array([[self.dT * cos(x[2]), 0], [self.dT * sin(x[2], 0)], [0, self.dT]], dtype=float)
        next_state = mtimes(self.A, x) + mtimes(self.B, u)
        return next_state

    # the next state is calculated with consideration of system noise
    def next_state_withSystemNoise(self, x, u, system_noise_cov):
        # the system_noise_covariance will be a flattened 1x4 array, provided by the output of an RNN. We need to
        # convert it into a 3x3 matrix. We will assume a constant noise in theta however.
        system_noise_cov_converted = np.array([[system_noise_cov[0], 0, 0], [0, system_noise_cov[3], 0], ...
        [0, 0, self.angle_noise_r1]])

        # sample a gaussian distribution of the system_noise covariance, and then take the square root of it
        system_noise = numpy.random.multivariate_normal(0, system_noise_cov_converted, check_valid='warn')

        # add the noise to the linear and angular velocity
        self.B = np.array([[self.dT * cos(x[2]), 0], [self.dT * sin(x[2], 0)], [0, self.dT]], dtype=float)
        next_state = mtimes(self.A, x) + mtimes(self.B, u) + system_noise
        return next_state

    # state and system noise covariances need to be updated to consider possible cooperative localization
    # between two robots
    def state_update(self, x_r1, x_r2, system_noise_cov_r1, system_noise_cov_r2):

        # inputs are the position of robot 1, robot 2, and their corresponding system noise covariance matrix provided
        # by the RNN

        # the output will be the updated state and also the updated system noise covariance matrix

        # check if this is the first time both robots are making contact, and if they are within distance
        if self.first_contact and np.linalg.norm(x_r1[0:2] - x_r2[0:2]):

            # calculate the measured state before update for both robots
            x_hat_r1 = x_r1 + numpy.random.multivariate_normal(0, self.measurement_noise_cov_r1, check_valid='warn')
            x_hat_r2 = x_r2 + numpy.random.multivariate_normal(0, self.measurement_noise_cov_r2, check_valid='warn')

            # calculate the relative positions of the two contacting robots
            z = x_r1 - x_r2

            # the system_noise_covariance will be a flattened 1x4 array, provided by the output of an RNN. We need to
            # convert it into a 3x3 matrix. We will assume a constant noise in theta however.
            #TODO Check to see if we need to include the off-diagonal values of the system noise covariance

            system_noise_cov_converted_r1 = np.array([[system_noise_cov_r1[0], 0, 0], [0, system_noise_cov_r1[3], 0],
            [0, 0, self.angle_noise_r1]])

            system_noise_cov_converted_r2 = np.array([[system_noise_cov_r2[0], 0, 0], [0, system_noise_cov_r2[3], 0],
            [0, 0, self.angle_noise_r2]])

            P11 = system_noise_cov_converted_r1
            P22 = system_noise_cov_converted_r2

            # placeholder R12 (relative measurement noise between robot 1 and 2)
            R12 = np.zeros(3,3)             #TODO: Add the correct R12 matrix here

            # calculate the S matrix
            S = P11 + P22 + R12

            # calculate the inverse S matrix, if not possible, assume zeros
            try:
                S_inv = np.linalg.inv(S)

            except ValueError:
                S_inv = np.zeros(3, 3)

            # update the cross-diagonal matrix
            self.P12 = mtimes(mtimes(P11 * S_inv) * P22)

            # calculate the kalman gain
            K = mtimes(P11, S_inv)

            # calculate the updated state for the ugv
            x_hat_r1 = x_hat_r1 + mtimes(K, z - (x_hat_r1 - x_hat_r2))

            # calculate the updated system noise covariance for the ugv
            P11 = P11 - mtimes(mtimes(P11, S_inv), P11)

            # ensure this function is only run at first contact
            self.first_contact = False

            return x_hat_r1, P11

        # the second update and beyond, the following equations are used if both robots are within contact
        elif (not self.first_contact) and np.linalg.norm(x_r1[0:2] - x_r2[0:2]):

            # calculate the measured state before update for both robots
            x_hat_r1 = x_r1 + numpy.random.multivariate_normal(0, self.measurement_noise_cov_r1, check_valid='warn')
            x_hat_r2 = x_r2 + numpy.random.multivariate_normal(0, self.measurement_noise_cov_r2, check_valid='warn')

            # calculate the relative positions of the two contacting robots
            z = x_r1 - x_r2

            # the system_noise_covariance will be a flattened 1x4 array, provided by the output of an RNN. We need to
            # convert it into a 3x3 matrix. We will assume a constant noise in theta however.
            # TODO Check to see if we need to include the off-diagonal values of the system noise covariance

            system_noise_cov_converted_r1 = np.array([[system_noise_cov_r1[0], 0, 0], [0, system_noise_cov_r1[3], 0],
            [0, 0, self.angle_noise_r1]])

            system_noise_cov_converted_r2 = np.array([[system_noise_cov_r2[0], 0, 0], [0, system_noise_cov_r2[3], 0],
            [0, 0, self.angle_noise_r2]])

            P11_before_upd = system_noise_cov_converted_r1
            P22_before_upd = system_noise_cov_converted_r2

            # placeholder R12 (relative measurement noise between robot 1 and 2)
            R12 = np.zeros(3,3)             #TODO: Add the correct R12 matrix here

            # calculate the S matrix
            S = P11_before_upd - self.P12 - np.transpose(self.P12) + P22_before_upd + R12

            # calculate the inverse S matrix, if not possible, assume zeros
            try:
                S_inv = np.linalg.inv(S)

            except ValueError:
                S_inv = np.zeros(3, 3)

            # calculate the kalman gain
            K = mtimes((P11_before_upd - self.P12), S_inv)

            # calculate the updated state for the ugv
            x_hat_r1 = x_hat_r1 + mtimes(K, z - (x_hat_r1 - x_hat_r2))


            # calculate the updated system noise covariance for the ugv
            P11 = P11_before_upd - mtimes(mtimes((P11_before_upd - self.P12), S_inv), P11_before_upd - self.P12)

            # update the cross-diagonal matrix
            self.P12 = self.P12 - mtimes(mtimes((P11_before_upd - self.P12), S_inv), (self.P11 - P22_before_upd))

            return x_hat_r1, P11

        # if the robots are not in contact range, then no cooperative localization calculations can be done
        else:
            # calculate the measured state before update for both robots
            x_hat_r1 = x_r1 + numpy.random.multivariate_normal(0, self.measurement_noise_cov_r1, check_valid='warn')

            # the system_noise_covariance will be a flattened 1x4 array, provided by the output of an RNN. We need to
            # convert it into a 3x3 matrix. We will assume a constant noise in theta however.
            # TODO Check to see if we need to include the off-diagonal values of the system noise covariance
            system_noise_cov_converted_r1 = np.array([[system_noise_cov_r1[0], 0, 0], [0, system_noise_cov_r1[3], 0],
                                                      [0, 0, self.angle_noise_r1]])
            P11 = system_noise_cov_converted_r1


            return x_hat_r1, P11

    def obj(self):
        pass
