#!/usr/bin/env python

"""FILE CREATED BY: Alexander Schperberg, aschperb@gmail.com
Copyright by RoMeLa (Robotics and Mechanisms Laboratory, University of California, Los Angeles)"""

# This file provides a simple model predictive controller for a unmanned ground vehicle for data collection
# purposes that moves a ground vehicle to several user-specified goal locations

from casadi import *
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import matplotlib.pyplot as plt
import math as m
import time
from ROS_interface import *
#from keras.models import load_model
#model = load_model('rnn_models/pf_SLAM.h5')

class MPC_UGV_Planner():

    def __init__(self, dT, mpc_horizon, curr_pos, lb_state,
                 ub_state, lb_control, ub_control, Q, R, robot_size, animate):

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
        self.R = R
        # initialize discretized state matrices A and B
        self.A = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
        # initialize robot's current position
        self.curr_pos = curr_pos
        self.initVariables()
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
        self.opti.set_initial(self.X, 0)

        # initialize, linear and angular velocity control variables (v, and w), and repeat above procedure
        self.U = self.opti.variable(2, self.N)
        self.opti.set_initial(self.U, 0)

        # initialize the current robot pos (x, y and th current position)
        self.r_pos = self.opti.parameter(3, 1)
        self.opti.set_value(self.r_pos, self.curr_pos)

        # initialize the goal point (x, y and th current position)
        self.r_goal = self.opti.parameter(3, 1)

        # initialize the objective function
        self.obj()

    def obj(self):
        self.objFunc = 0

        for k in range(0, self.N):
            con = self.U[:, k]
            st = self.X[:, k + 1]

            self.objFunc = self.objFunc + mtimes(mtimes((st - self.r_goal).T, self.Q), st - self.r_goal) + \
                           0.5*mtimes(mtimes(con.T, self.R), con)

        # initialize the objective function into the solver
        self.opti.minimize(self.objFunc)

        self.init_constraints()

    def init_constraints(self):
        # constrain the current state, and bound current and future states by their limits
        self.opti.subject_to(self.X[:, 0] == self.r_pos)
        self.opti.subject_to(self.opti.bounded(self.lb_state, self.X, self.ub_state))
        self.opti.subject_to(self.opti.bounded(self.lb_control, self.U, self.ub_control))

        # initiate multiple shooting constraints
        for k in range(0, self.N):
            next_state = self.next_state_nominal(self.X[:, k], self.U[:, k])
            self.opti.subject_to(self.X[:, k + 1] == next_state)

        # set solver
        self.pre_solve()

    def next_state_nominal(self, x, u):
        next_state = mtimes(self.A, x) + mtimes(self.dT,vertcat(u[0]*cos(x[2]), u[0]*sin(x[2]), u[1]))
        return next_state

    def pre_solve(self):

        # set solver options for ipopt (nonlinear programming)
        opts = {'ipopt': {'max_iter': 100, 'print_level': 0, 'acceptable_tol': 10**-9,
                      'acceptable_obj_change_tol': 10**-7}}
        opts.update({'print_time': 0})
        # create solver
        self.opti.solver('ipopt', opts)

    def rnn_cov(self, scans, mpc_future_states):
        # TODO: get scans, convert scan distances into x,y positions, propagate mpc solution to 'predict' future scan
        # TODO: information, which serves as input to the RNN function
        #RNN_input = MPC_Generated_Measurements[np.newaxis, :]
        #RNN_output_predict = model.predict(RNN_input)[0]
        return 0

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
        plt.show()
        plt.pause(0.001)

"""
if __name__ == '__main__':

    # initialize all required variables for the MPC solver
    dT = 0.1
    mpc_horizon = 5
    curr_pos = np.array([0, 0, 0]).reshape(3,1)
    goal_points = [[4, 4, 0]]

    robot_size = 0.5
    lb_state = np.array([[-20], [-20], [-2*pi]], dtype=float)
    ub_state = np.array([[20], [20], [2*pi]], dtype=float)
    lb_control = np.array([[-0.2], [-1.82]], dtype=float) # CHANGED
    ub_control = np.array([[0.2], [1.82]], dtype=float) # CHANGED
    Q = np.array([[1, 0, 0], [0, 2, 0], [0, 0, 0.1]])
    R = np.array([[0.5, 0], [0, 0.05]])
    animate = True

    MPC = MPC_UGV_Planner(dT, mpc_horizon, curr_pos, lb_state,
                            ub_state, lb_control, ub_control, Q, R, robot_size, animate)

    ROS = ROSInterface()
    rospy.init_node('ros_interface')

    for i in range(0, len(goal_points)):

        goal_pos = np.array(goal_points[i])
        MPC.opti.set_value(MPC.r_goal, goal_pos)

        while m.sqrt((curr_pos[0] - goal_pos[0]) ** 2 + (curr_pos[1] - goal_pos[1]) ** 2) > 0.5 and not rospy.is_shutdown():

            sol = MPC.opti.solve()
            u_vec = sol.value(MPC.U[:, 0])
            ROS.send_velocity(u_vec)
            curr_pos = ROS.get_current_pose()
            MPC.opti.set_value(MPC.r_pos, curr_pos)
            MPC.animate(curr_pos)
"""