"""FILE CREATED BY: Alexander Schperberg, aschperb@gmail.com
Copyright by RoMeLa (Robotics and Mechanisms Laboratory, University of California, Los Angeles)"""

# This file provides a simple model predictive controller for a unmanned aerial vehicle for data collection
# purposes that moves a aerial to several user-specified goal locations

from casadi import *
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import math as m
import control

class MPC_UAV_Planner():

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
        A1 = 0.7969
        A2 = 0.02247
        A3 = -1.7976
        A4 = 0.9767
        B1 = 0.01166
        B2 = 0.9921
        g = 9.81
        KT = 0.91
        m = 1.3

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

        # TODO: adding the C matrix does not work, potentially a problem with equations
        #self.C = np.array([[0],[0],[0],[0],[0],[0],[0],[0],[0.0031], [0.2453]])

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

        # initialize the current robot pos
        self.r_pos = self.opti.parameter(10, 1)
        self.opti.set_value(self.r_pos, self.curr_pos)

        # initialize the goal point (x, y and th current position)
        self.r_goal = self.opti.parameter(10, 1)

        # initialize the objective function
        self.obj()

    def obj(self):
        self.objFunc = 0

        for k in range(0, self.N):
            con = self.U[:, k]
            st = self.X[:, k + 1]

            self.objFunc = self.objFunc + mtimes(mtimes((st - self.r_goal).T, self.Q), st - self.r_goal) + \
                           mtimes(mtimes(con.T, self.R), con)

        self.init_constraints()

        # initialize the objective function into the solver
        self.opti.minimize(self.objFunc)

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
        next_state = mtimes(self.A, x) + mtimes(self.B, u) #+ self.C
        return next_state

    def pre_solve(self):

        # set solver options for ipopt (nonlinear programming)
        opts = {'ipopt': {'max_iter': 100, 'print_level': 0, 'acceptable_tol': 10**-9,
                      'acceptable_obj_change_tol': 10**-7}}
        opts.update({'print_time': 0})
        # create solver
        self.opti.solver('ipopt', opts)

    def animate(self, curr_pos):
        plt.cla()
        plt.xlim(0, 10)
        plt.ylim(0, 10)
        self.ax.set_zlim(0, 10)
        # graph robot as a round sphere for simplicity
        self.ax.plot_surface(self.x_fig + curr_pos[0], self.y_fig + curr_pos[4], self.z_fig + curr_pos[8],
                             rstride=4, cstride=4, color='b')
        #x_togo = 2 * np.cos(curr_pos[2])
        #y_togo = 2 * np.sin(curr_pos[2])

        # graph direction of the robot heading
        #self.ax.quiver(curr_pos[0], curr_pos[4], curr_pos[8], x_togo, y_togo, 0, color='red', alpha=.8, lw=3)
        plt.show()
        plt.pause(0.001)

if __name__ == '__main__':

    # initialize all required variables for the SMPC solver
    dT = 0.1
    mpc_horizon = 5
    # x, vel_x, th1, th1_vel, y, vel_y, th2, th2_vel, z, z_vel
    curr_pos = np.array([0,0,0,0,0,0,0,0,0,0]).reshape(10,1)
    goal_pos = np.array([8,0,0,0,8,0,0,0,8,0]).reshape(10,1)

    robot_size = 0.5
    lb_state = np.array([[-20], [-5], [-10*(pi/180)], [-50*pi/180],[-20], [-5], [-10*(pi/180)], [-50*pi/180],[-20],[-5]], dtype=float)
    ub_state = np.array([[20], [5], [10*(pi/180)], [50*pi/180],[20], [5], [10*(pi/180)], [50*pi/180],[20],[5]], dtype=float)
    lb_control = np.array([[-10*pi/180], [-10*pi/180], [-10*pi/180]], dtype=float)
    ub_control = np.array([[10*pi/180], [10*pi/180], [10*pi/180]], dtype=float)

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

    R = np.array([[.001, 0, 0], [0, .001, 0], [0, 0, 0.001]])
    animate = True

    MPC = MPC_UAV_Planner(dT, mpc_horizon, curr_pos, lb_state,
                            ub_state, lb_control, ub_control, Q, R, robot_size, animate)

    MPC.opti.set_value(MPC.r_goal, goal_pos)

    while m.sqrt((curr_pos[0] - goal_pos[0]) ** 2 + (curr_pos[4] - goal_pos[4]) ** 2 + (curr_pos[8] - goal_pos[8])**2) > 0.001:
        sol = MPC.opti.solve()
        x = sol.value(MPC.X)[:, 1]
        print(x)
        curr_pos = np.array(x).reshape(10, 1)
        MPC.opti.set_value(MPC.r_pos, x)
        MPC.animate(curr_pos)
