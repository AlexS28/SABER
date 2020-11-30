
"""FILE CREATED BY: Alexander Schperberg, aschperb@gmail.com
Copyright by RoMeLa (Robotics and Mechanisms Laboratory, University of California, Los Angeles)"""

# This file provides a simple model predictive controller for a unmanned aerial vehicle for data collection
# purposes that moves a aerial to several user-specified goal locations

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

    def __init__(self, dT, mpc_horizon, curr_pos, lb_state,
                 ub_state, lb_control, ub_control, Q, R, robot_size, obs, animate, multi_agent):

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

        # initialize the goal point (x, y and th current position)
        self.r_goal = self.opti.parameter(10, 1)

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
            next_state = self.next_state_nominal(self.X[:, k], self.U[:, k])
            self.opti.subject_to(self.X[:, k + 1] == next_state)

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

    def pre_solve(self):
        # TODO: As with the UGV, add line constraints for the UAV, this requires bonmin solver. For now, ipopt is used
        # TODO: as only circular constraints are considered for the UAV

        """
        opts = {'bonmin.warm_start': 'interior_point', 'discrete': OT_Boolvector, 'error_on_fail': True,
                'bonmin.time_limit': 0.5,
                'bonmin.acceptable_obj_change_tol': 1e40, 'bonmin.acceptable_tol': 1e-1}
        # create the solver
        self.opti.solver('bonmin', opts)
        """

        # set solver options for ipopt (nonlinear programming)
        opts = {'ipopt': {'max_iter': 1000, 'print_level': 0, 'acceptable_tol': 1,
                      'acceptable_obj_change_tol': 1}}
        opts.update({'print_time': 0})
        # create solver
        self.opti.solver('ipopt', opts)

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


"""
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

    ROS = ROSInterface(False)
    rospy.init_node('ros_interface')
    rate = rospy.Rate(10)
    ROS.send_velocityUAV([0, 0, 0, 0, 0])
    curr_posROS = ROS.get_current_poseUAV()
    curr_pos[0] = curr_posROS[0]
    curr_pos[4] = curr_posROS[1]
    curr_pos[8] = curr_posROS[2]
    animate = True
    SMPC = SMPC_UAV_Planner(dT, mpc_horizon, curr_pos, lb_state,
                            ub_state, lb_control, ub_control, Q, R, robot_size, obs, animate, multi_agent=False)

    for i in range(0, len(goal_points)):
        goal_pos = np.array(goal_points[i])
        SMPC.opti.set_value(SMPC.r_goal, goal_pos)

        while m.sqrt((curr_pos[0] - goal_pos[0]) ** 2 + (curr_pos[4] - goal_pos[4]) ** 2 + (curr_pos[8] - goal_pos[8])**2) > 0.5 and not rospy.is_shutdown():
            sol = SMPC.opti.solve()
            x = sol.value(SMPC.X)[:, 1]
            x_vel = sol.value(SMPC.X[:, SMPC.N])
            curr_pos = np.array(x).reshape(10, 1)
            curr_pos2 = np.array(x_vel).reshape(10, 1)

            ROS.send_velocityUAV([curr_pos[1], curr_pos[5], curr_pos[9], curr_pos[3], curr_pos[7]])
            curr_posROS = ROS.get_current_poseUAV()
            curr_pos[0] = curr_posROS[0]
            curr_pos[4] = curr_posROS[1]
            curr_pos[8] = curr_posROS[2]

            SMPC.opti.set_value(SMPC.r_pos, curr_pos)
            SMPC.check_obstacles([curr_pos[0], curr_pos[4]])
            SMPC.animate(curr_pos)
            rate.sleep()
"""