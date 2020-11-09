#!/usr/bin/env python
from sabr_pkg.SMPC_ugv import *
from sabr_pkg.SMPC_uav import *
from sabr_pkg.ROS_interface_ugv_uav import *
# This code runs the 'Synchronous and Autonomous Robots in Uncertain Environments with Reinforcement Learning' (SABR)
# algorithm. It combines a stochastic MPC for a multi-agent (UAV-UGV) system, which infers future uncertainty covariances
# using a recurrent neural network (trained on filter estimations provided by a particle-filter or visual-inertial
# SLAM system----depending if the robot is equipped with a LiDAR or RGB-D camera configuration respectively).
# Lastly, to provide global # target locations for both robots, is done using a DQN, which is rewarded based on whether
# both robots reach their # final goal destination, if their SMPC controller succeeds in finding solutions, and
# if uncertainty is always below a specified threshold.

# initialize obstacles

obs = {1: {'vertices': [[-3.+5, -1+5,0], [-3.02+5, 1.03+5,0], [3+5,1+5,0], [3.02+5, -1.05+5,0]], 'a': [], 'slopes': [], 'intercepts': [],
               'polygon_type': 4, 'risk': 0.4}}
obs.update(
        {2: {'vertices': [[6, 5,0], [7, 7,0], [8, 5.2,0]], 'a': [], 'slopes': [], 'intercepts': [], 'polygon_type': 3,
             'risk': 0.4}})
obs.update({3: {'vertices': [[2, 2]], 'size': 0.5, 'polygon_type': 1, 'risk': 0.4}})


# initialize prediction horizon and discretized time, and whether to animate
dT = 0.5
mpc_horizon = 10
animate = True

# initialize SMPC parameters for the UGV
curr_posUGV = np.array([0, 0, 0]).reshape(3,1)
goal_pointsUGV = [[10, 10, 0], [0, 0, 0]]
robot_size = 0.5
lb_state = np.array([[-20], [-20], [-2*pi]], dtype=float)
ub_state = np.array([[20], [20], [2*pi]], dtype=float)
lb_control = np.array([[-0.2], [-1.82/8]], dtype=float) # CHANGED
ub_control = np.array([[0.2], [1.82/8]], dtype=float) # CHANGED
Q = np.array([[1, 0, 0], [0, 2, 0], [0, 0, .05]])
R_init = np.array([[0.5, 0, 0], [0, 0.5, 0] ,[0, 0, 0.5]])


angle_noise_r1 = 0.0
angle_noise_r2 = 0.0
relative_measurement_noise_cov = np.array([[0.0,0], [0,0.0]])
maxComm_distance = -10
failure_count = 0

SMPC_UGV = SMPC_UGV_Planner(dT, mpc_horizon, curr_posUGV, robot_size, lb_state,
                            ub_state, lb_control, ub_control, Q, R_init, angle_noise_r1, angle_noise_r2,
                            relative_measurement_noise_cov, maxComm_distance, obs, animate)


# initialize SMPC parameters for the UAV
dT = 0.5
mpc_horizon = 10
# x, vel_x, th1, th1_vel, y, vel_y, th2, th2_vel, z, z_vel
curr_posUAV = np.array([5,0,0,0,0,0,0,0,0,0]).reshape(10,1)
goal_pointsUAV = [[5,0,0,0,0,0,0,0,5,0], [10,0,0,0,10,0,0,0,5,0]]
robot_size = 0.5

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

# if the UAV and UGV are to be animated in the same screen, multi_agent must be set to true
SMPC_UAV = SMPC_UAV_Planner(dT, 10, curr_posUAV, lb_state,
                            ub_state, lb_control, ub_control, Q, R, robot_size, obs, animate, multi_agent=True)

goal_posUGV = np.array(goal_pointsUGV[0])
SMPC_UGV.opti.set_value(SMPC_UGV.r1_goal, goal_posUGV)

goal_posUAV = np.array(goal_pointsUAV[0])
SMPC_UAV.opti.set_value(SMPC_UAV.r_goal, goal_posUAV)

indexUGV = 1
indexUAV = 1


# start ROS interface for ugv and uav combination
ROS = ROSInterfaceUGV_UAV()
rospy.init_node('ros_interface_ugv_uav')
rate = rospy.Rate(10)
#ROS.send_velocityUAV([0, 0, 0, 0, 0])
#curr_posROS = ROS.get_current_poseUAV()
#curr_posUAV[0] = curr_posROS[0]
#curr_posUAV[4] = curr_posROS[1]
#curr_posUAV[8] = curr_posROS[2]
#SMPC_UAV.opti.set_value(SMPC_UAV.r_pos, curr_posUAV)
#ROS.send_velocity([0, 0])
#curr_posROS = ROS.get_current_pose()
#curr_posUGV[0] = curr_posROS[0]
#curr_posUGV[1] = curr_posROS[1]
#curr_posUGV[2] = curr_posROS[2]
#SMPC_UGV.opti.set_value(SMPC_UGV.r1_pos, curr_posUGV)

while m.sqrt((curr_posUGV[0] - goal_posUGV[0]) ** 2 + (curr_posUGV[1] - goal_posUGV[1]) ** 2) > 0.5 or m.sqrt((curr_posUAV[0]
            - goal_posUAV[0]) ** 2 + (curr_posUAV[4] - goal_posUAV[4]) ** 2 + (curr_posUAV[8] - goal_posUAV[8])**2) > 0.5:

    # find SMPC solution for UGV
    try:
        solUGV = SMPC_UGV.opti.solve()
        uUGV = solUGV.value(SMPC_UGV.U[:, 0])
        ROS.send_velocity(uUGV)
        curr_posUGV = ROS.get_current_pose()
        curr_posUGV = np.array(curr_posUGV).reshape(3, 1)
        print(curr_posUGV)
        SMPC_UGV.check_obstacles(np.concatenate((curr_posUGV[0], curr_posUGV[1], [0])))
    except:
        curr_posUGV = ROS.get_current_pose()
        curr_posUGV = np.array(curr_posUGV).reshape(3, 1)
        SMPC_UGV.check_obstacles(np.concatenate((curr_posUGV[0], curr_posUGV[1], [0])))

    SMPC_UGV.opti.set_value(SMPC_UGV.r1_pos, curr_posUGV)


    # find SMPC solution for UAV
    try:
        solUAV = SMPC_UAV.opti.solve()
        uUAV = solUAV.value(SMPC_UAV.X[:, 1])
        curr_posUAV = np.array(uUAV).reshape(10, 1)
        ROS.send_velocityUAV([curr_posUAV[1], curr_posUAV[5], curr_posUAV[9], curr_posUAV[3], curr_posUAV[7]])

        curr_posUAVROS = ROS.get_current_poseUAV()
        curr_posUAV[0] = curr_posUAVROS[0]
        curr_posUAV[4] = curr_posUAVROS[1]
        curr_posUAV[8] = curr_posUAVROS[2]
        SMPC_UAV.check_obstacles([curr_posUAV[0], curr_posUAV[4]])
    except:
        curr_posUAVROS = ROS.get_current_poseUAV()
        curr_posUAV[0] = curr_posUAVROS[0]
        curr_posUAV[4] = curr_posUAVROS[1]
        curr_posUAV[8] = curr_posUAVROS[2]
        SMPC_UAV.check_obstacles([curr_posUAV[0], curr_posUAV[4]])

    SMPC_UAV.opti.set_value(SMPC_UAV.r_pos, curr_posUAV)

    dist_goalUGV = m.sqrt((curr_posUGV[0] - goal_posUGV[0]) ** 2 + (curr_posUGV[1] - goal_posUGV[1]) ** 2)
    dist_goalUAV = m.sqrt((curr_posUAV[0]
                           - goal_posUAV[0]) ** 2 + (curr_posUAV[4] - goal_posUAV[4]) ** 2 + (
                                      curr_posUAV[8] - goal_posUAV[8]) ** 2)

    if dist_goalUGV < 0.5 and indexUGV <= len(goal_pointsUGV) - 1:
        goal_posUGV = np.array(goal_pointsUGV[indexUGV])
        SMPC_UGV.opti.set_value(SMPC_UGV.r1_goal, goal_posUGV)
        indexUGV += 1

    if dist_goalUAV < 0.5 and indexUAV <= len(goal_pointsUAV) - 1:
        goal_posUAV = np.array(goal_pointsUAV[indexUAV])
        SMPC_UAV.opti.set_value(SMPC_UAV.r_goal, goal_posUAV)
        indexUAV += 1

    SMPC_UGV.animate(curr_posUGV)
    SMPC_UAV.animate_multi_agents(SMPC_UGV.ax, curr_posUAV)
    rate.sleep()
    plt.show()
    plt.pause(0.001)

    """
    # find SMPC solution for UGV
    try:
        solUGV = SMPC_UGV.opti.solve()
        x = solUGV.value(SMPC_UGV.X)[:, 1]
        u = solUGV.value(SMPC_UGV.U[:, SMPC_UGV.N - 1])
        SMPC_UGV.check_obstacles(x[0:2])
    except:
        u = solUGV.value(SMPC_UGV.U[:, 0])
        u[1] = 0
        x = SMPC_UGV.next_state_nominal(curr_posUGV, u)
        print('WARNING: Solver has failed for UGV, using previous control value for next input')
    curr_posUGV = np.array(x).reshape(3, 1)
    SMPC_UGV.opti.set_value(SMPC_UGV.r1_pos, x)

    # find SMPC solution for UAV
    try:
        solUAV = SMPC_UAV.opti.solve()
        x = solUAV.value(SMPC_UAV.X)[:, 1]
        u = solUAV.value(SMPC_UAV.U)[:, SMPC_UAV.N-1]
        SMPC_UAV.check_obstacles([curr_posUAV[0], curr_posUAV[4]])
    except:
        u = solUAV.value(SMPC_UAV.U[:,0])
        x = SMPC_UAV.next_state_nominal(curr_posUAV, u)
        print('WARNING: Solver has failed for UAV, using previous control value for next input')
    curr_posUAV = np.array(x).reshape(10, 1)
    SMPC_UAV.opti.set_value(SMPC_UAV.r_pos, x)

    dist_goalUGV = m.sqrt((curr_posUGV[0] - goal_posUGV[0]) ** 2 + (curr_posUGV[1] - goal_posUGV[1]) ** 2)
    dist_goalUAV = m.sqrt((curr_posUAV[0]
            - goal_posUAV[0]) ** 2 + (curr_posUAV[4] - goal_posUAV[4]) ** 2 + (curr_posUAV[8] - goal_posUAV[8])**2)

    if dist_goalUGV < 0.5 and indexUGV <= len(goal_pointsUGV)-1:
        goal_posUGV = np.array(goal_pointsUGV[indexUGV])
        SMPC_UGV.opti.set_value(SMPC_UGV.r1_goal, goal_posUGV)
        indexUGV += 1

    if dist_goalUAV < 0.5 and indexUAV <= len(goal_pointsUAV)-1:
        goal_posUAV = np.array(goal_pointsUAV[indexUAV])
        SMPC_UAV.opti.set_value(SMPC_UAV.r_goal, goal_posUAV)
        indexUAV += 1

    SMPC_UGV.animate(curr_posUGV)
    SMPC_UAV.animate_multi_agents(SMPC_UGV.ax, curr_posUAV)

    plt.show()
    plt.pause(0.001)
    """
