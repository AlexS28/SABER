#!/usr/bin/env python

from sabr_pkg.MPC_ugv_Simple import *
import time
import os

"""
Dataset 1: world_dist_testUGV: [9, 10, 0], [0, 10, 0], [0, -10, 0], [-9,-9, 0]
Dataset 2: world_feature_rich: [8,0,0], [7, 8, 0], [-1,6,0], [-7, 4, 0], [-7, 0, 0], [1,-4,0], [6,-4,0], [0,0,0]
Dataset 3/4: world_cube/cylinderUGV: [6, 4, 0], [6,7,0], [10,10,0]
Dataset5: world_cube_cylinderUGV: [3,2,0], [6,7,0], [5,10,0], [-6,10,0], [-5.0, 3,0],[-3,-5,0],[3,-8,0],[7,-4,0], [10,0,0]
"""
# initialize all required variables for the MPC solver
dT = 0.1
mpc_horizon = 5
curr_pos = np.array([0, 0, 0]).reshape(3,1)
goal_points = [[3,2,0], [6,7,0], [5,10,0], [-6,10,0], [-5.0, 3,0],[-3,-5,0],[3,-8,0],[7,-4,0], [10,0,0]]
robot_size = 0.5
lb_state = np.array([[-20], [-20], [-2*pi]], dtype=float)
ub_state = np.array([[20], [20], [2*pi]], dtype=float)
lb_control = np.array([[-0.2], [-1.82/2]], dtype=float) # CHANGED
ub_control = np.array([[0.2], [1.82/2]], dtype=float) # CHANGED
Q = np.array([[1, 0, 0], [0, 2, 0], [0, 0, 0.1]])
R = np.array([[0.5, 0], [0, 0.05]])
animate = True

MPC = MPC_UGV_Planner(dT, mpc_horizon, curr_pos, lb_state,
                            ub_state, lb_control, ub_control, Q, R, robot_size, animate)

ROS = ROSInterface(True)
rospy.init_node('ros_interface')
rate = rospy.Rate(10)

# start data collection
# initiate data collection list
dataset = np.zeros((1, 367))
time.sleep(5)
for i in range(0, len(goal_points)):

    goal_pos = np.array(goal_points[i])
    MPC.opti.set_value(MPC.r_goal, goal_pos)

    while m.sqrt((curr_pos[0] - goal_pos[0]) ** 2 + (curr_pos[1] - goal_pos[1]) ** 2) > 1.5 and not rospy.is_shutdown():

        sol = MPC.opti.solve()
        u_vec = sol.value(MPC.U[:, 0])
        ROS.send_velocity(u_vec)
        curr_pos = ROS.get_current_pose()
        MPC.opti.set_value(MPC.r_pos, curr_pos)
        #MPC.animate(curr_pos)

        # for data collection
        covs = ROS.get_current_poseCov()
        scans = ROS.get_current_scan()
        dataset = np.vstack((dataset, np.hstack((curr_pos, scans, covs)).reshape(1, 367)))
        rate.sleep()

# saving dataset into the data_collection folder
#dataset, ind = np.unique(dataset, axis=0, return_index=True)
#dataset = dataset[np.argsort(ind)]
if not os.path.isdir("data_collection"):
    os.makedirs("data_collection")
dataset = np.delete(dataset, 0, 0)
np.savetxt("data_collection/dataset4.csv", dataset, delimiter=",")
print("DATA COLLECTION COMPLETE: PLEASE CLOSE GAZEBO WINDOW THROUGH TERMINAL")