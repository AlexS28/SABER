from MPC_ugv_Simple import *
import time
import os


# initialize all required variables for the MPC solver
dT = 0.1
mpc_horizon = 5
curr_pos = np.array([0, 0, 0]).reshape(3,1)
goal_points = [[10, 10, 0], [0, 0, 0]]
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

ROS = ROSInterface()
rospy.init_node('ros_interface')
rate = rospy.Rate(10)

# start data collection
# initiate data collection list
dataset = np.zeros((1, 364))
time.sleep(5)
for i in range(0, len(goal_points)):

    goal_pos = np.array(goal_points[i])
    MPC.opti.set_value(MPC.r_goal, goal_pos)

    while m.sqrt((curr_pos[0] - goal_pos[0]) ** 2 + (curr_pos[1] - goal_pos[1]) ** 2) > 1 and not rospy.is_shutdown():

        sol = MPC.opti.solve()
        u_vec = sol.value(MPC.U[:, 0])
        ROS.send_velocity(u_vec)
        curr_pos = ROS.get_current_pose()
        MPC.opti.set_value(MPC.r_pos, curr_pos)
        #MPC.animate(curr_pos)

        # for data collection
        covs = ROS.get_current_poseCov()
        scans = ROS.get_current_scan()
        dataset = np.vstack((dataset, np.hstack((scans, covs)).reshape(1, 364)))
        rate.sleep()

# saving dataset into the data_collection folder
#dataset, ind = np.unique(dataset, axis=0, return_index=True)
#dataset = dataset[np.argsort(ind)]
dataset = np.delete(dataset, 0, 0)
np.savetxt("data_collection/dataset1.csv", dataset, delimiter=",")
print("DATA COLLECTION COMPLETE: PLEASE CLOSE GAZEBO WINDOW THROUGH TERMINAL")