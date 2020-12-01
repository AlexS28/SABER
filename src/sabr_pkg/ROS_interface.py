# Example Code
# https://roboticsbackend.com/oop-with-ros-in-python/
# https://roslibpy.readthedocs.io/en/latest/examples.html#first-connection
# http://wiki.ros.org/ROS/Tutorials/WritingPublisherSubscriber%28python%29

import rospy
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from geometry_msgs.msg import PoseWithCovarianceStamped
from tf.transformations import euler_from_quaternion
from sensor_msgs.msg import LaserScan
from geometry_msgs.msg import PoseStamped
from geometry_msgs.msg import Vector3Stamped
import math
import numpy as np
from hector_uav_msgs.srv import EnableMotors
from xivo_ros.msg import FeatureMap, FeatureData



class ROSInterface:

    def __init__(self, ugv, num_slam_features=15):
        if not ugv:
            motors_on = rospy.ServiceProxy('enable_motors', EnableMotors)
            motors_on.call(1)
        self.current_pose = Odometry()
        self.current_poseEulerUAV = Vector3Stamped()
        self.current_poseVelUAV = Odometry()
        self.current_posCov = PoseWithCovarianceStamped()
        self.current_scan = LaserScan()
        self.current_poseUAV = PoseStamped()

        self.num_slam_features = num_slam_features
        self.current_slam_map = FeatureMap()
        self.current_slam_pose = PoseWithCovarianceStamped()

        # TODO: Make them remappable
        self.pub_vel = rospy.Publisher("/cmd_vel", Twist, queue_size=10) #/cmd/vel
        self.sub_pose = rospy.Subscriber("odom", Odometry, self.receive_pose) #/pose_in
        self.sub_poseUAV = rospy.Subscriber("ground_truth_to_tf/pose", PoseStamped, self.receive_poseUAV)
        self.sub_poseEulerUAV = rospy.Subscriber("ground_truth_to_tf/euler", Vector3Stamped, self.receive_poseEulerUAV)
        self.sub_poseVelUAV = rospy.Subscriber("ground_truth/state", Odometry, self.receive_poseVelUAV)
        self.sub_posCov = rospy.Subscriber("amcl_pose", PoseWithCovarianceStamped, self.receive_posCov)
        self.sub_scan = rospy.Subscriber("scan", LaserScan, self.receive_scan)
        self.xivo_map_sub = rospy.Subscriber("xivo/map", FeatureMap, self.receive_xivomap)
        self.xivo_state_sub = rospy.Subscriber("xivo/pose", PoseWithCovarianceStamped, self.receive_xivostate)

    # Send MPC controls trough ROS to Gazebo simulation or real hardware
    def send_velocity(self, u_vec):
        new_msg = Twist()
        # U from MPC solution (Turtlebot3 can only handle 2 control values, x velocity, and angular velocity)
        new_msg.linear.x = u_vec[0]
        new_msg.angular.z = u_vec[1]
        # Constant, not used in 2D case
        new_msg.linear.z = 0
        new_msg.angular.x = 0
        new_msg.angular.y = 0
        # Publish U from MPC as /cmd/vel
        self.pub_vel.publish(new_msg)

    def send_velocityUAV(self, u_vec):
        new_msg = Twist()
        new_msg.linear.x = u_vec[0]
        new_msg.linear.y = u_vec[1]
        new_msg.linear.z = u_vec[2]
        new_msg.angular.x = u_vec[3]
        new_msg.angular.y = u_vec[4]
        new_msg.angular.z = 0
        self.pub_vel.publish(new_msg)

    # Process pose measurement from Gazebo simulation or tracking system from ODOM
    def receive_pose(self, msg):
        self.current_pose = msg

    # Process pose measurement from Gazebo simulation or tracking system from Pose
    def receive_poseUAV(self, msg):
        self.current_poseUAV = msg

    # Process pose measurement from Gazebo simulation or tracking system from Pose
    def receive_poseEulerUAV(self, msg):
        self.current_poseEulerUAV = msg

    # Process pose measurement from Gazebo simulation or tracking system from Pose
    def receive_poseVelUAV(self, msg):
        self.current_poseVelUAV = msg

    # Process pose and cov measurement from Gazebo simulation from SLAM
    def receive_posCov(self, msg):
        self.current_posCov = msg

    def receive_scan(self, msg):
        self.current_scan = msg

    def receive_xivomap(self, msg):
        self.current_slam_map = msg

    def receive_xivostate(self, msg):
        self.current_slam_pose = msg

    # Convert pose to MPC state format
    def get_current_pose(self):
        euler_angles = euler_from_quaternion([self.current_pose.pose.pose.orientation.x, self.current_pose.pose.pose.orientation.y, self.current_pose.pose.pose.orientation.z, self.current_pose.pose.pose.orientation.w])
        return [self.current_pose.pose.pose.position.x, self.current_pose.pose.pose.position.y, euler_angles[2]]

    def get_current_poseCovUAV(self):
        cov = self.current_slam_pose.pose.covariance  # 36 long array
        cov = np.array(cov)
        cov = np.reshape(cov, (6,6))
        cov_Tsb = cov[3:5,3:5]
        return cov_Tsb.flatten().tolist()

    def get_current_poseUAV_XIVO(self):
        Tsb = self.current_slam_pose.pose.pose.position
        return [ Tsb.x, Tsb.y, Tsb.z ]

    def get_features(self):
        num_features = min(self.num_slam_features,
                           self.current_slam_map.num_features)
        feature_pos = np.zeros((num_features,3))
        for i in range(num_features):
            pos = self.current_slam_map.features[i].Xs
            feature_pos[i,0] = pos.x
            feature_pos[i,1] = pos.y
            feature_pos[i,2] = pos.z
        return feature_pos.flatten()

    def get_current_poseUAV(self):
        x_pos = self.current_poseUAV.pose.position.x
        y_pos = self.current_poseUAV.pose.position.y
        z_pos = self.current_poseUAV.pose.position.z
        return [x_pos, y_pos, z_pos]

    def get_current_poseEulerUAV(self):
        x_ang = self.current_poseEulerUAV.vector.x
        y_ang = self.current_poseEulerUAV.vector.y
        z_ang = self.current_poseEulerUAV.vector.z
        return [x_ang, y_ang, z_ang]

    def get_current_poseVelUAV(self):
        x_vel = self.current_poseVelUAV.twist.twist.linear.x
        y_vel = self.current_poseVelUAV.twist.twist.linear.y
        z_vel = self.current_poseVelUAV.twist.twist.linear.z
        x_ang_vel = self.current_poseVelUAV.twist.twist.angular.x
        y_ang_vel = self.current_poseVelUAV.twist.twist.angular.y
        z_ang_vel = self.current_poseVelUAV.twist.twist.angular.z
        return [x_vel, y_vel, z_vel, x_ang_vel, y_ang_vel, z_ang_vel]

    # Convert Pose Cov to MPC state format
    def get_current_poseCov(self):
        if self.current_posCov.pose.covariance[0] >= 1:
            xx = 1
        else:
            xx = self.current_posCov.pose.covariance[0]
        if self.current_posCov.pose.covariance[1] >= 1:
            xy = 1
        else:
            xy = self.current_posCov.pose.covariance[1]
        if self.current_posCov.pose.covariance[6] >= 1:
            yx = 1
        else:
            yx = self.current_posCov.pose.covariance[6]
        if self.current_posCov.pose.covariance[7] >= 1:
            yy = 1
        else:
            yy = self.current_posCov.pose.covariance[7]

        return [xx, xy, yx, yy]

    # Convert Scan to Data collection format (convert inf, nan, to max value of 4)
    def get_current_scan(self):
        scans = list(self.current_scan.ranges)

        if not scans:
            scans = np.zeros((360,), dtype=float)
            scans = scans.tolist()
        else:
            for i in range(0, len(scans)):
                if math.isinf(scans[i]) or np.isnan(scans[i]):
                    scans[i] = float(4)
                # scans = sorted(scans)
        return scans