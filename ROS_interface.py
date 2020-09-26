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
import math
import numpy as np

class ROSInterface:

    def __init__(self):
        self.current_pose = Odometry()
        self.current_posCov = PoseWithCovarianceStamped()
        self.current_scan = LaserScan()

        # TODO: Make them remappable
        self.pub_vel = rospy.Publisher("/cmd_vel", Twist, queue_size=10) #/cmd/vel
        self.sub_pose = rospy.Subscriber("/odom", Odometry, self.receive_pose) #/pose_in
        self.sub_posCov = rospy.Subscriber("/amcl_pose", PoseWithCovarianceStamped, self.receive_posCov)
        self.sub_scan = rospy.Subscriber("/scan", LaserScan, self.receive_scan)

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

    # Process pose measurement from Gazebo simulation or tracking system from ODOM
    def receive_pose(self, msg):
        self.current_pose = msg

    # Process pose and cov measurement from Gazebo simulation from SLAM
    def receive_posCov(self, msg):
        self.current_posCov = msg

    def receive_scan(self, msg):
        self.current_scan = msg

    # Convert pose to MPC state format
    def get_current_pose(self):
        euler_angles = euler_from_quaternion([self.current_pose.pose.pose.orientation.x, self.current_pose.pose.pose.orientation.y, self.current_pose.pose.pose.orientation.z, self.current_pose.pose.pose.orientation.w])
        return [self.current_pose.pose.pose.position.x, self.current_pose.pose.pose.position.y, euler_angles[2]]

    # Convert Pose Cov to MPC state format
    def get_current_poseCov(self):
        return [self.current_posCov.pose.covariance[0], self.current_posCov.pose.covariance[1], self.current_posCov.pose.covariance[6], self.current_posCov.pose.covariance[7]]

    # Convert Scan to Data collection format (order scans from least to greatest, and convert inf, nan, to zeros)
    def get_current_scan(self):
        scans = list(self.current_scan.ranges)

        if not scans:
            scans = np.zeros((360,), dtype=float)
            scans = scans.tolist()
        else:
            for i in range(0, len(scans)):
                if math.isinf(scans[i]) or scans[i] <= 10**-5 or np.isnan(scans[i]):
                    scans[i] = float(0)
            scans = sorted(scans)
        return scans
