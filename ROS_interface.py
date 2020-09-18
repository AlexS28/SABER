# Example Code
# https://roboticsbackend.com/oop-with-ros-in-python/
# https://roslibpy.readthedocs.io/en/latest/examples.html#first-connection
# http://wiki.ros.org/ROS/Tutorials/WritingPublisherSubscriber%28python%29

import rospy

from geometry_msgs.msg import Twist
from geometry_msgs.msg import Pose
from nav_msgs.msg import Odometry

from tf.transformations import euler_from_quaternion

class ROSInterface:

    def __init__(self):
        self.current_pose = Odometry()
        # TODO: Make them remappable
        self.pub_vel = rospy.Publisher("/mobile_base/commands/velocity", Twist, queue_size=10) #/cmd/vel
        self.sub_pose = rospy.Subscriber("/odom_robot", Odometry, self.receive_pose) # /pose_in

    # Send MPC controls trough ROS to Gazebo simulation or real hardware
    def send_velocity(self, u_vec):
        new_msg = Twist()
        # U from MPC solution
        new_msg.linear.x = u_vec[0]
        new_msg.linear.y = u_vec[1]
        new_msg.angular.z = u_vec[2]
        # Constant, not used in 2D case
        new_msg.linear.z = 0
        new_msg.angular.x = 0
        new_msg.angular.y = 0
        # Publish U from MPC as /cmd/vel
        self.pub_vel.publish(new_msg)

    # Process pose measurement from Gazebo simulation or tracking system
    # def receive_pose(self, msg):
    #     self.current_pose = msg
    def receive_pose(self, msg):
        self.current_pose = msg

    # Convert pose to MPC state format
    def get_current_pose(self):
        euler_angles = euler_from_quaternion([self.current_pose.pose.pose.orientation.x, self.current_pose.pose.pose.orientation.y, self.current_pose.pose.pose.orientation.z, self.current_pose.pose.pose.orientation.w])
        return [self.current_pose.pose.pose.position.x, self.current_pose.pose.pose.position.y, euler_angles[2]]