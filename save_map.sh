#!/usr/bin/env bash
export TURTLEBOT3_MODEL="waffle"
rosrun map_server map_saver -f /home/alexander/catkin_SABR/src/sabr_pkg/src/SABR/maps/world5
rosnode kill -a; killall -9 rosmaster; killall -9 roscore
roslaunch sabr_pkg sabr_gazebo_collect.launch