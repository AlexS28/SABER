cmake_minimum_required(VERSION 2.8.3)
project(sabr_pkg)

find_package(catkin REQUIRED COMPONENTS
  message_generation
  rospy
  geometry_msgs
  nav_msgs
  sensor_msgs
  tf
  )

find_package(gazebo REQUIRED)

catkin_python_setup()

#add_message_files(
#  FILES
#  type1.msg
#  type2.msg
#)
#add_service_files(
#  FILES
#  type3.srv
#)
#generate_messages(
#  DEPENDENCIES
#  geometry_msgs
#)

catkin_package(
  CATKIN_DEPENDS geometry_msgs
                 message_runtime
                 rospy
                 nav_msgs
                 sensor_msgs
                 tf)

include_directories(
  ${catkin_INCLUDE_DIRS}
  )

install(FILES launch/sabr_gazebo_collect.launch
              launch/sabr_gazebo_drone.launch
              launch/sabr_gazebo_map.launch
              launch/sabr_gazebo_ugv_uav.launch
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}/launch)


install(DIRECTORY worlds rviz
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}
  PATTERN "*~" EXCLUDE)

catkin_install_python(PROGRAMS scripts/play2.py
                               scripts/play3.py
                               scripts/SABER.py
                               src/sabr_pkg/MPC_ugv_Simple.py
                               src/sabr_pkg/ROS_interface.py
                               src/sabr_pkg/ROS_interface_ugv_uav.py
                               DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION})
