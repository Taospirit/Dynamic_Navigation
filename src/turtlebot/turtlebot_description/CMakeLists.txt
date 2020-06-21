cmake_minimum_required(VERSION 2.8.3)
project(turtlebot_description)

find_package(catkin REQUIRED COMPONENTS)

catkin_package(
#  INCLUDE_DIRS include
#  LIBRARIES turtlebot_description
#  CATKIN_DEPENDS rospkg rospy
#  DEPENDS system_lib
)

include_directories(
  ${catkin_INCLUDE_DIRS}
)
