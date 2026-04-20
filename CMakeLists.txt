cmake_minimum_required(VERSION 3.0.2)
project(p2_goldberga_seferie)

find_package(catkin REQUIRED COMPONENTS
  rospy
  std_msgs
  sensor_msgs
  geometry_msgs
  cv_bridge
)

catkin_package()

catkin_install_python(PROGRAMS
  script/ball_chase.py
  script/goal_sequence.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)
