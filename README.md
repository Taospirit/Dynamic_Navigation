# Dynamic_Navigation
A Gazebo simulation for robot navigation in dynamics via deep reinforcement learning

### Dependencies
- Ubuntu 16.04
- ROS-kinetic
- Gazebo 9 (with actor suport)
- turtlebot2

### Build

1. Add the repositories of Gazebo 8 and ROS kinetic

2. Build packages

for tf2 support for python, you must use the following cmd:

    catkin_make --cmake-args \
            -DCMAKE_BUILD_TYPE=Release \
            -DPYTHON_EXECUTABLE=/usr/bin/python3 \
            -DPYTHON_INCLUDE_DIR=/usr/include/python3.5m \
            -DPYTHON_LIBRARY=/usr/lib/x86_64-linux-gnu/libpython3.5m.so