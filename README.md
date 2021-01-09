# Dynamic_Navigation
A Gazebo simulation for robot navigation in dynamics via deep reinforcement learning

### Dependencies
- Ubuntu 16.04
- ROS-kinetic
- Gazebo 8 (with actor suport)

### Build

1. Add the repositories of Gazebo 8 and ROS kinetic

2. Build packages

for tf2 support for python, you must use the following cmd:

    catkin_make --cmake-args \
            -DCMAKE_BUILD_TYPE=Release \
            -DPYTHON_EXECUTABLE=/usr/bin/python3 \
            -DPYTHON_INCLUDE_DIR=/usr/include/python3.5m \
            -DPYTHON_LIBRARY=/usr/lib/x86_64-linux-gnu/libpython3.5m.so

### Test


### Start
1. 启用：


### Problem & solution
#### 1. gazebo中行人collision属性

这个问题困扰我很久, 然后刚巧被同学研究解决了, 原博文 [link](https://blog.csdn.net/tanjia6999/article/details/102629735#commentBox)
1. 下载 [actor_collisions.tar.gz(gazebo 行人碰撞插件 穿墙)](https://download.csdn.net/download/tanjia6999/11879702), 后续考虑整合进该项目. 目前文件保存在`./src/gym_ped_sim/actor_collisions`.
2. 在下载好的目录下编译, 生成.so文件.
```shell
cmake .
make 
```
3. 测试是否成功:
```shell
export GAZEBO_PLUGIN_PATH=${GAZEBO_PLUGIN_PATH}:~/Documents/actor_collisions
gazebo actor_collisions.world
```
如果出现了行人, 并且可以右键查看到collision橙色. 说明成功了.

4. 应用到项目:
- 在~/.zshrc里面 `export GAZEBO_PLUGIN_PATH=${GAZEBO_PLUGIN_PATH}:.../actor_collisions`
- 在含行人的`.world`文件中, 需要在`<actor>`标签下添加`<plugin>`内容来实现`collision`的效果, 具体内容查看 `actor_collisions.world` 文件内容.
```shell
    <plugin name="actor_collisions_plugin" filename="libActorCollisionsPlugin.so">
        ......
        </plugin>
```
- 更改`laser`插件内容, 需要带gpu的雷达才能在gazebo中实现对行人collision的检测:

将`<sensor type="ray" name="laser_sensor">` -> `<sensor type="gpu_ray" name="laser_sensor">`.

讲`<plugin name="gazebo_ros_head_hokuyo_controller" filename="libgazebo_ros_laser.so">` -> `<plugin name="gazebo_ros_head_hokuyo_controller" filename="libgazebo_ros_gpu_laser.so">.
`
