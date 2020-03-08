from gym.envs.registration import register

register(
    id='robot-v0',
    entry_point='gym_gazebo.envs:RobotEnv',
)
# register(
#     id='gazebo-extrahard-v0',
#     entry_point='gym_gazebo.envs:FooExtraHardEnv',
# )