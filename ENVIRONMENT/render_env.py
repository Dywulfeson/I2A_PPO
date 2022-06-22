import gym

# import env
from ENVIRONMENT.envs.utils.robot import Robot
from ENVIRONMENT.envs.utils.state import transform
import ENVIRONMENT.configs.i2a as i2a_config
# import placeholder policy so that env doesnt produce errors
from ENVIRONMENT.envs.policy.placeholder import Placeholder
from ENVIRONMENT.envs.utils.state import JointState
from ENVIRONMENT.envs.utils.info import *
from ENVIRONMENT.envs.utils.action import ActionXY

# Instantiate environment
env_config = i2a_config.EnvConfig(False)
env = gym.make('CrowdSim-v0')
env.configure(env_config)
robot = Robot(env_config, 'robot')
robot.time_step = env.time_step
env.set_robot(robot)
policy = Placeholder()
robot.set_policy(policy)

obs = env.reset()
for i in range(40):
    action = ActionXY(0,0.5)
    obs = env.step(action)
env.render(mode='traj')

print('nice')