from collections import namedtuple
import numpy as np
import torch

ActionXY = namedtuple('ActionXY', ['vx', 'vy'])
ActionRot = namedtuple('ActionRot', ['v', 'r'])


def build_action_space(speed_samples, rotation_samples, kinematics):
    """
        Build the actions for all categorical actions chosen by the robot
    """

    holonomic = True if kinematics == 'holonomic' else False
    # speeds = [(np.exp((i + 1) / self.speed_samples) - 1) / (np.e - 1) * v_pref for i in range(self.speed_samples)]
    delta = 1 / speed_samples
    speeds = [(1 + i) * delta for i in range(speed_samples)]

    if holonomic:
        rotations = np.linspace(0, 2 * np.pi, rotation_samples, endpoint=False)
    else:
        raise NotImplementedError()
        # rotations = np.linspace(-rotation_constraint, rotation_constraint, rotation_samples)

    action_space = [ActionXY(0, 0) if holonomic else ActionRot(0, 0)]
    for j, speed in enumerate(speeds):
        if j == 0:
            # index for action (0, 0)
            # self.action_group_index.append(0)
            pass
        # only two groups in speeds
        if j < 3:
            speed_index = 0
        else:
            speed_index = 1

        for i, rotation in enumerate(rotations):
            rotation_index = i // 2

            # action_index = speed_index * rotation_samples + rotation_index
            # self.action_group_index.append(action_index)

            if holonomic:
                action_space.append(ActionXY(speed * np.cos(rotation), speed * np.sin(rotation)))
            else:
                action_space.append(ActionRot(speed, rotation))

    # self.speeds = speeds
    # self.rotations = rotations
    return action_space

def build_torch_action_space(speed_samples, rotation_samples, kinematics):
    """
            Build the actions for all categorical actions chosen by the robot
    """

    holonomic = True if kinematics == 'holonomic' else False
    # speeds = [(np.exp((i + 1) / self.speed_samples) - 1) / (np.e - 1) * v_pref for i in range(self.speed_samples)]
    speeds = torch.linspace(0,1,speed_samples+1)

    if holonomic:
        rotations = torch.linspace(0, 2 * np.pi, rotation_samples+1) # +1 because torch includes last element. it gets neglected in counting
    else:
        raise NotImplementedError()
        # rotations = np.linspace(-rotation_constraint, rotation_constraint, rotation_samples)

    action_space = torch.zeros((speed_samples*rotation_samples+1, 2), dtype=torch.float64)

    for j in range(1,speeds.shape[0]):
        index = 1 % j * rotation_samples + 1
        for i in range(0, rotations.shape[0]-1):
            if holonomic:
                action_space[index+i][0] = speeds[j] * torch.cos(rotations[i])
                action_space[index+i][1] = speeds[j] * torch.sin(rotations[i])
                # action_space.append(ActionXY(speed * np.cos(rotation), speed * np.sin(rotation)))
            else:
                raise NotImplementedError()

    return action_space