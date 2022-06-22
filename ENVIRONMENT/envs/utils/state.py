import torch


class FullState(object):
    def __init__(self, px, py, vx, vy, radius, gx, gy, v_pref, theta):
        self.px = px
        self.py = py
        self.vx = vx
        self.vy = vy
        self.radius = radius
        self.gx = gx
        self.gy = gy
        self.v_pref = v_pref
        self.theta = theta

        self.position = (self.px, self.py)
        self.goal_position = (self.gx, self.gy)
        self.velocity = (self.vx, self.vy)

    def __add__(self, other):
        return other + (self.px, self.py, self.vx, self.vy, self.radius, self.gx, self.gy, self.v_pref, self.theta)

    def __str__(self):
        return ' '.join([str(x) for x in [self.px, self.py, self.vx, self.vy, self.radius, self.gx, self.gy,
                                          self.v_pref, self.theta]])

    def to_tuple(self):
        return self.px, self.py, self.vx, self.vy, self.radius, self.gx, self.gy, self.v_pref, self.theta

    def get_observable_state(self):
        return ObservableState(self.px, self.py, self.vx, self.vy, self.radius)


class ObservableState(object):
    def __init__(self, px, py, vx, vy, radius):
        self.px = px
        self.py = py
        self.vx = vx
        self.vy = vy
        self.radius = radius

        self.position = (self.px, self.py)
        self.velocity = (self.vx, self.vy)

    def __add__(self, other):
        return other + (self.px, self.py, self.vx, self.vy, self.radius)

    def __str__(self):
        return ' '.join([str(x) for x in [self.px, self.py, self.vx, self.vy, self.radius]])

    def to_tuple(self):
        return self.px, self.py, self.vx, self.vy, self.radius


class JointState(object):
    def __init__(self, robot_state, human_states):
        assert isinstance(robot_state, FullState)
        for human_state in human_states:
            assert isinstance(human_state, ObservableState)

        self.robot_state = robot_state
        self.human_states = human_states

    def to_tensor(self, add_batch_size=False, device=None):
        robot_state_tensor = torch.Tensor([self.robot_state.to_tuple()])
        human_states_tensor = torch.Tensor([human_state.to_tuple() for human_state in self.human_states])

        if add_batch_size:
            robot_state_tensor = robot_state_tensor.unsqueeze(0)
            human_states_tensor = human_states_tensor.unsqueeze(0)

        if device == torch.device('cuda:0'):
            robot_state_tensor = robot_state_tensor.cuda()
            human_states_tensor = human_states_tensor.cuda()
        elif device is not None:
            robot_state_tensor.to(device)
            human_states_tensor.to(device)

        return robot_state_tensor, human_states_tensor


def tensor_to_joint_state(state):
    robot_state, human_states = state

    robot_state = robot_state.cpu().squeeze().data.numpy()
    robot_state = FullState(robot_state[0], robot_state[1], robot_state[2], robot_state[3], robot_state[4],
                            robot_state[5], robot_state[6], robot_state[7], robot_state[8])
    human_states = human_states.cpu().squeeze(0).data.numpy()
    human_states = [ObservableState(human_state[0], human_state[1], human_state[2], human_state[3],
                                    human_state[4]) for human_state in human_states]

    return JointState(robot_state, human_states)

def transform(state):
    """
    Take the state passed from agent and transform it to the input of value network

    :param state:
    :return: tensor of shape (# of humans, len(state))
    """
    robot_state_tensor = torch.Tensor([state.robot_state.to_tuple()])
    human_states_tensor = torch.Tensor([human_state.to_tuple() for human_state in state.human_states])

    robot_state_tensor = robot_state_tensor.reshape(9)

    rotated_robot_state, rotated_human_states = rotate_state(robot_state_tensor, human_states_tensor)

    human_num = rotated_human_states.shape[0]
    rotated_human_states = rotated_human_states.reshape(human_num * 7)
    concat_state_tensor = torch.cat((rotated_robot_state, rotated_human_states))

    return concat_state_tensor

def rotate_state(robot_state, human_states):
    """
    Transform the coordinate to agent-centric.
    Local coordinate system has the same rotation as the global one.
    Input state tensor is of size (batch_size, state_length)

    """
    # robot_state
    # 'px', 'py', 'vx', 'vy', 'radius', 'gx', 'gy', 'v_pref', 'theta'
    #  0     1      2     3      4        5     6      7         8
    dx = (robot_state[5]-robot_state[0])
    dy = (robot_state[6]-robot_state[1])
    # rot = torch.atan2(robot_state[6]-robot_state[1],robot_state[5]-robot_state[0])
    # dg = torch.norm(torch.stack((dx, dy)), 2)
    v_pref = robot_state[7]
    vx = robot_state[2]
    vy = robot_state[3]
    radius = robot_state[4]
    theta = robot_state[8]
    new_robot_state = torch.stack((dx, dy, vx, vy, radius, v_pref, theta))

    # human_states
    # 'px', 'py', 'vx', 'vy', 'radius'
    #  0     1      2     3      4
    human_num = human_states.shape[0]
    vx_humans = (human_states[:, 2]).reshape((human_num,-1))
    vy_humans = (human_states[:, 3]).reshape((human_num, -1))
    px_humans = (human_states[:, 0] - robot_state[0].expand(human_num)).reshape((human_num, -1))
    py_humans = (human_states[:, 1] - robot_state[1].expand(human_num)).reshape((human_num, -1))
    radius_humans = human_states[:, 4].reshape((human_num, -1))
    radius_sum = radius.expand(human_num,1) + radius_humans
    # da = distances to each individual human
    da = torch.norm(torch.cat([(robot_state[0].expand(human_num) - human_states[:, 0]).reshape((human_num, -1)),
                               (robot_state[1].expand(human_num) - human_states[:, 1]).reshape((human_num, -1))], dim=1),
                    2, dim=1, keepdim=True)
    new_human_states = torch.cat([px_humans, py_humans, vx_humans, vy_humans, radius_humans, da, radius_sum], dim=1)

    # new_robot_state
    # 'gx', 'gy', 'vx', 'vy', 'radius', 'v_pref', 'theta'
    #  0     1     2     3     4         5         6
    # new_human_states
    # 'px', 'py', 'vx', 'vy', 'radius', 'distance to robot', 'radius_sum'
    #  0     1      2     3      4        5                  6
    return new_robot_state, new_human_states
