from ENVIRONMENT.envs.policy.policy import Policy

class Placeholder(Policy):
    def __init__(self):
        super(Placeholder, self).__init__()
        self.kinematics = 'holonomic'
        self.multiagent_training = True
