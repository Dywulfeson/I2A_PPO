from ENVIRONMENT.envs.utils.agent import Agent
from ENVIRONMENT.envs.utils.state import JointState


class Human(Agent):
    def __init__(self, config, section):
        super().__init__(config, section)
        self.id = None

    def act(self, ob):
        """
        The state for human is its full state and all other agents' observable states
        :param ob:
        :return:
        """
        state = JointState(self.get_full_state(), ob)
        action = self.policy.predict(state)
        return action
