from ENVIRONMENT.envs.utils.agent import Agent
from ENVIRONMENT.envs.utils.state import JointState


class Robot(Agent):
    def __init__(self, config, section):
        super().__init__(config, section)

    def act(self, ob):
        if self.policy is None:
            raise AttributeError('Policy attribute has to be set!')

        state = JointState(self.get_full_state(), ob)
        action = self.policy.predict(state)
        return action

    def act_ppo(self, ob):
        if self.policy is None:
            raise AttributeError('Policy attribute has to be set!')

        state = JointState(self.get_full_state(), ob)
        action, value, logp, act_id = self.policy.predict(state, rl=True)
        return action, value, logp, act_id
