from ENVIRONMENT.envs.policy.linear import Linear
from ENVIRONMENT.envs.policy.orca import ORCA, CentralizedORCA
from ENVIRONMENT.envs.policy.socialforce import SocialForce, CentralizedSocialForce



def none_policy():
    return None


policy_factory = dict()
policy_factory['linear'] = Linear
policy_factory['orca'] = ORCA
policy_factory['socialforce'] = SocialForce
policy_factory['centralized_orca'] = CentralizedORCA
policy_factory['centralized_socialforce'] = CentralizedSocialForce
policy_factory['none'] = none_policy
