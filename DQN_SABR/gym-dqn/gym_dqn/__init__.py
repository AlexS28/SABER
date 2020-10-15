from gym.envs.registration import register

register(
    id='dqn-v0',
    entry_point='gym_dqn.envs:dqnEnv',
)