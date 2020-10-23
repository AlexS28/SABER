from gym.envs.registration import register

register(
    id='dqnprev-v0',
    entry_point='gym_dqnprev.envs:dqnprevEnv',
)