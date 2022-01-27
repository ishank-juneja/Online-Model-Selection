from gym.envs.registration import register

# Register all the used gym environments
register(
    id='Conkers-v0',
    entry_point='gym_cenvs.envs:ConkersEnv',
)

register(
    id='MujocoCartpole-v0',
    entry_point='gym_cenvs.envs:MujocoCartPoleEnv',
)

register(
    id='MujocoBall-v0',
    entry_point='gym_cenvs.envs:MujocoBall',
)
