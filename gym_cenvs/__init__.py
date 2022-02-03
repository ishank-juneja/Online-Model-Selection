from gym.envs.registration import register

# Register all the used gym environments
register(
    id='Conkers-v0',
    entry_point='gym_cenvs.envs:ConkersEnv',
)

register(
    id='Kendama-v0',
    entry_point='gym_cenvs.envs:KendamaEnv',
)

register(
    id='Conkers-v1',
    entry_point='gym_cenvs.envs:ConkersEnv',
    kwargs={'transparent_rope': True}
)

register(
    id='MujocoCartpole-v0',
    entry_point='gym_cenvs.envs:MujocoCartPoleEnv',
)

register(
    id='MujocoDoubleCartpole-v0',
    entry_point='gym_cenvs.envs:MujocoDoubleCartPoleEnv',
)

register(
    id='MujocoBall-v0',
    entry_point='gym_cenvs.envs:MujocoBall',
)
