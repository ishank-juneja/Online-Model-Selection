"""
Register the py_mjcf format models as gym environments
v-1 and v-2 of Mujoco models were used for inter-model relationships and
are now depracated
"""
from gym.envs.registration import register

register(
    id='Tmp-v0',
    entry_point='gym_cenvs.envs:Tmp',
)


# Register LVSPC complex model MuJoCo environments

register(
    id='Conkers-v0',
    entry_point='gym_cenvs.envs:Conkers',
)

register(
    id='Kendama-v0',
    entry_point='gym_cenvs.envs:Kendama',
)

register(
    id='Catching-v0',
    entry_point='gym_cenvs.envs:Catching',
)

# Register simple model environments used to create visual models (percpetion system)
#  Prefix "Mujoco" used to avoid namespace collision with in built gym envs

# Vanilla Doublecartpole version
register(
    id='MujocoDoublecartpole-v0',
    entry_point='gym_cenvs.envs:Doublecartpole'
)

# Doublecartpole version with all parts visible but state only returned for ball
register(
    id='MujocoDoublecartpole-v1',
    entry_point='gym_cenvs.envs:Doublecartpole',
    kwargs={'for_ball': True}
)

# Doublecartpole version with only ball visible and only ball state returned
#  for getting ball masks out of cartpole for example
register(
    id='MujocoDoublecartpole-v2',
    entry_point='gym_cenvs.envs:Doublecartpole',
    kwargs={'for_ball': True, 'invisible_rods': True}
)

# Original cartpole version
register(
    id='MujocoCartpole-v0',
    entry_point='gym_cenvs.envs:Cartpole'
)

# Cartpole version with all parts visible but state only returned for ball
register(
    id='MujocoCartpole-v1',
    entry_point='gym_cenvs.envs:Cartpole',
    kwargs={'for_ball': True}
)

# Cartpole version with only ball visible and only ball state returned
#  for getting ball masks out of cartpole for example
register(
    id='MujocoCartpole-v2',
    entry_point='gym_cenvs.envs:Cartpole',
    kwargs={'for_ball': True, 'invisible_rods': True}
)

# Freely falling ball model
register(
    id='MujocoBall-v0',
    entry_point='gym_cenvs.envs:Ball',
)

# Kinematic (Dubins) car model
register(
    id='MujocoDubins-v0',
    entry_point='gym_cenvs.envs:Dubins',
)

