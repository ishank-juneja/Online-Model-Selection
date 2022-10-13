from dm_control import mjcf


def check_body_collision(physics: mjcf.Physics, body1: str, body2: str):
    """
    Check whether the given two bodies have collision. The given body name can be either child body or parent body
    *NOTICE*: this method may be unsafe and cause hidden bug, since the way of retrieving body is to check whether
    the given name is a sub-string of the collision body
    :param physics: a MuJoCo physics engine
    :param body1: the name of body1
    :param body2: the name of body2
    :return collision: a bool variable
    """
    collision = False
    for geom1, geom2 in zip(physics.data.contact.geom1, physics.data.contact.geom2):
        bodyid1 = physics.model.geom_bodyid[geom1]
        bodyid2 = physics.model.geom_bodyid[geom2]
        bodyname1 = physics.model.id2name(bodyid1, 'body')
        bodyname2 = physics.model.id2name(bodyid2, 'body')
        if (body1 in bodyname1 and body2 in bodyname2) or (body2 in bodyname1 and body1 in bodyname2):
            collision = True
            break
    return collision

