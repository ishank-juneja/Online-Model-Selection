from dm_control import composer
from dm_control.composer.observation import observable
from dm_control import mjcf
from dm_control.rl.control import Environment
from math import sqrt, pi, sin, cos
import matplotlib.pyplot as plt
from random import randint
import numpy as np


class FallingPolygonTask(composer.NullTask):
    """Task to wrap around falling polygon environment"""
    def __init__(self, shape):
        super().__init__(shape)
        # - - - - - - - Configure Observables - - - - - - -
        self._root_entity.observables.global_pos_xy.enabled = True
        self._root_entity.observables.global_vel_xy.enabled = True


# PyMJCF model for random polygons: size and nvertices, freely falling under gravity
class FallingPolygon(composer.Entity):
    """A regular polygon (without fill) with 3-8 vertices that falls freely under gravity"""
    def _build(self):
        self._model = mjcf.RootElement(model="regular-polygon")
        # - - - - - - - MJCF Global Options - - - - - - -
        self._model.compiler.coordinate = 'local'
        self._model.compiler.inertiafromgeom = 'true'
        self._model.visual.__getattr__('global').offheight = 512
        # self._mjcf_root.visual.headlight.set_attributes(
        #     ambient=[.4, .4, .4], diffuse=[.8, .8, .8], specular=[.1, .1, .1])
        # - - - - - - - - - - - - - - - - - - - - - - -

        # - - - - - - - MJCF Defaults - - - - - - -
        self._model.default.joint.damping = 0.2
        self._model.default.geom.contype = 3
        self._model.default.geom.conaffinity = 3
        self._model.default.geom.friction = [0., 0., 0.]
        self._model.default.geom.rgba = [1.0, 1.0, 1.0, 1.0]
        # - - - - - - - - - - - - - - - - - - - - - - -

        # - - - - - - - MJCF Options - - - - - - -
        self._model.option.gravity = [0.0, 0.0, -9.81]
        # - - - - - - - - - - - - - - - - - - - - - - -

        # - - - - - - - MJCF Memory Size Options - - - - - - -
        self._model.size.nstack = 3000
        # - - - - - - - - - - - - - - - - - - - - - - -

        # - - - - - - - Common Arena Type Params - - - - - - - - -
        # Fixed camera that looks head on onto the x-z plane
        sqrt2_inv = 1 / sqrt(2)
        self._model.worldbody.add('camera', name="myrgb", mode="fixed", pos="0.0 -4 0.0", quat=[sqrt2_inv, sqrt2_inv, 0., 0.])
        # - - - - - - - - - - - - - - - - - - - - - - -

        # - - - - - - - - -  Body in scene - - - - - - - - - -
        # Starting location for the polygon shape
        # TODO: Make these uniform random between -1.7 and 1.7
        # Random location of end pt of first edge
        vx = np.random.uniform(-1.5, 1.5)
        vz = np.random.uniform(-1.5, 1.5)
        # Random number of edges
        nedges = randint(3, 8)
        # Random 2D rotation in xz plane about y axisas a quat
        rand_theta = np.random.uniform(0, 2*pi)
        rand_quat = [cos(rand_theta/2), 0, sin(rand_theta/2), 0]
        # Random semi edge length of capsules composing shapes
        semi_edgel = np.random.uniform(0.2, 0.5)
        # Random cap radius of capsules composing shapes
        cap_rad = np.random.uniform(0.05, 0.17)
        # Random geom color, avoid dark colors
        rand_rgba_nums = np.random.uniform(0.5, 1, 3)
        rand_rgba = [rand_rgba_nums[0], rand_rgba_nums[1], rand_rgba_nums[2], 1.0]
        self.first_edge_body = self.make_polygon_geom(self._model.worldbody, nedges, vx, vz, inital_quat=rand_quat,
                                                      semi_edgel=semi_edgel, cap_rad=cap_rad, rand_rgba=rand_rgba)
        # Add free joint to first edge attached to worldbody to allow freefall
        self.first_edge_body.add('joint', name="dummy", type='free')

    def _build_observables(self):
        return FallingPolygonObservables(self)

    @property
    def mjcf_model(self):
        return self._model

    # Adds a n-vertexed shape outline to the passed body
    def make_polygon_geom(self, parent_body, nedges: int, vertex_x: float, vertex_z: float,
                     inital_quat: list = (1.0, 0., 0., 0.), semi_edgel: float=0.4, cap_rad: float=0.15,
                          rand_rgba=(1, 1, 1, 1)):
        # Angle that the new body has to be rotated along y-axis relative to its parent body
        theta_y = 2 * pi / nedges
        # Position of center of next edge in the x-z plane
        next_vx = vertex_x
        next_vz = vertex_z
        next_quat = list(inital_quat)
        # Parent body onto which we want to add the next edge
        cur_parent_body = parent_body
        # Return the first edge back to attach free joint
        first_edge_body = None
        for idx in range(nedges):
            # Add the edge as a new body with a coordinate frame aligned
            cur_parent_body = cur_parent_body.add('body', name="edge{0}".format(idx + 1),
                                              pos=[next_vx, 0., next_vz], quat=next_quat)
            cur_parent_body.add('geom', type="capsule", rgba=rand_rgba, size=[cap_rad], fromto=[0, 0, 0, 0, 0, 2*semi_edgel])
            # Obtain next coordinates and orientation in cur_parent_body frame
            next_vx = 0.0
            next_vz = 2*semi_edgel
            next_quat = [cos(theta_y/2), 0., sin(theta_y/2), 0.]
            if idx == 0:
                first_edge_body = cur_parent_body
        return first_edge_body


class FallingPolygonObservables(composer.Observables):
    """Position and Velocity observables for FallingPolygon"""
    @composer.observable
    def global_pos_xy(self):
        free_joint = self._entity.first_edge_body.joint["dummy"]
        return observable.MJCFFeature('qpos', free_joint)

    @composer.observable
    def global_vel_xy(self):
        free_joint = self._entity.first_edge_body.joint["dummy"]
        return observable.MJCFFeature('qvel', free_joint)

    # @composer.observable
    # def image_frame(self, physics):
    #     free_joint = self._entity.first_edge_body.joint["dummy"]
    #     return observable.MJCFFeature('qvel', free_joint)

#
# if __name__ == '__main__':
#     shape = FallingPolygon()
#     task = FallingPolygonTask(shape)
#
#     env = composer.Environment(task)
#
#     spec = env.action_spec()
#     time_step = env.reset()
#
#     while not time_step.last():
#         action = np.random.uniform(spec.minimum, spec.maximum, spec.shape)
#         time_step = env.step(action)
#         foo = True

# if __name__ == '__main__':
#     duration = 10  # (Seconds)
#     framerate = 30  # (Hz)
#     video = []
#     pos_x = []
#     pos_y = []
#
#     physics = mjcf.Physics.from_mjcf_model(FallingPolygon().mjcf_model)
#
#     # Simulate, saving video frames and torso locations.
#     physics.reset()
#     while physics.data.time < duration:
#         physics.step()
#
#         # Save video frames.
#         if len(video) < physics.data.time * framerate:
#             pixels = physics.render(height=64, width=64, camera_id=0, depth=False)
#             video.append(pixels.copy())
#             plt.imshow(pixels)
#             plt.show()


if __name__ == '__main__':
    shape = FallingPolygon()

    # physics = print(mjcf.Physics.from_mjcf_model(shape.mjcf_model.to_xml()))

    physics = mjcf.Physics.from_mjcf_model(shape.mjcf_model)

    # print(shape.mjcf_model.to_xml_string())

    pixels = physics.render(height=64, width=64, camera_id=0, depth=False)

    plt.imshow(pixels)
    plt.show()
