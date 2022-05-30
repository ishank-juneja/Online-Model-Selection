import gym
import gym_cenvs
from src.perception_tests import BaseVizTest


class KendamaVizTest(BaseVizTest):
    def __init__(self, enc_model_name: str, seg_model_name: str = None):
        super(KendamaVizTest, self).__init__(enc_model_name=enc_model_name, seg_model_name=seg_model_name)

        # Create a dir for caching these results
        self.dir_manager.add_location('vid_results', 'results/videos/perception/{0}_kendama'.format(self.model.model_name))

        # Create a dir for saving the frames that make up the GIFs
        self.dir_manager.add_location('tmp', 'results/videos/perception/{0}_kendama/tmp'.format(self.model.model_name))

        # Create and seed kendama environment object
        self.env_name = "Kendama-v0"

        # Env needed for finding dt even if env not stepped through
        #  Whether env will be stepped through or not for test depends on
        #  param self.env.dt
        self.env = gym.make(self.env_name)
        self.env.seed(0)
        self.env.action_space.seed(0)

        # Set the dt for plotting
        self.viz.set_delta_t(self.env.dt)


class ConkersVizTest(BaseVizTest):
    def __init__(self, enc_model_name: str, seg_model_name: str = None):
        super(ConkersVizTest, self).__init__(enc_model_name=enc_model_name, seg_model_name=seg_model_name)

        # Create a dir for caching these results
        self.dir_manager.add_location('vid_results',
                                      'results/videos/perception/{0}_conkers'.format(self.model.model_name))

        # Create a dir for saving the frames that make up the GIFs
        self.dir_manager.add_location('tmp', 'results/videos/perception/{0}_conkers/tmp'.format(self.model.model_name))

        # TODO: Replace the Kendama env frames with the conkers environment stuff ...

        # Create and seed Conkers environment object
        self.env_name = "Conkers-v0"

        self.env = gym.make(self.env_name)
        self.env.seed(0)
        self.env.action_space.seed(0)

        # Whether to use env frames or Kendama frames
        self.use_env = True

        # Set the dt for plotting
        self.viz.set_delta_t(self.env.dt)


class BallVizTest(BaseVizTest):
    def __init__(self, enc_model_name: str, seg_model_name: str = None):
        super(BallVizTest, self).__init__(enc_model_name=enc_model_name, seg_model_name=seg_model_name)

        self.dir_manager.add_location('vid_results', 'results/videos/perception/{0}_cartpole'.format(self.model.model_name))

        # Create a dir for saving the frames that make up the GIFs
        self.dir_manager.add_location('tmp', 'results/videos/perception/{0}_cartpole/tmp'.format(self.model.model_name))

        # Create and seed cartpole environment object
        self.env_name = "MujocoBall-v0"
        self.env = gym.make(self.env_name)
        self.env.seed(0)
        self.env.action_space.seed(0)

        self.use_env = True

        # Set the dt for plotting
        self.viz.set_delta_t(self.env.dt)

        # Reduce this param since Balls tend to out of view quickly
        self.max_env_traj = 15


class CartpoleVizTest(BaseVizTest):
    def __init__(self, enc_model_name: str, seg_model_name: str = None):
        super(CartpoleVizTest, self).__init__(enc_model_name=enc_model_name, seg_model_name=seg_model_name)

        self.dir_manager.add_location('vid_results', 'results/videos/perception/{0}_cartpole'.format(self.model.model_name))

        # Create a dir for saving the frames that make up the GIFs
        self.dir_manager.add_location('tmp', 'results/videos/perception/{0}_cartpole/tmp'.format(self.model.model_name))

        # Create and seed cartpole environment object
        self.env_name = "MujocoCartpole-v0"
        self.env = gym.make(self.env_name)
        self.env.seed(0)
        self.env.action_space.seed(0)

        self.use_env = True

        # Set the dt for plotting
        self.viz.set_delta_t(self.env.dt)


class DcartpoleVizTest(BaseVizTest):
    def __init__(self, enc_model_name: str, seg_model_name: str = None):
        super(DcartpoleVizTest, self).__init__(enc_model_name=enc_model_name, seg_model_name=seg_model_name)

        self.dir_manager.add_location('vid_results',
                                      'results/videos/perception/{0}_dcartpole'.format(self.model.model_name))

        # Create a dir for saving the frames that make up the GIFs
        self.dir_manager.add_location('tmp', 'results/videos/perception/{0}_dcartpole/tmp'.format(self.model.model_name))

        # Create and seed cartpole environment object
        self.env_name = "MujocoDoublecartpole-v0"
        self.env = gym.make(self.env_name)
        self.env.seed(0)
        self.env.action_space.seed(0)

        self.use_env = True

        # Set the dt for plotting
        self.viz.set_delta_t(self.env.dt)


class DubinsVizTest(BaseVizTest):
    def __init__(self, enc_model_name: str, seg_model_name: str = None):
        super(DubinsVizTest, self).__init__(enc_model_name=enc_model_name, seg_model_name=seg_model_name)

        self.dir_manager.add_location('vid_results',
                                      'results/videos/perception/{0}_dubins'.format(self.model.model_name))

        # Create a dir for saving the frames that make up the GIFs
        self.dir_manager.add_location('tmp', 'results/videos/perception/{0}_dubins/tmp'.format(self.model.model_name))

        # Create and seed cartpole environment object
        self.env_name = "MujocoDubins-v0"
        self.env = gym.make(self.env_name)
        self.env.seed(0)
        self.env.action_space.seed(0)

        self.use_env = True

        # Set the dt for plotting
        self.viz.set_delta_t(self.env.dt)
