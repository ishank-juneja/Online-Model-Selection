import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from src.agents.agent import Agent


class ConkersAgent(Agent):
    def __init__(self, config, load_name=None):
        super(ConkersAgent, self).__init__(config, load_name, fixed_environment=True)

        self.plot_true_state = self.config.env == 'MujocoCartpole-v0'

        if self.config.viz_control:
            self.init_vizualisation()

    def init_vizualisation(self):
        self.fig, self.ax = plt.subplots(1)
        self.fig2, self.state_axes = plt.subplots(self.config.state_dimension)
        if self.config.do_sys_id:
            self.fig3, self.param_axes = plt.subplots(2)
        plt.ion()

    def render(self):
        return self.env.render(mode='rgb_array', width=64, height=64, camera_id=0)

    def vizualise(self):
        self.plot_state_estimate()
        if self.config.do_sys_id:
            self.plot_param_var()
        self.draw_scene()

    def draw_scene(self):
        observation = self.img_history[-1]
        self.ax.cla()
        goal = self.env.get_goal()
        goal_pixel_x = int(goal[0] * 21) + 32
        goal_pixel_y = -int(goal[1] * 21) + 32
        self.ax.imshow(observation)
        self.ax.add_patch(Circle((goal_pixel_x, goal_pixel_y), 2))
        self.draw_trajectory(self.ax)
        plt.draw()
        plt.pause(0.001)

    def plot_state_estimate(self):
        if self.plot_true_state:
            true_states = [self.preprocess_state(state) for state in self.true_state_history]
            true = np.asarray(true_states)

        estim = np.asarray(self.state_mu_history)
        cov = np.asarray(self.state_cov_history)
        x = np.arange(0, len(estim))
        labels = ['x', 'x_mass', 'y_mass', 'v', 'th_dot']
        for i in range(self.config.state_dimension):
            self.state_axes[i].cla()
            if self.plot_true_state:
                self.state_axes[i].plot(true[:, i], label='True')
            self.state_axes[i].plot(estim[:, i], label='Estim')
            self.state_axes[i].set_title(labels[i])
            error = cov[:, i, i]
            self.state_axes[i].fill_between(x, estim[:, i] - error, estim[:, i] + error, color='r', alpha=0.1)
            if i > 2:
                self.state_axes[i].set_ylim([-15, 15])
        plt.legend()
        plt.draw()
        plt.pause(0.0001)

    def plot_param_var(self):
        param_labels = ['pole_mass', 'cart_mass', 'length', 'lin damping', 'angular_damping']

        mu = np.asarray(self.param_mu_history)
        var = np.asarray(self.param_cov_history)
        self.param_axes[0].cla()
        self.param_axes[1].cla()
        for i in range(self.config.param_dimension):
            self.param_axes[0].plot(var[:, i], label=param_labels[i])
            self.param_axes[1].plot(mu[:, i], label=param_labels[i])
            self.param_axes[0].set_title('Param Variance')
            self.param_axes[1].set_title('Param mu')

        plt.legend()
        plt.draw()
        plt.pause(0.0001)

    # TODO: refactor if this is used by me
    def preprocess_state(self, mujoco_state):
        return mujoco_state[:5]
        state = np.zeros(5)
        state[0] = -mujoco_state[0]
        th = np.arctan2(mujoco_state[1], mujoco_state[2]) - np.pi
        state[1] = np.sin(th)
        state[2] = np.cos(th)
        state[3] = -mujoco_state[3]
        state[4] = mujoco_state[4]
        return state

    def draw_trajectory(self, ax):
        init_state = self.x_mu
        init_state = torch.cat((init_state, torch.zeros_like(init_state)), dim=1)
        states = self.controller.get_rollouts(init_state)[0]

        z = torch.cat((init_state, states), dim=0)[:, :3].cpu().numpy()

        alphas = 1./np.arange(1, z.shape[0] + 1)
        alphas = np.sqrt(alphas)
        colours = ['r', 'g']
        for t in range(z.shape[0]):
            tip_y = int(-z[t, 2] * 21 + 32)
            tip_x = int(21 * z[t, 1]) + 32
            base_x = int(21 * z[t, 0]) + 32
            c = colours[1] if t else colours[0]
            if 0 < tip_x < 64 and 0 < tip_y < 64:
                ax.add_patch(Circle((tip_x, tip_y), 1, color=c, alpha=alphas[t]))
            if 0 < base_x < 64:
                ax.add_patch(Circle((base_x, 32), 1, color=c, alpha=alphas[t]))
            if 0 < tip_x < 64 and 0 < tip_y < 64 and 0 < base_x < 64:
                ax.plot([base_x, tip_x], [32, tip_y], color=c, alpha=alphas[t])
