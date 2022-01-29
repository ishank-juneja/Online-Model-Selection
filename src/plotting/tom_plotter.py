import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.patches import Ellipse
import seaborn as sns
sns.set()


def eigsorted(cov):
    '''
    Eigenvalues and eigenvectors of the covariance matrix.
    '''
    vals, vecs = np.linalg.eigh(cov)
    order = vals.argsort()[::-1]
    return vals[order], vecs[:, order]


def cov_ellipse(cov, nstd):
    """
    Source: http://stackoverflow.com/a/12321306/1391441
    """

    vals, vecs = eigsorted(cov)
    theta = np.degrees(np.arctan2(*vecs[:, 0][::-1]))

    # Width and height are "full" widths, not radius
    width, height = 2 * nstd * np.sqrt(vals)

    return width, height, theta


def frames_to_video(frames, filename, grey=True, cmap=None):
    '''
    Takes batched sequence of frames and saves a video
    :param frames: numpy array (N x T x C x W, H) frames
    '''
    with sns.axes_style("white"):
        N, T, C, W, H = frames.shape
        frames = np.swapaxes(frames, 0, 1)
        frames = np.swapaxes(frames, 2, 4)
        #frames = np.swapaxes(frames, 3, 4)

        if cmap is None and grey:
            cmap = 'gray'

        if grey:
            frames = frames.reshape((T, N, H, W))

        rows, cols = np.sqrt(N), np.sqrt(N)
        if np.mod(rows, 1):
            raise ValueError("Number of frames must be square for display")

        rows, cols = int(rows), int(cols)
        fig, axes = plt.subplots(rows, cols)

        if N == 1:
            axes = np.asarray([[axes]])
        # init
        lines = []
        for i in range(N):
            lines.append(axes.flatten()[i].imshow(frames[0, i], cmap=cmap))

        def updatefig(frame):
            for i in range(N):
                lines[i] = axes.flatten()[i].imshow(frame[i], cmap=cmap)
            return lines

        ani = animation.FuncAnimation(fig, updatefig, frames=frames, interval=500, blit=True)
        writer = animation.writers['avconv']
        writer = writer(fps=3)
        ani.save(filename, writer=writer)
        plt.close(fig)


def plot_img_trajectory(frames_1, frames_2, folder, grey=True):
    N, T, C, W, H = frames_1.shape
    if grey:
        frames_1 = frames_1.reshape((N, T, H, W))
        frames_2 = frames_2.reshape((N, T, H, W))

    L = 8
    for n in range(N):

        filename = '{}/image_traj_{}.png'.format(folder, n)

        fig, axes = plt.subplots(2, L, figsize=(16, 4))
        for t in range(L):
            axes[0, t].imshow(frames_1[n, 8 + t].T, cmap='gray')
            axes[1, t].imshow(frames_2[n, 8 + t].T, cmap='gray')

            axes[0, t].axis('off')
            axes[1, t].axis('off')

        fig.savefig(filename)


def plot_two_frames(frames_1, frames_2, filename, grey=True):
    ''' Plots two videos overlayed on one another'''
    '''
      Takes batched sequence of frames and saves a video
      :param frames: numpy array (N x T x C x W, H) frames
      '''
    frames_1[frames_1 > 0.1] = 1.0
    #frames_2[frames_2 < 0.9] = 0.0

    frames_2 = 1 - 0.5 * frames_2
    frames = frames_2 - frames_1
    frames_to_video(frames, filename, grey, cmap=plt.get_cmap('gist_heat'))


def plot_latent_observations(z_encoded_mu,
                             z_encoded_sigma,
                             z_rollout_mu,
                             z_rollout_sigma,
                             filename):
    '''

    :param z_mu: N x T x z_dim
    :param z_sigma: N x T x z_dim x z_dim
    :return:
    '''
    with sns.axes_style("white"):
        N, T, z_dim = z_encoded_mu.shape
        import warnings
        if z_dim > 3:
            warnings.warn("Cannot plot z_dim for higher than 3D")
            return

        rows, cols = np.sqrt(N), np.sqrt(N)
        if np.mod(rows, 1):
            raise ValueError("Number of frames must be square for display")

        rows, cols = int(rows), int(cols)
        if z_dim == 2:
            fig, axes = plt.subplots(rows, cols, figsize=(16, 16))
        elif z_dim == 3:
            from mpl_toolkits import mplot3d
            #fig, axes = plt.subplots(rows, cols, figsize=(16, 16), projection='3d')
            fig = plt.figure(figsize=(16, 16))
            axes = []
            for i in range(N):
                axes.append(fig.add_subplot(rows, cols, i+1, projection='3d'))

            axes = np.asarray(axes)

        # plot mean
        for i in range(N):
            if z_dim == 2:
                axes.flatten()[i].plot(z_encoded_mu[i, :, 0], z_encoded_mu[i, :, 1], '-bo')
                axes.flatten()[i].plot(z_rollout_mu[i, :, 0], z_rollout_mu[i, :, 1], '-ro')
            else:
                axes.flatten()[i].plot3D(z_encoded_mu[i, :, 0], z_encoded_mu[i, :, 1], z_encoded_mu[i, :, 2], '-bo', label='encoded')
                axes.flatten()[i].plot3D(z_rollout_mu[i, :, 0], z_rollout_mu[i, :, 1], z_rollout_mu[i, :, 2], '-ro', label='predicted')

            axes.flatten()[i].legend()

            continue

            # Plot covariance ellipses
            for t in range(T):
                width, height, theta = cov_ellipse(z_encoded_sigma[i, t], 1)
                ellipse_encoded = Ellipse(xy=z_encoded_mu[i, t],
                                          width=width, height=height,
                                          angle=theta, color='b', alpha=0.2)
                width, height, theta = cov_ellipse(z_rollout_sigma[i, t], 1)
                ellipse_rollout = Ellipse(xy=z_rollout_mu[i, t],
                                          width=width, height=height,
                                          angle=theta, color='r', alpha=0.2)

                axes.flatten()[i].add_patch(ellipse_encoded)
                axes.flatten()[i].add_patch(ellipse_rollout)
        fig.savefig(filename)


def plot_mixing_parameters(alpha, filename):
    '''
    :param alpha: N x T x K describing mixing parameters
    '''
    N, T, K = alpha.shape
    rows, cols = np.sqrt(N), np.sqrt(N)
    if np.mod(rows, 1):
        raise ValueError("Number of frames must be square for display")

    rows, cols = int(rows), int(cols)
    fig, axes = plt.subplots(rows, cols, figsize=(16, 16))

    # plot means
    for i in range(N):
        for k in range(K):
            axes.flatten()[i].plot(alpha[i, :, k])

    fig.savefig(filename)


def plot_grid_images(images, filename):

    N, C, W, H = images.size()

    rows, cols = np.sqrt(N), np.sqrt(N)
    if np.mod(rows, 1):
        raise ValueError("Number of frames must be square for display")

    rows, cols = int(rows), int(cols)
    fig, axes = plt.subplots(rows, cols, figsize=(16, 16))

    if C == 1:
        images = images.view(N, W, H).cpu().numpy()
        images = np.swapaxes(images, 1, 2)
        cmap = 'gray'
    else:
        images = images.cpu().numpy()
        images = np.swapaxes(images, 1, 3)
        cmap = None

    # Show images
    for i in range(N):
        axes.flatten()[i].imshow(images[i], cmap=cmap)

    fig.savefig(filename)
