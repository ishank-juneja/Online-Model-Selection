import argparse
import numpy as np
from src.learned_models.ensemble import EncoderEnsemble
from src.plotting import GIFmaker, SMVOffline
import torch


def run_perception_on_test(self, frames):
    # raw = raw incoming frames from simulator, masked = masked frames from segmenter
    raw_frames = []
    masked_frames = []

    # Allocate arrays to hold mean and stddev predictions over the observations
    traj_mu = np.zeros((frames.shape[0], self.obs_dim), dtype=np.float64)

    # 3 kinds of uncertainties: total, alea, epi
    traj_stddev = np.zeros((frames.shape[0], 3, self.obs_dim), dtype=np.float64)

    # To hold separate mean predictions for every ensemble member
    full_mu = np.zeros((frames.shape[0], self.nmembers, self.obs_dim), dtype=np.float64)

    # Reference to the cur_obs at time {t-1}
    prev_obs = None

    # See results from perception on this cached sequence of actions
    for idx in range(frames.shape[0]):
        cur_obs = frames[idx]
        if self.nframes == 2:
            # For the very first frame, duplicate cur_obs (would want to do something like this online)
            if prev_obs is None:
                # Raw frame for visualization without white line que
                raw_frames.append(self.frames_handler.concat(cur_obs, cur_obs))
                # Test frame to pass to model with white line que
                test_frame = self.frames_handler.concat(cur_obs, cur_obs)
            else:
                raw_frames.append(self.frames_handler.concat(prev_obs, cur_obs))
                test_frame = self.frames_handler.concat(prev_obs, cur_obs)
            # Update prev_obs for next iteration
            prev_obs = cur_obs
        # Else use single frames
        elif self.nframes == 1:
            raw_frames.append(cur_obs)
            test_frame = cur_obs
        else:
            raise NotImplementedError("Tests for {0} frames not implemented".format(self.nframes))

        # Invoke perception model
        masked, mu_np, stddev_np = self.invoke_perception(test_frame)

        # Add masked frame to results
        masked_frames.append(masked)

        # Predicted observable state means
        full_mu[idx] = mu_np
        traj_mu[idx, :] = mu_np.mean(axis=0)

        # Compute uncertainty terms
        traj_stddev[idx, 1, :] = self.get_aleatoric_uncertainty(stddev_np)
        traj_stddev[idx, 2, :] = self.get_epistemic_uncertainty(mu_np)
        traj_stddev[idx, 0, :] = np.sqrt(
            np.square(traj_stddev[idx, 1, :]) + np.square(traj_stddev[idx, 2, :]))

    dir_save = "results/"
    # self.viz.save_animation_frames_seg(raw_frames, masked_frames, dir_save)
    self.viz.save_test_animation_frames_ver1(masked_frames, traj_mu, traj_stddev, full_mu, dir_save)

    gif_path = "results/test.gif"
    gif_maker = GIFmaker()
    gif_maker.make_gif(gif_path, dir_save)

    return


def main(args):

    encoder = EncoderEnsemble(model_name="model_cartpole_enc_2frame_Apr08_09-58-07", load_model=True)
    encoder.send_model_to_gpu()

    viz = SMVOffline(simp_model='cartpole')

    gif_maker = GIFmaker()

    # input = np.zeros((64, 64, 6), dtype=np.uint8)

    img_traj = np.load("data/cartpole_enc_2frame/train_traj_1/traj_observations.npy")

    traj_states = np.load("data/cartpole_enc_2frame/train_traj_1/traj_states.npy")

    for idx in range(img_traj.shape[0]):
        input = img_traj[idx]

        in1 = input[..., :3]
        in2 = input[..., 3:]
        in_hstack = np.hstack((in1, in2))


        mu_pt, std_pt = encoder.encode_single_obs(input[..., ::-1])

        mu_np = mu_pt.cpu().detach().numpy().reshape((10, 5))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    args = parser.parse_args()

    main(args)

