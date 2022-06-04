import cv2
import gym
import gym_cenvs
import numpy as np
import matplotlib.pyplot as plt
import os
from pynput import keyboard
import signal
from src.config import CommonEncConfig, SegConfig
from src.utils import CachedData, ResultDirManager
from typing import List, Tuple


class ClickableMujocoEnv:
    def __init__(self, env_name: str = None):
        if env_name == "kendama":
            self.long_name = 'Kendama-v0'
        elif env_name == "conkers":
            self.long_name = 'Conkers-v0'
        else:
            self.long_name = None
        self.env_name = env_name

        if self.long_name is not None:
            # Make environment
            self.env = gym.make(self.long_name)
            _ = self.env.reset_trial()

            # Get camera matrix for this environment to convert world coord to pixel coord
            self.cam_mat = self.env.get_cam_mat()

        # cv2 display window name
        self.win_name = "Env: left-click next action, q to exit"

        # segmentation and encoder config objects
        self.seg_config = SegConfig()
        self.enc_config = CommonEncConfig()

        # Get the up sample ratio so as to use right image shape for camera matrix
        self.upsample_by = self.seg_config.imsize / self.enc_config.imsize

        # y coordinate and z coordinate of actuated part in mujoco worldbody frame
        self.carty = self.cartz = 0.0

        # Whether a click has happened, acts as internal state for cv2 clicking callback
        self.clicked = False
        # For callback to update and caller to use
        self.clicked_pt = None
        # Whether to exit ongoing trajectory collection
        self.exit_loop = False
        # path where file written
        self.written_path = None

        # Time interval at which we check for a left click (ms)
        self.click_check_dt = 10

        # Structs to hold sequence of observed states, actions, and observations in traj being recorded
        self.states = []
        self.actions = []
        self.obs = []
        # Location where trajectory was saved to verify content via replay
        self.saved_loc = None

        # dir_manager_object
        self.dir_manager = ResultDirManager()
        # Get absolute path to the locations being saved to
        self.save_loc = self.dir_manager.add_location("tests", 'data/hand_made_tests')

    @staticmethod
    def homogenous_to_regular_coordinates(array: np.ndarray):
        """
        Converts a 3 tuple homogenous pixel coordinate to a normalized 2 tuple pixel coordinatr
        :param array:
        :return:
        """
        return array[:-1] / array[-1]

    def draw_circle_on_frame(self, frame: np.ndarray, location: List[float], color: Tuple[int, int, int]):
        # Radius of solid circular point (8 for 512)
        radius = 8 * self.seg_config.imsize // 512

        # Line thickness of -1 px for solid fill
        thickness = -1
        # Convert ti tuple of ints as required by cv2
        location_int = (int(location[0]), int(location[1]))

        # This is necessary for circle overlay to work
        # https://stackoverflow.com/a/31316516/3642162
        frame = frame.copy()

        # Using cv2.circle() method
        # Draw a circle of red color of thickness -1 px
        frame = cv2.circle(frame, location_int, radius, color, thickness)
        return frame

    # function to display the coordinates of
    # of the points clicked on the image
    def click_event(self, event, x, y, flags, params):
        """
        Mouse click event callback from:
        https://www.geeksforgeeks.org/displaying-the-coordinates-of-the-points-clicked-on-the-image-using-python-opencv/
        :param event:
        :param x: pixel row of click
        :param y: pixel column of click
        :param flags: unused default param of callback
        :param params: unused default param of callback
        :return:
        """
        # checking for left mouse clicks
        if event == cv2.EVENT_LBUTTONDOWN:
            self.clicked_pt = (x, y)
            self.clicked = True

    def step(self, action: float):
        """
        Return current complex model observation with robot/cup/cart position overlay
        also return pixel coordinates of robot to use to generate next action
        :return:
        """
        obs, cost, done, info = self.env.step(action)
        # Add to trajectory structs
        state = info['state']
        self.states.append(state)
        self.actions.append(action)
        self.obs.append(obs)
        # Get the current pixel coordinate of the cart from the world coord
        # In all 1D actuated models, 0th index of state is the location of actuator/cart
        cart_x = state[0]
        cart_world = np.array([cart_x, self.carty, self.cartz, 1.0])
        cart_pixel = self.cam_mat @ cart_world
        cart_pixel = self.homogenous_to_regular_coordinates(cart_pixel)
        # Up scale pixel coordinate to make sense in higher res frame
        cart_pixel *= self.upsample_by
        # Overlay cart_pixel position onto frame for viz
        obs = self.draw_circle_on_frame(frame=obs, location=cart_pixel, color=(255, 0, 0))
        return obs, done, cart_pixel

    def cache_cleanup_exit(self):
        # Retrieve next available path
        next_avail_pth = self.dir_manager.next_path(loc_name='tests', prefix=self.env_name, postfix='-%s.npz')
        self.written_path = next_avail_pth
        # Cache obs, states, actions
        np.savez(file=self.written_path, states=np.array(self.states), actions=np.array(self.actions), obs=self.obs)
        print("Cached trajectory of length {0} at {1}".format(len(self.actions), next_avail_pth))
        # Close environment
        self.env.close()
        # Close window
        cv2.destroyWindow(self.win_name)

    def on_key_press(self, key):
        # Exit key
        try:
            if key.char == 'q' or key.char == 'Q':
                self.exit_loop = True
            else:
                pass
        except AttributeError:
            pass

    def playback(self, npz_path: str):
        # # Make fresh environment
        # self.env = gym.make(self.long_name)
        # _ = self.env.reset()

        # Read in saved npz file
        npz_file = CachedData(npz_path)

        obs = npz_file.get_attribute('obs')
        print(obs.shape[0])

        # Play observations
        idx = 0
        img = None
        while idx < obs.shape[0]:
            if idx == 0:
                img = plt.imshow(obs[idx])
            else:
                img.set_data(obs[idx])
            plt.pause(0.1)
            plt.draw()
            idx += 1
        plt.close()

    def record_trajectory(self, playback_cached=True):
        # Create key listener objects in **separate** thread
        listener = keyboard.Listener(on_press=self.on_key_press)
        listener.start()

        init_action = 0.0
        action = init_action
        done = False
        while not done and not self.exit_loop:
            frame, done, cart_pixel = self.step(action)

            # Flip RGB np array to BGR for cv2
            cv2.imshow(self.win_name, frame[:, :, ::-1])
            cv2.setMouseCallback(self.win_name, self.click_event)

            # Make the displayed window persist until left click
            # Check if we have already left-clicked every 10ms
            #  waitkey also needed for video to be displayed
            while (not self.exit_loop) and (not self.clicked):
                cv2.waitKey(self.click_check_dt)

            # Check exit condition here in case inner loop broke because of exit and not click
            if self.exit_loop:
                break

            # Reset clicked state to wait for next click
            self.clicked = False

            # Mark the clicked point as an overlay
            frame = self.draw_circle_on_frame(frame, self.clicked_pt, color=(0, 255, 0))
            cv2.imshow(self.win_name, frame[:, :, ::-1])
            cv2.waitKey(1)

            # Reset step-back in case there was a step back
            self.step_back = False

            # Generate the next action based on difference between pixel x coord of cup base and click
            action = (self.clicked_pt[0] - self.seg_config.imsize/2) / self.seg_config.imsize
        print("Game ended, ... Save this trajectory (y/n)? ...")
        response = input()
        if response == 'y':
            self.cache_cleanup_exit()
            if playback_cached:
                self.playback(self.written_path)


if __name__ == '__main__':
    clickable_env = ClickableMujocoEnv('kendama')

    clickable_env.record_trajectory()

    # clickable_env.playback('data/hand_made_tests/kendama-1.npz')
