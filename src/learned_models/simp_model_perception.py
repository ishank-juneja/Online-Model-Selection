import logging
import numpy as np
from src.learned_models.ensemble import EncoderEnsemble
from src.learned_models.segmenter import Segmenter
from torch import nn


class SimpModPerception(nn.Module):
    """
    Visual model that stitches together trained Segmenter and EncoderEnsemble models
    """
    def __init__(self, encoder_model_name: str, seg_model_name: str = None):
        super(SimpModPerception, self).__init__()

        # Format for a seg_model_name is: model_{simp_name}_seg_{frame_type}_{arch}_{date}_{time}
        self.seg_model_name = seg_model_name
        # Format for enc_model_name is: model_{simp_name}_enc_{frame_type}_{date}_{time}
        self.enc_model_name = encoder_model_name

        # Segmenter is optional, when there is no segmentation we just show the encoder unprocessed frames
        self.segmenter = Segmenter(seg_model_name)

        # Encoder
        self.encoder = EncoderEnsemble(encoder_model_name, load_model=True)
        self.encoder.send_model_to_gpu()
        # This wrapper class is only invoked at test time, never at train so always call below...
        self.encoder.eval_mode()

        # Check encoder and segmenter for compatibility and retrieve the down-sample ratio
        self.downsample_by = self.check_compatible()

        # Simple model trained on
        self.simp_model = self.encoder.simp_model

        # Extract unique parts out of enc_model and seg_model
        if self.seg_model_name is not None:
            seg_name_split = self.seg_model_name.split("_")
            seg_model_unique = seg_name_split[2] + '_' + seg_name_split[3] + '_' + seg_name_split[4] + '_' \
                               + seg_name_split[5] + '_' + seg_name_split[6]
        else:
            seg_model_unique = "no_seg"
        enc_name_split = self.enc_model_name.split('_')
        enc_model_unique = enc_name_split[2] + '_' + enc_name_split[3] + '_' + enc_name_split[4] + '_' \
                           + enc_name_split[5]
        # Stitch together model names into single coherent irredundant  name
        self.model_name = self.simp_model + '_' + seg_model_unique + '_' + enc_model_unique

        logging.info("Loaded in perception model {0}".format(self.model_name))

    def forward(self, frame: np.ndarray):
        """
        Takes in a single frame as a numpy array and outputs the encoded state of the simple model
        :param frame:
        :return:
        """
        # Determine nature of passed frame
        height, width, channels = frame.shape

        # Frames are received h-stacked together by SimpModPerception
        if width // height != self.get_nframes():
            raise ValueError("Incorrect input dimension for model {0}".format(self.model_name))

        # Segmenter is optional
        if self.seg_model_name is not None:
            # If segmenter is trained to operate on single frames and encoder on 2,
            # we pass the images one at a time twice
            if self.segmenter.nframes == '1frame' and self.encoder.nframes == 2:
                masked1, conf_left = self.segmenter(frame[:, :self.segmenter.seg_config.imsize])
                masked2, conf_right = self.segmenter(frame[:, self.segmenter.seg_config.imsize:])
                masked = np.hstack((masked1, masked2))
                # We are treating this as a joint segmentation, so let the confidence be the average over the two
                conf = (conf_left + conf_right)/2
            else:
                masked, conf = self.segmenter(frame)
        else:
            # No masking
            masked = frame
            conf = None

        # Bring mask down to size encoder is trained to handle
        masked_downsamp = masked[::self.downsample_by, ::self.downsample_by, :]

        if self.get_nframes() == 2:
            # Stack images along channel dimension for encoder compatibility
            masked_downsamp_dstack = np.dstack((masked_downsamp[:, :self.encoder.config.imsize],
                                                masked_downsamp[:, self.encoder.config.imsize:]))
        else:
            masked_downsamp_dstack = masked_downsamp
        mu, stdev = self.encoder.encode_single_obs(masked_downsamp_dstack)

        # Return both the masked image and the encoder output
        return masked_downsamp, conf, mu, stdev

    def check_compatible(self) -> int:
        """
        Segmenter and encoder are compatible if the output res of the segmenter
        is a whole number multiple of the input res of the encoder and
        if both are trained to work with the same nframes, n in {1, 2}
        Exception to above rule is that segmenter can be 1 frame while encoder is 2 frame
        but not the other way around ...
        and if both are trained for the same simple model
        :return:
        """
        compatible = True
        if self.seg_model_name is not None:
            if self.segmenter.seg_config.imsize % self.encoder.config.imsize != 0:
                compatible = False
            elif self.segmenter.simp_model != self.encoder.simp_model:
                compatible = False
            elif '2frame' in self.segmenter.nframes:
                if self.encoder.nframes != 2:
                    compatible = False
        # Always compatible ig seg is None
        if not compatible:
            raise ValueError("Provided segmenter {0} and encoder {1} are incompatible".format(self.seg_model_name,
                                                                                              self.enc_model_name))
        else:
            # OW return the downsample ratio
            if self.segmenter is None:
                return 512 // self.encoder.config.imsize
            else:
                return self.segmenter.seg_config.imsize // self.encoder.config.imsize

    def get_input_imsize(self) -> int:
        """
        Returns the size of square images this visual model is trained to deal with
        which is also the shape of images the first stage segmenter is trained to deal with
        :return:
        """
        return self.segmenter.seg_config.imsize

    def get_nframes(self) -> int:
        """
        Get the number of frames this visual model is trained to work with
        :return:
        """
        return self.encoder.nframes

    def cfg(self):
        """
        Returns the config of the encoder part of the perception
        Encoder config doubles as config for simple-model itself since during
         online execution
        :return:
        """
        return self.encoder.config

    def get_model_name(self):
        """
        Return the simple model name as a string
        :return:
        """
        return self.encoder.simp_model
