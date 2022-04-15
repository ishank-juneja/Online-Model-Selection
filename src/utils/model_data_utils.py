from src.config import BallConfig, CartpoleConfig, DcartpoleConfig, DubinsConfig


class EncDataset:
    def __init__(self, data_dir_name: str):
        """
        Takes in a data dir name and build a struct of the most important meta data out of it
        Raises Value Error if invalid format
        :param data_dir_name:
        """
        dir_as_lst = data_dir_name.split('_')
        # Raise error message if problem detected in name at any point
        format_err = "Incorrect format: A segmentation dataset folder name should follow format: " \
                     "{simp_model}_enc_{nframes}, nframes with n in {1, 2}"
        simp_model_name_err = "Unknown simple model name, should be in {cartpole, ball, dcartpole, dubins}"
        invalid_frames = "Invalid number of frames, should be in [1frame, 2frame]"

        # If ok set data_dir_name
        self.data_folder = data_dir_name

        # Check if valid format
        if len(dir_as_lst) != 3:
            raise ValueError(format_err)
        elif dir_as_lst[0] not in ['cartpole', 'dcartpole', 'ball', 'dubins']:
            raise ValueError(simp_model_name_err)
        elif dir_as_lst[1] != 'enc':
            raise ValueError(format_err)
        elif dir_as_lst[2] not in ['1frame', '2frame']:
            raise ValueError(invalid_frames)

        self._simp_model = dir_as_lst[0]
        if dir_as_lst[2] == '1frame':
            self._nframes = 1
        else:
            self._nframes = 2

    def get_nframe(self) -> int:
        return self._nframes

    def get_simp_model(self) -> str:
        return self._simp_model

    def get_enc_cfg(self):
        if self._simp_model == 'ball':
            config = BallConfig(self.data_folder)
        elif self._simp_model == 'cartpole':
            config = CartpoleConfig(self.data_folder)
        elif self._simp_model == 'dcartpole':
            config = DcartpoleConfig(self.data_folder)
        elif self._simp_model == 'dubins':
            config = DubinsConfig(self.data_folder)
        else:
            raise NotImplementedError("Config for simple model {0} is not implemented".format(self._simp_model))
        return config


class SegDataset:
    def __init__(self, data_dir_name: str):
        """
        Takes in a data dir name and build a struct of the most important meta data out of it
        Raises Value Error if invalid format
        :param data_dir_name:
        """
        dir_as_lst = data_dir_name.split('_')
        # Raise error message if problem detected in name at any point
        format_err = "Incorrect format: A segmentation dataset folder name should follow format: " \
                     "{simp_model}_seg_{nframes}, nframes in {1, 2}"
        simp_model_name_err = "Unknown simple model name, should be in {cartpole, ball, dcartpole, dubins}"
        invalid_frames = "Invalid number of frames should be in [1frame, 2frame1mask, 2frame2mask]"

        # Check if valid format
        if len(dir_as_lst) != 3:
            raise ValueError(format_err)
        elif dir_as_lst[0] not in ['cartpole', 'dcartpole', 'ball', 'dubins']:
            raise ValueError(simp_model_name_err)
        elif dir_as_lst[1] != 'seg':
            raise ValueError(format_err)
        elif dir_as_lst[2] not in ['1frame', '2frame1mask', '2frame2mask']:
            raise ValueError(invalid_frames)

        self._simp_model = dir_as_lst[0]
        self._nframes = dir_as_lst[2]
        if self._nframes == '2frame1mask' or self._nframes == '1frame':
            self._nmasks = 1
        else:
            self._nmasks = 2

    def get_nframe(self) -> str:
        return self._nframes

    def get_simp_model(self) -> str:
        return self._simp_model

    def get_nmasks(self) -> int:
        return self._nmasks
