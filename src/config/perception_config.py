

class PerceptionConfig:
    """
    Herein specified are names of the models that are to be used for perception/simple model library
    """

    def __init__(self):
        self.names = {
            'ball': {
                'encoder_model_name': "model_ball_enc_1frame_Apr16_17-53-05",
                'seg_model_name': "model_ball_seg_1frame_MRCNN_Apr16_08-31-49"
            },

            'cartpole': {
                'encoder_model_name': "model_cartpole_enc_1frame_Dec09_17-48-17",
                'seg_model_name': "model_cartpole_seg_1frame_MRCNN_Dec09_11-30-34"
            },

            'dcartpole': {
                'encoder_model_name': "model_dcartpole_enc_1frame_Apr16_22-16-25",
                'seg_model_name': "model_dcartpole_seg_1frame_MRCNN_Apr16_09-19-58"
            },

            'dubins': {
                'encoder_model_name': "model_dubins_enc_1frame_Apr16_23-13-05",
                'seg_model_name': "model_dubins_seg_1frame_MRCNN_Apr16_09-52-30"
            }
        }

    def __getitem__(self, item: str):
        return self.names[item]
