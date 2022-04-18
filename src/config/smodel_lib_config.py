

class SimpModLibConfig:
    """
    Herein specified are names of the models that are to be used for perception/simple model library
    """
    def __init__(self):
        self.names = {
            'ball': {
                'encoder': "model_ball_enc_1frame_Apr16_17-53-05",
                'segmenter': "model_ball_seg_1frame_MRCNN_Apr16_08-31-49"
            },

            'cartpole': {
                'encoder': "model_cartpole_enc_1frame_Apr16_18-53-27",
                'segmenter': "model_cartpole_seg_1frame_MRCNN_Apr16_08-59-19"
            },

            'dcartpole': {
                'encoder': "model_dcartpole_enc_1frame_Apr16_22-16-25",
                'segmenter': "model_dcartpole_seg_1frame_MRCNN_Apr16_09-19-58"
            },

            'dubins': {
                'encoder': "model_dubins_enc_1frame_Apr16_23-13-05",
                'segmenter': "model_dubins_seg_1frame_MRCNN_Apr16_09-52-30"
            }
        }

    def __getitem__(self, item: str):
        return self.names[item]
