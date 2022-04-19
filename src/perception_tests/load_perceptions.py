import argparse
from src.learned_models.simp_model_perception import SimpModPerception


def main(args):
    # Testing if all the perceptions are loadable ...

    ball_per = SimpModPerception(seg_model_name="model_ball_seg_1frame_MRCNN_Apr16_08-31-49",
                      encoder_model_name="model_ball_enc_2frame_Apr16_07-30-33")

    cartpole_per = SimpModPerception(seg_model_name="model_cartpole_seg_1frame_MRCNN_Apr16_08-59-19",
                      encoder_model_name="model_cartpole_enc_2frame_Apr16_07-04-38")

    dcartpole_per = SimpModPerception(seg_model_name="model_dcartpole_seg_1frame_MRCNN_Apr16_09-19-58",
                      encoder_model_name="model_dcartpole_enc_2frame_Apr16_07-58-08")

    dubins_per = SimpModPerception(seg_model_name="model_dubins_seg_1frame_MRCNN_Apr16_09-52-30",
                      encoder_model_name="model_dubins_enc_2frame_Apr15_23-07-38")

    # Hold in memory till user input
    input()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    args = parser.parse_args()

    main(args)
