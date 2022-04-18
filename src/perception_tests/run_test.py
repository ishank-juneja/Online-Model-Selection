import argparse
from src.perception_tests import BallVizTest, CartpoleVizTest, DcartpoleVizTest, DubinsVizTest, \
    ConkersVizTest, KendamaVizTest


def main(args):
    # Uncomment out the tester-run command pair to be used
    # tester = KendamaVizTest(seg_model_name=args.seg_model_name, enc_model_name=args.enc_model_name)

    # Tests on simple model images from training distribution
    # tester = BallVizTest(enc_model_name=args.enc_model_name)
    # tester = CartpoleVizTest(enc_model_name=args.enc_model_name)
    # tester = DcartpoleVizTest(enc_model_name=args.enc_model_name)
    tester = DubinsVizTest(enc_model_name=args.enc_model_name)
    tester.run_perception_on_tests(viz_ver="ver3")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument("--seg-model-name",
                        action='store',
                        type=str,
                        help="Name of the .pt file to use in models/segmentation",
                        metavar="seg_model_name",
                        dest="seg_model_name")

    parser.add_argument("--enc-model-name",
                        action='store',
                        type=str,
                        help="Name of the .pt file to use in models/encoder",
                        metavar="enc_model_name",
                        dest="enc_model_name")

    args = parser.parse_args()

    main(args)
