"""
All models in ensemble may not learn the required mapping from images to states
Since aleatoric uncertainty is built into the NN predictions, an ensemble member may converge to a frozen distribution
with a fixed mean and variance regardless of distribution
"""
import argparse
from src.learned_models.ensemble import EncoderEnsemble
from src.training import MyDatasetBuilder
from torch.utils.data import DataLoader, RandomSampler


def main(args):
    loaded_ensemble = EncoderEnsemble(args.enc_model_name)
    loaded_ensemble.send_model_to_gpu()

    nmembers = loaded_ensemble.get_n_ensembles()

    THRESH = 1.0

    # Get config bject of senmble
    config = loaded_ensemble.get_config()

    # Exclude all sugs list
    exclude_all_augs_lst = ["no_bg_simp_model", "no_bg_imgnet", "no_fg_texture", "no_bg_shape", "no_noise"]
    # Make dataloader to iterate over test dataset
    dataset_builder = MyDatasetBuilder(config=config, excluded_augs=exclude_all_augs_lst)

    test_dataset = dataset_builder.get_dataset(dataset_type='test')

    # Perform inference on batches of test dataset and compute MSE separately for every member using GT labels
    #  If MSE exceeds THRESH for any member for any batch, then declare that member an outlier ...

    # Number of test-points
    ntestpts = 100
    ctr = 0

    # Load in a trajectory of images
    for obs, state, action in test_dataset:
        img_np = obs[0].cpu().permute((1, 2, 0)).detach().numpy()
        if ctr < ntestpts:
            # N, T, _ = state.size()

            obs = obs.cpu().squeeze().detach()
            obs = obs.permute((0, 2, 3, 1))
            obs = obs.numpy()

            for idx in range(len(obs)):
                z_mu, _ = loaded_ensemble.encode_single_obs(obs[idx])

                z_mu = z_mu.view(nmembers, config.obs_dim)
                mu_np = z_mu.cpu().detach().numpy()

                print(mu_np)
                print(state)

                ctr += 1
        else:
            break


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument("--enc-model-name",
                        action='store',
                        type=str,
                        help="Name of the .pt file to use in models/encoder",
                        metavar="enc_model_name",
                        dest="enc_model_name")

    args = parser.parse_args()

    main(args)
