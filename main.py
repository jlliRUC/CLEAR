from config import Config
import argparse
from train import train


def parse_args():
    # Set some key configs from screen

    parser = argparse.ArgumentParser(description="main.py")

    parser.add_argument("-dataset_name",  # ["porto", "geolife", "tdrive", "aisus"]
                        help="Name of dataset")

    parser.add_argument("-partition_method",  # ["grid", "quadtree"]
                        help="The way of space partitioning for spatial token_generation")

    parser.add_argument("-aug1_name", type=str)

    parser.add_argument("-aug1_rate")

    parser.add_argument("-aug2_name", type=str)

    parser.add_argument("-aug2_rate")

    parser.add_argument("-combination", type=str,
                        help="Different ways for combine pair or multiple augmentations")

    parser.add_argument("-model_name", type=str,
                        help="The name of clear's variant")

    parser.add_argument("-model_settings", type=str,
                        help="suffix to mark special cases")

    parser.add_argument("-loss", type=str,
                        help="Contrastive Loss Function")

    parser.add_argument("-pretrain_mode", type=str,
                        help="np means no pretrain (by default), pf means pretrain-freeze, pt means pretrain-train")

    parser.add_argument("-pretrain_method", type=str,
                        help="node2vec or word2vec")

    parser.add_argument("-batch_size", type=int,
                        help="The batch size")

    parser.add_argument("-cell_size", type=int)

    parser.add_argument("-minfreq", type=int)

    parser.add_argument("-exp_list", nargs='+')


    args = parser.parse_args()

    params = {}
    for param, value in args._get_kwargs():
        if value is not None:
            params[param] = value

    return params


if __name__ == '__main__':
    configs = Config()
    configs.default_update(parse_args())
    for param, value in vars(configs).items():
        print(f"{param} = {value}")

    train(configs)
