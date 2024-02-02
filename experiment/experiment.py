import os.path
import sys
sys.path.append("..")
import pandas as pd
import argparse
from experiment_utils import *
import pickle
import warnings
warnings.filterwarnings("ignore")
from preprocess.grid_partitioning import *
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

    parser.add_argument("-exp_mode", help="Preparing data or execute experiments")

    args = parser.parse_args()

    params = {}
    for param, value in args._get_kwargs():
        if value is not None:
            params[param] = value

    return params


if __name__ == "__main__":
    configs = Config()
    configs.default_update(parse_args())
    region_settings = {"min_lon": configs.min_lon,
                       "max_lon": configs.max_lon,
                       "min_lat": configs.min_lat,
                       "max_lat": configs.max_lat}

    if configs.partition_method == "grid":
        region = region_grid(configs.dataset_name, configs.cell_size, configs.minfreq, region_settings)

    region.makeVocab()

    if configs.exp_mode == "data":
        for exp_label in configs.exp_list:
            if configs.partition_method == "grid":
                folder = f"{exp_label}/{configs.dataset_name}/cellsize-{configs.cell_size}_minfreq-{configs.minfreq}"
            if not os.path.exists(folder):
                os.makedirs(folder)
            if exp_label == "self-similarity":
                data_self_similarity(configs, region)
            elif exp_label == "cross-similarity":
                data_cross_similarity(configs, region)
            elif exp_label == "knn":
                data_KNN(configs, region)
            elif exp_label == "cluster":
                data_cluster(configs, region)
    elif configs.exp_mode == "encode":
        multi_encode(configs, configs.exp_list)
    elif configs.exp_mode == "exp":
        for exp_label in configs.exp_list:
            if exp_label == "self-similarity":
                results = exp_self_similarity(configs)
                df_ss = pd.DataFrame(results).T
                df_ss.to_csv(f"{configs.suffix}_{exp_label}.csv")
            elif exp_label == "cross-similarity":
                results = exp_cross_similarity(configs)
                df_cs = pd.DataFrame(results)
                df_cs.to_csv(f"{configs.suffix}_{exp_label}.csv")
            elif exp_label == "knn":
                results = exp_KNN(configs)
                df_knn = pd.DataFrame(results).T
                df_knn.to_csv(f"{configs.suffix}_{exp_label}.csv")
            elif exp_label == 'cluster':
                results = exp_cluster(configs)
                df_cluster = pd.DataFrame(results)
                df_cluster.to_csv(f"{configs.suffix}_{exp_label}.csv")



