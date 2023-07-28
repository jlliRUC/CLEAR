import sys
sys.path.append("..")
import pandas as pd
import argparse
from experiment_utils import *
from token_generation.STToken import region_grid


parser = argparse.ArgumentParser(description="experiment.py")

parser.add_argument("-model_name", default="clear-S", help="Name of model. Also the suffix of generated vec.h5")

parser.add_argument("-batch_size", default=64, help="batch_size for encoding")

parser.add_argument("-file_suffix")

parser.add_argument("-dataset_name", default="porto", help="Name of dataset")

parser.add_argument("-mode", default="data", help="Preparing data or execute experiments")

parser.add_argument("-exp_list", nargs='+', default=["self-similarity", "cross-similarity", "knn"], help="Type of exp to be done")

parser.add_argument("-num_query_ss", default=1000, help="Query number for self-similarity")

parser.add_argument("-num_key_ss", default=10000, help="Key number for self-similarity")

parser.add_argument("-num_pair", default=10000, help="Pair number for cross-similarity")

parser.add_argument("-num_set", default=10000, help="Set number for cluster")

parser.add_argument("-num_query_knn", default=1000, help="Query number of KNN")

parser.add_argument("-num_key_knn", default=10000, help="Key number of KNN")

parser.add_argument("-key_sizes", nargs='+', default=[2000, 4000, 6000, 8000, 10000],
                    help="key sizes for exp-self-similarity")

parser.add_argument("-data_path", default="../data", help="Path to save all data files.")

parser.add_argument("-min_length", default=30, help="Minimum of points a trajectory has to consist")

parser.add_argument("-max_length", default=1000000, help="Maximum of points a trajectory has to consist")

parser.add_argument("-name_list", nargs='+', default=["original", "downsampling", "distort", "interpolation"],
                    help="transformer names")

parser.add_argument("-rate_list", nargs='+', default=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6], help="transformer parameters")

parser.add_argument("-distance", default=lambda x, y: np.linalg.norm(x-y), help="Function for calculating pairwise-distance")

parser.add_argument("-max_k", default=50, help="Max k for exp-KNN")

parser.add_argument("-k_list", nargs='+', default=[20, 30, 40, 50], help="k_list for exp-KNN")

# Model-Encoder
parser.add_argument("-num_layers", type=int, default=1,
                    help="Number of layers in the RNN cell of encoder")

parser.add_argument("-bidirectional", type=bool, default=True,
                    help="True if use bidirectional rnn in encoder")

parser.add_argument("-hidden_size", type=int, default=256,
                    help="The hidden state size in the RNN cell of encoder")

parser.add_argument("-embed_size", type=int, default=256,
                    help="The word (cell) embedding size")

parser.add_argument("-dropout", type=float, default=0.2,
                    help="The dropout probability")

parser.add_argument("-checkpoint", default='best.pt',
                    help="The saved checkpoint")

parser.add_argument("-spatial_type", default='grid',
                    help="The way of space partitioning for spatial token")

parser.add_argument("-cell_size", default=100,
                    help="The cell size for grid partitioning")

parser.add_argument("-minfreq", default=50,
                    help="The min frequency for hot cell")

parser.add_argument("-aug1_name", type=str, default='distort')

parser.add_argument("-aug1_rate", default=0.4)

parser.add_argument("-aug2_name", type=str, default="downsampling")

parser.add_argument("-aug2_rate", default=0.4)

parser.add_argument("-combination", type=str, default="single",
                    help="suffix to mark special cases")

parser.add_argument("-loss", type=str, default="pos-rank-out-all",
                    help="suffix to mark special cases")

parser.add_argument("-model_settings", type=str, default=None,
                    help="suffix to mark special cases")

parser.add_argument("-model_name", type=str, default='clear-S',
                    help="suffix to mark special cases")


def generate_suffix(spatial_type, cell_size, minfreq, combination, loss, batch_size, aug1_name, aug1_rate, aug2_name, aug2_rate):
    suffix = ''
    # partitioning
    if spatial_type == "grid":
        suffix += f"{spatial_type}_cell-{cell_size}_minfreq-{minfreq}_"

    # combination
    if combination == 'default':
        suffix += f"default-{aug1_name}-rate-{aug1_rate}-{aug2_name}-rate-{aug2_rate}"
    elif combination == 'single':
        suffix += f"multi-single-downsampling-distort-246"
    elif combination == 'mix':
        suffix += f"multi-mix-downsampling-distort-246"

    # loss function
    suffix += f"_{loss}"

    # batch_size
    suffix += f"_batch-{batch_size}"

    return suffix


if __name__ == "__main__":
    args = parser.parse_args()
    # Parameter initialization based on args.dataset_name
    if args.dataset_name == "geolife":
        args.start = 50000 + 10000
        args.vocab_size = 37110
    elif args.dataset_name == "aisus":
        args.start = 100000 + 10000
        args.vocab_size = 74911
    elif args.dataset_name == "porto":
        args.start = 1000000 + 10000
        args.vocab_size = 23753
        args.num_pair = 10000
    else:
        print("Unexpected dataset!")

    if args.model_name.startswith("clear"):
        args.checkpoint = f"../data/{args.dataset_name}/{args.file_suffix}_best.pt"
    else:
        args.checkpoint = f"../baseline/{args.file_suffix}.pt"

    if args.spatial_type == "grid":
        region = region_grid(args.dataset_name, args.cell_size, args.minfreq)
    region.makeVocab()

    for k, v in args._get_kwargs():
        print("{0} =  {1}".format(k, v))

    suffix = generate_suffix(args.spatial_type, args.cell_size, args.minfreq, args.combination, args.loss, args.batch_size, args.aug1_name, args.aug1_rate, args.aug2_name, args.aug2_rate)
    if args.model_settings is not None:  # extra info/hyper-parameters
        file_suffix = f"{args.model_name}_{suffix}_{args.model_settings}_{args.dataset_name}"
    else:
        file_suffix = f"{args.model_name}_{suffix}_{args.dataset_name}"

    if args.mode == "data":
        for exp_label in args.exp_list:
            if args.spatial_type == "grid":
                folder = f"{exp_label}/{args.dataset_name}/cell-{args.cell_size}_minfreq-{args.minfreq}"
                if not os.path.exists(folder):
                    os.makedirs(folder)
            if exp_label == "self-similarity":
                data_self_similarity(args, region)
            elif exp_label == "cross-similarity":
                data_cross_similarity(args, region)
            elif exp_label == "knn":
                data_KNN(args, region)
    elif args.mode == "encode":
        multi_encode(args, args.exp_list)
    elif args.mode == "exp":
        for exp_label in args.exp_list:
            if exp_label == "self-similarity":
                results = exp_self_similarity(args)
                df_ss = pd.DataFrame(results).T
                df_ss.to_csv(f"{args.file_suffix}_{exp_label}.csv")
            elif exp_label == "cross-similarity":
                results = exp_cross_similarity(args)
                df_cs = pd.DataFrame(results)
                df_cs.to_csv(f"{args.file_suffix}_{exp_label}.csv")
            elif exp_label == "knn":
                results = exp_KNN(args)
                df_knn = pd.DataFrame(results).T
                df_knn.to_csv(f"{args.file_suffix}_{exp_label}.csv")



