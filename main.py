import argparse
import torch
from train import train

parser = argparse.ArgumentParser(description="main.py")

# Preprocessing

parser.add_argument("-dataset_name", default="porto",  # ['porto', 'geolife', 'tdrive', 'aisus']
                    help="Name of dataset")

parser.add_argument("-min_length", default=30,
                    help="Minimum length for trajectory")

# Note that julia may not handle with trajectory longer than 40000 points.
# We still preprocess the super-long trajectories for precise statistics
# but will discard them during the training due to the GPU limitation.
parser.add_argument("-max_length", default=1000000,
                    help="Maximum length for trajectory")

parser.add_argument("-shuffle", default=True,
                    help="When generate batch set, shuffle or not")

# Model
parser.add_argument("-model_name", type=str, default="clear-S",
                    help="The name of clear's variant")

parser.add_argument("-spatial_type", default='grid',
                    help="The way of space partitioning for spatial token_generation")

parser.add_argument("-cell_size", default=100,
                    help="The cell size for grid partitioning")

parser.add_argument("-minfreq", default=50,
                    help="The min frequency for hot cell")

parser.add_argument("-aug1_name", type=str, default='distort')

parser.add_argument("-aug1_rate", default=0.4)

parser.add_argument("-aug2_name", type=str, default="downsampling")

parser.add_argument("-aug2_rate", default=0.4)

parser.add_argument("-model_settings", type=str, default=None,
                    help="suffix to mark special cases")

# Model-Encoder
parser.add_argument("-num_layers", type=int, default=1,
                    help="Number of layers in the RNN cell of encoder")

parser.add_argument("-bidirectional", type=bool, default=True,
                    help="True if use bidirectional rnn in encoder")

parser.add_argument("-hidden_size", type=int, default=256,
                    help="The hidden state size in the RNN cell of encoder")

parser.add_argument("-embed_size", type=int, default=256,
                    help="The word (cell) embedding size")

parser.add_argument("-max_grad_norm", type=float, default=5.0,
                    help="The maximum gradient norm")

parser.add_argument("-dropout", type=float, default=0.2,
                    help="The dropout probability")

parser.add_argument("-learning_rate", type=float, default=0.001)


# Model-contrastive loss

parser.add_argument("-n_views", default=2,
                    help="Numbers of data augmentation approaches")

parser.add_argument("-criterion_name", default="CrossEntropy",
                    help="CrossEntropy Loss")

parser.add_argument("-temperature", default=0.07,
                    help="Hyperparameter for controlling the InfoNCE Loss")

# Training parameters
parser.add_argument("-start_iteration", type=int, default=0)

parser.add_argument("-epochs", type=int, default=10,
                    help="The number of training epochs")

parser.add_argument("-print_freq", type=int, default=50,
                    help="Print frequency")

parser.add_argument("-save_freq", type=int, default=1000,
                    help="Save frequency")

parser.add_argument("-best_threshold", type=int, default=5000,
                    help="Training stops if no improvement after best_threshold iteration")

parser.add_argument("-batch_size", type=int, default=32,
                    help="The batch size")

parser.add_argument("-combination", type=str, default='default',
                    help="Different ways for combine pair or multiple augmentations")

parser.add_argument("-loss", type=str, default='NCE',
                    help="Contrastive Loss Function")

parser.add_argument("-cuda", type=bool, default=True,
                    help="True if we use GPU to train the model")

parser.add_argument("-flag", type=int, default=1,
                    help="If False, test model with toy training dataset")


def generate_transformers(args):
    transformer_list = []
    if args.combination == 'default':
        transformer_list.append({'name': args.aug1_name,
                                 'parameters': {'rate': f'{args.aug1_rate}'}})
        transformer_list.append({'name': args.aug2_name,
                                 'parameters': {'rate': f'{args.aug2_rate}'}})
    elif args.combination == 'single':  # args.n_views = 6
        for rate in [0.2, 0.4, 0.6]:
            for name in ['distort', 'downsampling']:
                transformer_list.append({'name': name,
                                         'parameters': {'rate': f'{rate}'}})
    elif args.combination == 'mix':  # args.n_views = 9
        for rate1 in [0.2, 0.4, 0.6]:
            for rate2 in [0.2, 0.4, 0.6]:
                transformer_list.append({'name': 'downsampling_distort',
                                         'parameters': {'rate': f'{rate1}_{rate2}'}})
    return transformer_list


def generate_suffix(args):
    suffix = ''
    # partitioning
    if args.spatial_type == "grid":
        suffix += f"{args.spatial_type}_cell-{args.cell_size}_minfreq-{args.minfreq}_"
    # combination
    if args.combination == 'default':
        suffix += f"default-{args.aug1_name}-rate-{args.aug1_rate}-{args.aug2_name}-rate-{args.aug2_rate}"
    elif args.combination == 'single':
        suffix += f"multi-single-downsampling-distort-246"
    elif args.combination == 'mix':
        suffix += f"multi-mix-downsampling-distort-246"
    # loss function
    suffix += f"_{args.loss}"
    # batch_size
    suffix += f"_batch-{args.batch_size}"

    return suffix


if __name__ == '__main__':
    args = parser.parse_args()

    # Parameter initialization based on args.dataset_name
    if args.dataset_name == "geolife":
        args.num_train = 50000
        args.num_val = 10000
        args.vocab_size = 37110
        args.save_freq = 200
    elif args.dataset_name == "porto":
        args.num_train = 1000000
        args.num_val = 10000
        args.vocab_size = 23753
        args.save_freq = 1000
    elif args.dataset_name == 'aisus':
        args.num_train = 100000
        args.num_val = 10000
        args.vocab_size = 74911
        args.save_freq = 200
    args.start = args.num_train + args.num_val

    if args.flag == 0:
        args.num_train = 1000

    args.data_path = f"data/{args.dataset_name}"
    args.json_file = f"data/{args.dataset_name}/{args.data_path}_hyper-parameters.json"

    # Generate augmented samples
    args.transformer_list = generate_transformers(args)
    args.n_views = len(args.transformer_list)

    # Generate suffix of file name
    suffix = generate_suffix(args)
    if args.model_settings is not None:  # extra info/hyper-parameters
        args.file_suffix = f"{args.model_name}_{suffix}_{args.model_settings}_{args.dataset_name}"
    else:
        args.file_suffix = f"{args.model_name}_{suffix}_{args.dataset_name}"
    args.checkpoint = f"data/{args.dataset_name}/{args.file_suffix}_checkpoint.pt"

    # check if gpu training is available
    if args.cuda and torch.cuda.is_available():
        args.device = torch.device('cuda')
    else:
        args.device = torch.device('cpu')

    for k, v in args._get_kwargs():
        print("{0} =  {1}".format(k, v))

    train(args)


