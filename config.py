import os
import torch
import numpy as np


class Config:
    def __init__(self):
        # Default settings
        self.root_dir = "./"
        self.dataset_name = "porto"
        self.data_path = os.path.join(self.root_dir, f"data/")
        ## Preprocessing
        self.min_length = 30  # Length filter for trajectory data
        self.max_length = 2000  # transformers need a small max_length to be fast

        ## Partitioning
        ### Grid
        self.partition_method = "grid"
        self.cell_size = 100
        self.minfreq = 50

        ## Augmentation
        self.aug1_name = "distort"
        self.aug1_rate = 0.4
        self.aug2_name = "downsampling"
        self.aug2_rate = 0.4
        self.combination = "default"

        ## Model
        self.model_name = "clear-S"
        self.model_settings = None
        self.num_layers = 1
        self.bidirectional = True
        self.hidden_size = 256
        self.embed_size = 256
        self.max_grad_norm = 5.0  # The maximum gradient norm
        self.dropout = 0.2
        self.learning_rate = 0.001
        self.num_heads = 4
        self.loss = "NCE"
        self.n_views = 2
        self.criterion_name = "CrossEntropy"
        self.temperature = 0.07
        self.pretrain_mode = "np"
        self.pretrain_method = "node2vec"

        ## Training
        self.shuffle = True  # Shuffle training set or not
        self.start_iteration = 0
        self.epochs = 10
        self.print_freq = 100
        self.save_freq = 1000
        self.best_threshold = 5000
        self.batch_size = 64
        self.cuda = True
        self.lr_degrade_gamma = 0.5
        self.lr_degrade_step = 5

        ## Exp
        self.exp_mode = "data"
        self.exp_list = ["self-similarity", "cross-similarity", "knn", "cluster"]
        self.num_query_ss = 1000
        self.num_key_ss = 10000
        self.num_pair = 1000
        self.num_set = 10000
        self.num_query_knn = 1000
        self.num_key_knn = 10000
        self.key_sizes = [2000, 4000, 6000, 8000, 10000]
        self.name_list = ["original", "distort", "downsampling"]
        self.rate_list = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]
        self.distance = lambda x, y: np.linalg.norm(x - y)
        self.max_k = 50
        self.k_list = [5, 10, 20, 30, 40, 50]

        ## Debugging
        self.debug = False  # If True, debug model with toy (1000) dataset

    def config_dataset(self):
        if self.dataset_name == "porto":
            # Boundary setting
            self.min_lon = -8.735152
            self.min_lat = 40.953673
            self.max_lon = -8.156309
            self.max_lat = 41.307945
            # Training setting
            self.num_train = 200000
            self.num_val = 10000
            self.vocab_size = 23751
            # Exp setting
            self.exp_start = 1010000
        elif self.dataset_name == "geolife":
            # Boundary setting
            self.min_lon = 115.416666
            self.min_lat = 39.45
            self.max_lon = 117.5
            self.max_lat = 41.05
            # Training setting
            self.num_train = 50000
            self.num_val = 5000
            self.vocab_size = 37106
            self.save_freq = 200
            # Exp setting
            self.exp_start = self.num_train + self.num_val
        else:
            print("Unknown dataset!")

    def config_debug(self):
        if self.debug:
            self.num_train = 1000

    def config_augmentation(self):
        transformer_list = []
        if self.combination == 'default':
            transformer_list.append({'name': self.aug1_name,
                                     'rate': f'{self.aug1_rate}'})
            transformer_list.append({'name': self.aug2_name,
                                     'rate': f'{self.aug2_rate}'})
        elif self.combination == 'multi':
            for name in ["distort", "downsampling"]:
                for rate in [0.2, 0.4, 0.6]:
                    transformer_list.append({'name': name,
                                             'rate': f'{rate}'})
        else:
            print("Unknown augmentation!")
        self.transformer_list = transformer_list
        self.n_views = len(self.transformer_list)

    def config_device(self):
        if self.cuda and torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")

    def config_exp(self):
        if self.model_name.startswith("clear"):
            self.bestpoint = os.path.join(self.root_dir, f"data/{self.dataset_name}/{self.suffix}_best.pt")
        else:
            self.checkpoint = os.path.join(self.root_dir, f"baseline/{self.suffix}.pt")

    def suffix_generation(self):
        suffix = self.model_name+"_"
        if self.model_name.startswith("clear"):
            # partitioning
            suffix += f"{self.partition_method}_cellsize-{self.cell_size}_minfreq-{self.minfreq}_"
            # combination
            if self.combination == 'default':
                suffix += f"default-{self.aug1_name}-rate-{self.aug1_rate}-{self.aug2_name}-rate-{self.aug2_rate}"
            elif self.combination == 'multi':
                suffix += f"multi-downsampling-distort-246"

            # loss function
            suffix += f"_{self.loss}"

            # batch_size
            suffix += f"_batch-{self.batch_size}"

            # pretrain
            suffix += f"_pretrain-{self.pretrain_method}-{self.pretrain_mode}"

            # special model setting
            if self.model_settings is not None:
                suffix += f"_{self.model_settings}"

            # dataset_name
            suffix += f"_{self.dataset_name}"

            self.suffix = suffix
            self.checkpoint = os.path.join(self.root_dir, f"data/{self.dataset_name}/{self.suffix}_checkpoint.pt")

    def default_update(self, params):
        for param, value in params.items():
            if param in self.__dict__:
                setattr(self, param, value)
        self.config_dataset()
        self.config_augmentation()
        self.config_device()
        self.suffix_generation()
        self.config_exp()



