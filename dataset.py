import numpy as np
import torch
import constants
import pickle
import random

random.seed(2023)


def pad_array(a, max_length, pad=constants.PAD):
    """

    :param a: a (array[int32])
    :param max_length:
    :param pad:
    :return:
    """
    return np.concatenate((a, [pad] * (max_length - len(a))))


def pad_arrays(a, max_length=None):
    """

    :param a:
    :param max_length:
    :return:
    """
    if max_length is None:
        max_length = max(map(len, a))
    a = [pad_array(a[i], max_length) for i in range(len(a))]
    a = np.stack(a).astype(np.int32)
    return torch.LongTensor(a)


class ExpDataLoader:
    def __init__(self, data_path, batch_size):
        self.data_path = data_path
        self.batch_size = batch_size
        self.s = []
        self.t = []
        self.start = 0

    def load(self):
        with open(self.data_path, "rb") as f:
            trajs = pickle.load(f)
        self.s = trajs
        self.size = len(self.s)
        self.start = 0

    def get_one_batch(self):
        if self.start >= self.size:
            return None, None
        s = self.s[self.start:self.start + self.batch_size]
        self.start += self.batch_size
        lengths = list(map(len, s))
        lengths = torch.LongTensor(lengths)
        return pad_arrays(s), lengths


class ClearDataLoader:
    def __init__(self, dataset_name, num_train, num_val, transformer_list, istrain, batch_size, partition_method, cell_size, minfreq, shuffle=True):

        self.dataset_name = dataset_name
        self.num_train = num_train
        self.num_val = num_val
        self.transformer_list = transformer_list
        self.istrain = istrain
        self.batch_size = batch_size
        self.partition_method = partition_method
        self.cell_size = cell_size
        self.minfreq = minfreq
        self.shuffle = shuffle

    def load(self):
        # train_val index
        index_list = [i for i in range(0, self.num_train)] if self.istrain else [i for i in range(self.num_train,
                                                                                                  self.num_train + self.num_val)]
        self.s_list = []
        self.c_list = []
        # S info
        for transformer in self.transformer_list:
            suffix = f"{transformer['name']}_rate_{transformer['rate']}"
            path = f"data/{self.dataset_name}/token/cellsize-{self.cell_size}_minfreq-{self.minfreq}/{self.dataset_name}_{suffix}_token.pkl"
            s = []
            with open(path, "rb") as f:
                df = pickle.load(f)
                for index in index_list:
                    s.append(df.loc[index, "token"][:])
                self.s_list.append(np.array(s))
        self.num_s = len(self.s_list[0])
        self.start = 0

    def get_one_batch(self):
        if self.start + self.batch_size >= self.num_s:
            return None, None
        if self.shuffle:
            index = list(range(self.num_s))
            random.shuffle(index)
            self.s_list = [s[index] for s in self.s_list]
            self.shuffle = False  # Just shuffle once

        sub_s_list = [pad_arrays(s[self.start:self.start + self.batch_size]) for s in self.s_list]
        lengths_list = [torch.LongTensor(list(map(len, s[self.start:self.start + self.batch_size]))) for s in self.s_list]

        self.start += self.batch_size

        return sub_s_list, lengths_list

