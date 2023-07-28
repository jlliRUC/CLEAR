import numpy as np
import torch
import constants
import h5py
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
        f = h5py.File(self.data_path, "r")
        num = f.attrs["num"]
        self.size = num
        for i in range(0, num):
            self.s.append(f[f"/Locations/{i}"][:])
        self.s = np.array(self.s)
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
    def __init__(self, dataset_name, num_train, num_val, transformer_list, istrain, batch_size, spatial_type='grid', cell_size=100, minfreq=50, density=4, shuffle=True):

        self.dataset_name = dataset_name
        self.num_train = num_train
        self.num_val = num_val
        self.transformer_list = transformer_list
        self.istrain = istrain
        self.batch_size = batch_size
        self.spatial_type = spatial_type
        self.cell_size = cell_size
        self.minfreq = minfreq
        self.density = density
        self.shuffle = shuffle

    def load(self):
        # train_val index
        index_list = [i for i in range(0, self.num_train)] if self.istrain else [i for i in range(self.num_train,
                                                                                                  self.num_train + self.num_val)]
        self.s_list = []
        # S info
        for transformer in self.transformer_list:
            if len(transformer["name"].split("_")) == 3:  # interpolation + mix:
                _, name1, name2 = transformer["name"].split("_")
                rate1, rate2 = transformer["parameters"]["rate"].split("_")
                suffix = f"interpolation_{name1}_rate_{rate1}_{name2}_rate_{rate2}"
            elif '_' in transformer["name"]:  # mix
                if "interpolation" in transformer["name"]:
                    name1, name2 = transformer["name"].split("_")
                    rate2 = transformer["parameters"]["rate"]
                    suffix = f"{name1}_{name2}_rate_{rate2}"
                else:
                    name1, name2 = transformer["name"].split("_")
                    rate1, rate2 = transformer["parameters"]["rate"].split("_")
                    suffix = f"{name1}_rate_{rate1}_{name2}_rate_{rate2}"
            else:  # single
                suffix = transformer["name"] + '_'
                for key in transformer["parameters"]:
                    if not key == "radius_ts":  # not distort
                        suffix += (key + "_") + (str(transformer["parameters"][key]) + "_")
                suffix = suffix.rstrip("_")
            if self.spatial_type == "grid":
                path = f"data/{self.dataset_name}/token/cell-{self.cell_size}_minfreq-{self.minfreq}/{self.dataset_name}_{suffix}_seq.h5"
            elif self.spatial_type == "quadtree":
                path = f"data/{self.dataset_name}/token/density-{self.density}_minfreq-{self.minfreq}/{self.dataset_name}_{suffix}_seq.h5"
            s = []
            with h5py.File(path, "r") as f:
                for index in index_list:
                    s.append(f[f"/Locations/{index}"][:])
                self.s_list.append(np.array(s))

        self.num_s = len(self.s_list[0])
        self.start = 0

    def get_one_batch(self):
        if self.start + self.batch_size >= self.num_s:
            return None, None
        if self.shuffle:
            index = list(range(self.num_s))
            random.shuffle(index)
            # batch_index = random.sample(index, self.batch_size)
            self.s_list = [s[index] for s in self.s_list]
            self.shuffle = False  # Just shuffle once

        sub_s_list = [pad_arrays(s[self.start:self.start + self.batch_size]) for s in self.s_list]
        lengths_list = [torch.LongTensor(list(map(len, sub_s))) for sub_s in sub_s_list]

        self.start += self.batch_size

        return sub_s_list, lengths_list



