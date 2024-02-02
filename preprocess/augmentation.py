import sys
sys.path.append("..")
import time
import numpy as np
import math
import random
import copy
from utils import lonlat2meters, meters2lonlat
import pickle
import os
import functools
random.seed(1996)


def data_augmentation(name):
    if name == "distort":
        return distort
    elif name == "downsampling":
        return downsampling
    elif name == "original":
        return original


def original(locations):
    return locations


def distort_seq(seq, rate, radius=1):
    noise_seq = []
    for i in range(0, len(seq)):
        if random.random() <= rate:
            noise = random.gauss(0, 1)
            noise_seq.append(max(0, int(round(seq[i] + radius * noise, 0))))
        else:
            noise_seq.append(seq[i])
    return noise_seq


def downsampling_seq(seq, rate, mode='stochastic'):
    """
    Downsample traj data in a stochastic way based on the rate,
    or keep fix_length of subtrajectory based on the rate
    :param traj: Array of traj GPS coordinate data since each traj has variant length
    :param rate:
    :param mode:
    :return:
    """

    ori_length = len(seq)
    if mode == 'stochastic':
        idx = np.random.rand(ori_length) > rate
        idx[0], idx[-1] = True, True  # Keep the start and end
    elif mode == 'fixed_length':
        idx = random.sample(range(1, ori_length - 1), int(ori_length * (1 - rate)) - 2)
        idx += [0, ori_length - 1]
        idx.sort()
    return list(np.array(seq)[idx])


def distort(locations, rate, radius_loc=30, mu=0, sigma=1):
    """
    Add Gaussian Noise (Default is (0, 1)) to traj data.
    :param locations:
    :param rate:
    :param radius_loc: Offset of location noise, default is 30
    :param mu: Parameter for Gaussian Noise
    :param sigma: Parameter for Gaussian Noise
    :return: Array of distorted traj GPS coordinate data
    """
    noise_locations = copy.copy(locations)

    for i in range(0, len(noise_locations)):
        if rate == 'random':
            distort_rate = random.randint(1, 6) / 10
        else:
            distort_rate = rate

        if random.random() <= distort_rate:
            x, y = lonlat2meters(*noise_locations[i, :])
            while True:
                xnoise = random.gauss(mu, sigma) * radius_loc
                if -radius_loc <= xnoise <= radius_loc:
                    break
            while True:
                ynoise = random.gauss(mu, sigma) * radius_loc
                if -radius_loc <= ynoise <= radius_loc:
                    break
            noise_locations[i, :] = meters2lonlat(x + (xnoise), y + (ynoise))
    return noise_locations


def downsampling(locations, rate):
    """
    Downsample traj data in a stochastic way based on the rate,
    or keep fix_length of subtrajectory based on the rate
    :param locations:
    :param rate:
    :return:
    """

    ori_length = len(locations)
    if rate == 'random':
        idx = []
        for i in range(0, len(locations)):
            downsampling_rate = random.randint(1, 6) / 10
            idx.append(random.random() > downsampling_rate)
    else:
        idx = np.random.rand(ori_length) > rate
    idx[0], idx[-1] = True, True  # Keep the start and end

    return locations[idx, :]


def run_single(dataset_name, method):
    start = time.time()
    with open(f"../data/{dataset_name}/{dataset_name}.pkl", "rb") as f:
        df = pickle.load(f)

    # Function for data augmentation
    transformer = functools.partial(data_augmentation(method["name"]), rate=method["rate"])

    # Information for file name
    suffix = f"{method['name']}_rate_{method['rate']}"
    print(f"{suffix} for {dataset_name}")

    aug_folder = f"../data/{dataset_name}/augmentation"
    if not os.path.exists(aug_folder):
        os.mkdir(aug_folder)

    df["Locations"] = df["Locations"].map(lambda traj_str: transformer(np.array(traj_str)))

    with open(os.path.join(aug_folder, f"{dataset_name}_{suffix}.pkl"), "wb") as f:
        pickle.dump(df, f)

    print(f"{suffix} cost: {time.time() - start} s")


if __name__ == "__main__":
    aug_method = {"name": "distort", "rate": 0.6}
    run_single("porto", aug_method)




