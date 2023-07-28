import sys
sys.path.append("..")
import time
import pandas as pd
import json
import numpy as np
import math
import random
import copy
from utils import lonlat2meters, meters2lonlat
import h5py
import os
import functools
random.seed(1996)


def original(locations, timestamps):
    return locations, timestamps


def distort(locations, timestamps, rate, radius_ts, radius_loc=30, mu=0, sigma=1):
    """
    Add Gaussian Noise (Default is (0, 1)) to traj data.
    :param locations:
    :param timestamps:
    :param rate:
    :param radius_ts: Offset of timestamp noise, default is mean_interval/10
    :param radius_loc: Offset of location noise, default is 30
    :param mu: Parameter for Gaussian Noise
    :param sigma: Parameter for Gaussian Noise
    :return: Array of distorted traj GPS coordinate data
    """
    noise_locations = copy.copy(locations)
    noise_timestamps = copy.copy(timestamps)

    for i in range(0, len(noise_locations)):
        if rate == 'random':
            distort_rate = random.randint(1, 6) / 10
        else:
            distort_rate = rate

        if random.random() <= distort_rate:
            x, y = lonlat2meters(*noise_locations[i, :])
            xnoise, ynoise = random.gauss(mu, sigma), random.gauss(mu, sigma)
            noise_locations[i, :] = meters2lonlat(x + (radius_loc * xnoise), y + (radius_loc * ynoise))
            tnoise = random.gauss(mu, sigma)
            noise_timestamps[i] += tnoise * radius_ts
    return noise_locations, noise_timestamps


def downsampling(locations, timestamps, rate, mode='stochastic'):
    """
    Downsample traj data in a stochastic way based on the rate,
    or keep fix_length of subtrajectory based on the rate
    :param locations:
    :param timestamps:
    :param rate:
    :param mode:
    :return:
    """

    ori_length = len(locations)
    if mode == 'stochastic':
        if rate == 'random':
            idx = []
            for i in range(0, len(locations)):
                downsampling_rate = random.randint(1, 6) / 10
                idx.append(random.random() > downsampling_rate)
        else:
            idx = np.random.rand(ori_length) > rate
        idx[0], idx[-1] = True, True  # Keep the start and end
    elif mode == 'fixed_length':
        idx = random.sample(range(1, ori_length - 1), int(ori_length * (1 - rate)) - 2)
        idx += [0, ori_length - 1]
        idx.sort()

    return locations[idx, :], timestamps[idx]


def data_augmentation(name):
    if name == "distort":
        return distort
    elif name == "downsampling":
        return downsampling
    elif name == "original":
        return original


def run_single(dataset_name, method):
    with h5py.File(f"../data/{dataset_name}/{dataset_name}.h5") as f:
        mean_interval = f.attrs["mean_interval"]
        # Function for data augmentation
        if method["name"] == "distort":
            method["params"]["radius_ts"] = mean_interval
        transformer = functools.partial(data_augmentation(method["name"]), **method["params"])

        # Information for file name
        suffix = method["name"] + '_'
        for key in method["params"]:
            if not key == "radius_ts":
                suffix += (key + "_") + (str(method["params"][key]) + "_")
        suffix = suffix.rstrip("_")
        print(f"{suffix} for {dataset_name}")

        aug_folder = f"../data/{dataset_name}/augmentation"
        if not os.path.exists(aug_folder):
            os.mkdir(aug_folder)
        file_name = os.path.join(aug_folder, f"{dataset_name}_{suffix}.h5")
        start = time.perf_counter()
        if not os.path.exists(file_name):
            f_new = h5py.File(file_name, "w")
            num = f.attrs["num"]
            for i in range(num):
                locations = np.array(f[f"/Locations/{i}"])
                timestamps = np.array(f[f"/Timestamps/{i}"])
                locations_new, timestamps_new = transformer(locations, timestamps)
                f_new[f"/Locations/{i}"] = locations_new
                f_new[f"/Timestamps/{i}"] = timestamps_new
                f_new[f"/Type/{i}"] = str(f[f"/Type/{i}"])
                if i % 100000 == 0 and i != 0:
                    print(f"Processed {i} trajectories, avg time is {(time.perf_counter() - start) / i},"
                          f" job is expected to be finished in {((time.perf_counter() - start) / i) * (num - i)} s")
            f_new.attrs["num"] = num
            f_new.close()
    print(f"{suffix} cost: {time.perf_counter() - start} s")


def run_mix(dataset_name, method1, method2):
    with h5py.File(f"../data/{dataset_name}/{dataset_name}.h5") as f:
        mean_interval = f.attrs["mean_interval"]
        # Function for data augmentation
        if method1["name"] == "distort":
            method1["params"]["radius_ts"] = mean_interval
        transformer1 = functools.partial(data_augmentation(method1["name"]), **method1["params"])
        if method2["name"] == "distort":
            method2["params"]["radius_ts"] = mean_interval
        transformer2 = functools.partial(data_augmentation(method2["name"]), **method2["params"])

        # Information for file name
        suffix = method1["name"] + '_'
        for key in method1["params"]:
            if not key == "radius_ts":
                suffix += (key + "_") + (str(method1["params"][key]) + "_")
        suffix += method2["name"] + '_'
        for key in method2["params"]:
            if not key == "radius_ts":
                suffix += (key + "_") + (str(method2["params"][key]) + "_")
        suffix = suffix.rstrip("_")
        print(f"{suffix} for {dataset_name}")

        aug_folder = f"../data/{dataset_name}/augmentation"
        if not os.path.exists(aug_folder):
            os.mkdir(aug_folder)
        file_name = os.path.join(aug_folder, f"{dataset_name}_{suffix}.h5")
        start = time.perf_counter()
        if not os.path.exists(file_name):
            f_new = h5py.File(file_name, "w")
            num = f.attrs["num"]
            for i in range(num):
                locations = np.array(f[f"/Locations/{i}"])
                timestamps = np.array(f[f"/Timestamps/{i}"])
                locations_new, timestamps_new = transformer2(*transformer1(locations, timestamps))
                f_new[f"/Locations/{i}"] = locations_new
                f_new[f"/Timestamps/{i}"] = timestamps_new
                f_new[f"/Type/{i}"] = str(f[f"/Type/{i}"])
                if i % 100000 == 0 and i != 0:
                    print(f"Processed {i} trajectories, avg time is {(time.perf_counter() - start) / i},"
                          f" job is expected to be finished in {((time.perf_counter() - start) / i) * (num - i)} s")
            f_new.attrs["num"] = num
            f_new.close()
    print(f"{suffix} cost: {time.perf_counter() - start} s")


