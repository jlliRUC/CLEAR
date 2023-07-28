import time
import h5py
import pandas as pd
import json
import numpy as np
from statistics import mean_time_interval
import argparse

parser = argparse.ArgumentParser(description="csv2h5.py")
parser.add_argument("-dataset_name", help="Name of dataset")


def csv2h5(dataset_name, min=30, max=4000):
    """
    Transform CSV to h5 for python
    :param args:
    :return:
    """
    start = time.time()
    csv_path = f"../data/{dataset_name}/{dataset_name}_filter.csv"
    df = pd.read_csv(csv_path)
    df = df[df.Missing_Data == False].reset_index(drop=True) # Delete the incomplete traj. Note that len(traj)==0 doesn't mean MISSING_DATA==TRUE
    print(f"Processing {df.shape[0]} trajectories")
    f_full = h5py.File(f"../data/{dataset_name}/{dataset_name}_full.h5", "w")  # full dataset for building vocabulary
    f_filter = h5py.File(f"../data/{dataset_name}/{dataset_name}.h5", "w")  # filter dataset for token_generation generation and model training
    num_full, num_filter, num_incompleted, num_short = 0, 0, 0, 0,
    for i in range(df.shape[0]):
        locations = df.loc[i, "Locations"]
        timestamps = df.loc[i, "Timestamps"]
        try:
            if not isinstance(locations, list):
                locations = json.loads(locations)
            timestamps = df.loc[i, "Timestamps"]
            if not isinstance(timestamps, list):
                timestamps = json.loads(timestamps)
        except Exception:
            num_incompleted += 1
        traj_length = len(locations)
        locations = np.array(locations)
        timestamps = np.array(timestamps)
        if traj_length == 0:
            num_short += 1
            continue
        elif min <= traj_length <= max:
            f_full[f"/Locations/{num_full}"] = locations
            f_full[f"/Timestamps/{num_full}"] = timestamps
            f_full[f"/Type/{num_full}"] = df.loc[i, "Type"]
            f_filter[f"/Locations/{num_filter}"] = locations
            f_filter[f"/Timestamps/{num_filter}"] = timestamps
            f_filter[f"/Type/{num_filter}"] = df.loc[i, "Type"]
            num_full += 1
            num_filter += 1
        else:
            f_full[f"/Locations/{num_full}"] = locations
            f_full[f"/Timestamps/{num_full}"] = timestamps
            f_full[f"/Type/{num_full}"] = df.loc[i, "Type"]
            num_full += 1
            num_short += 1
        if num_full % 100000 == 0 and num_full != 0:
            print(f"Processed {num_full} trajectories, avg time is {(time.perf_counter()-start)/num_full}, "
                  f"job is expected to be finished in {((time.perf_counter() - start) / num_full) * (df.shape[0] - i)} s")
    f_full.attrs["num"] = num_full
    f_filter.attrs["num"] = num_filter
    mean_interval = mean_time_interval(df)
    f_full.attrs["mean_interval"] = round(mean_interval, 4)
    f_filter.attrs["mean_interval"] = round(mean_interval, 4)
    f_full.close()
    f_filter.close()
    print(f"Incomplete traj: {num_incompleted}, Short traj: {num_short}, Saved traj: {num_full}, training traj: {num_filter}")
    print(f"csv2h5 time cost: {time.time()-start} s")


if __name__ == '__main__':
    args = parser.parse_args()
    csv2h5(args.dataset_name)
