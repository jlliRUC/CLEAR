from FormatUnify import *
from Filter import filter_stop, filter_duplicate, filter_missing
from csv2h5 import csv2h5
import pandas as pd
import argparse
import time
import os

parser = argparse.ArgumentParser(description="execute_unify.py")
parser.add_argument("-dataset_name", help="Name of dataset")


def unify(dataset_name):
    print(dataset_name)
    print(f"Processing {dataset_name}")
    start = time.time()
    if dataset_name.startswith("ais"):
        folder = f"../data/{dataset_name}/data"
        df_list = []
        for file in os.listdir(folder):
            if not file.endswith("unify.csv") and not file.endswith("filter.csv"):
                if dataset_name == "aisus":
                    df = aisus_unify(os.path.join(folder, file))
                print(f"Unify completed: {time.time() - start}")
                df_list.append(df)
        df_unify = pd.concat(df_list)
        df_unify.to_csv(f"../data/{dataset_name}/{dataset_name}_unify.csv")

    elif dataset_name == "porto":
        porto_unify("../data/porto/porto.csv")
    elif dataset_name == "geolife":
        geolife_unify("../data/geolife/Data")


def filter(dataset_name, filter_list=["duplicate", "stop", "missing"]):
    print(dataset_name)
    print(f"Processing {dataset_name}")
    if dataset_name.startswith("geolife-labeled"):
        threshold_missing = 60
        threshold_stop = 1800
    else:
        threshold_missing = 1800
        threshold_stop = 1800

    if dataset_name.startswith("ais"):
        folder = f"../data/{dataset_name}/data"
        df_list = []
        for file in os.listdir(folder):
            if file.endswith("unify.csv"):
                df = pd.read_csv(os.path.join(folder, file))
                for filter in filter_list:
                    start = time.time()
                    if filter == "duplicate":
                        df = filter_duplicate(df)
                    elif filter == "missing":
                        df = filter_missing(df, threshold=threshold_missing)
                    elif filter == "stop":
                        df = filter_stop(df, threshold=threshold_stop)
                    print(f"{filter} completed: {time.time() - start}")
                df.to_csv(f"../data/{dataset_name}/data/{file.split('_unify.csv')[-2]}_filter.csv", index=False)
                df_list.append(df)
        df_filter = pd.concat(df_list)
        df_filter.to_csv(f"../data/{dataset_name}/{dataset_name}_filter.csv", index=False)

    else:
        df = pd.read_csv(f"../data/{dataset_name}/{dataset_name}_unify.csv")
        for filter in filter_list:
            start = time.time()
            if filter == "duplicate":
                df = filter_duplicate(df)
            elif filter == "missing":
                df = filter_missing(df, threshold=threshold_missing)
            elif filter == "stop":
                df = filter_stop(df, threshold=threshold_stop)
            print(f"{filter} completed: {time.time() - start}")
        df.to_csv(f"../data/{dataset_name}/{dataset_name}_filter.csv", index=False)


if __name__ == '__main__':
    args = parser.parse_args()
    unify(args.dataset_name)
    filter(args.dataset_name)
    csv2h5(args.dataset_name)