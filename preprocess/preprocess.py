import sys
sys.path.append("..")
import pandas as pd
import os
from utils import str2ts
import json
import glob
import time
import pickle
import h5py
import numpy as np
from config import Config
import argparse


def parse_args():
    # Set some key configs from screen

    parser = argparse.ArgumentParser(description="preprocess.py")

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


    args = parser.parse_args()

    params = {}
    for param, value in args._get_kwargs():
        if value is not None:
            params[param] = value

    return params


"""
Format unifying
# Different datasets may be stored in different formats.
# We'll unify the format of each dataset to dataframe with unified columns
# ["Traj_ID", "Obj_ID", "Timestamps", "Locations", "Type", "Missing_Data"].
# All data in the original files will be left for further filtering.
"""


def unify(dataset_name):
    if dataset_name == "porto":
        return porto_unify
    elif dataset_name == "geolife":
        return geolife_unify
    else:
        print("Unknown dataset for unify!")
        return


def porto_unify(path="../data/porto/porto.csv"):
    """
    Porto is originally stored in .csv format. We need to
    1. extract necessary columns and unify the column names.
    2. There is only the start timestamp for each trajectory, we can't filter.
       Even worse, we need to complete the rest timestamps for each trajectory.
    :param path:
    :return:
    """

    df_porto_ori = pd.read_csv(path)
    # Create Timestamps & Type
    ts_list = []
    type_list = []
    index = []
    for i in range(df_porto_ori.shape[0]):
        length = len(json.loads(df_porto_ori.loc[i, "POLYLINE"]))
        if length > 0:
            index.append(i)
            ts_start = df_porto_ori.loc[i, "TIMESTAMP"]
            timestamps = [ts_start]
            for j in range(1, length):
                timestamps.append(ts_start+j*15)  # According to the description of the dataset, time interval is evenly 15s
            ts_list.append(timestamps)
            type_list.append("Unknown")  # No ground-truth type for Porto
    # Remove empty records, extract necessary columns and unify the column names
    df_porto_ori = df_porto_ori.loc[index, :]
    df_porto_ori["Timestamps"] = ts_list
    df_porto_ori["Type"] = type_list
    df_porto_new = df_porto_ori[["TRIP_ID", "TAXI_ID", "Timestamps", "POLYLINE", "Type", "MISSING_DATA"]]
    df_porto_new.columns = ["Traj_ID", "Obj_ID", "Timestamps", "Locations", "Type", "Missing_Data"]
    df_porto_new.to_csv(f"../data/porto/porto_unify.csv", index=False)


def geolife_unify(path="../data/geolife/Data"):
    """
    GeoLife is originally stored in a folder,
    each sub-folder represents one object, each .plt file in sub-folder stores a trajectory of this object.
    1. The first six lines of each file are meaningless
    2. Three-steps filter: duplicate-filter (same location, same timestamp), missing-filter (time interval is too long), stop-filter (too many continuous points with same location but different timestamps)
    3. No missing data
    4. Some of the records have types.
    :param path:
    :return:
    """
    dir_0 = path
    file_list = []
    for user_id in os.listdir(dir_0):
        if not user_id == '.DS_Store':
            for traj_file in os.listdir(os.path.join(dir_0, user_id, "Trajectory")):
                full_path = os.path.join(os.path.join(dir_0, user_id, "Trajectory", traj_file))
                lines = open(full_path, "r").readlines()[6:]
                if not len(lines) == 0:
                    file_list.append(full_path)

    traj_ID_list = []
    obj_ID_list = []
    timestamp_list = []
    location_list = []
    type_list = []
    missing_data_list = []

    for file in file_list:
        obj_ID = file.split('/')[-3]
        lines = open(file, "r").readlines()[6:]

        ts_list = []
        point_list = []
        for i in range(len(lines)):
            line = lines[i].removesuffix("\n").split(",")
            ts = str2ts(f"{line[-2]} {line[-1]}", "%Y-%m-%d %H:%M:%S")
            point = [json.loads(line[1]), json.loads(line[0])]  # Field 1 is latitude, Field 2 is longitude
            ts_list.append(ts)
            point_list.append(point)
        traj_ID_list.append(obj_ID)  # Each file represents one trajectory, thus traj_ID = obj_ID
        obj_ID_list.append(obj_ID)
        timestamp_list.append(ts_list)
        location_list.append(point_list)
        missing_data_list.append("False")
        # Type. For each object, there might be a "labels.txt" file about the trajectory type of each period.
        label_file = os.path.join(dir_0, obj_ID, "labels.txt")
        if os.path.exists(label_file):
            start_temp = file.split('/')[-1].strip(".plt")
            start_time = f"{start_temp[:4]}/{start_temp[4:6]}/{start_temp[6:8]} {start_temp[8:10]}:{start_temp[10:12]}:{start_temp[12:14]}"
            df1 = pd.read_csv(label_file, sep='\t')
            if start_time in list(df1["Start Time"]):
                label = df1[df1["Start Time"] == start_time]["Transportation Mode"].values[0]
                type_list.append(label)
            else:
                type_list.append("Unknown")  # Current start time is not in the records.
        else:
            type_list.append("Unknown")  # No "labels.txt"

    dataset_dict = {"Traj_ID": traj_ID_list,
                    "Obj_ID": obj_ID_list,
                    "Timestamps": timestamp_list,
                    "Locations": location_list,
                    "Type": type_list,
                    "Missing_Data": missing_data_list}
    df = pd.DataFrame.from_dict(dataset_dict)
    print(df.shape)
    df.to_csv(f"../data/geolife/geolife_unify.csv", index=False)


"""
Filter
Many datasets put all points of one object as a whole trajectory, which doesn't make sense.
Normally, a taxi or a pedestrian shouldn't move for more than one day. 
Their locations also shouldn't stop for a long time. 
For a taxi, when it finishes current business, it may stop and wait for the next customer, 
which implies another trajectory. Therefore, we define three rules to filter the trajectories in {}_unify.csv:
1. Duplicate: same location, same timestamp. We only keep the first one.
2. Missing: The time interval between two continuous points is longer than the threshold, 
            we can't rely on existing points to depict the whole trajectory. 
            It's better to cut the whole trajectory between these two points.
3. Stop: same location, different timestamp. 
         The object might stay at one position for a preriod due to traffic jam. 
         But if the stay time is longer than the threshold, we argue it cannot be considered as a whole trajectory 
         since there are some stops. We'll keep the short stay and split the whole trajectory from those stops.
Besides, due to the limitation of model training, we need to further filter by trajectory's location and length
4. Length: Few trajectories are too long or short, which are hard for model training, especially long trajectories for transformers.
5. Inregion: For general models who deals with out-of-region point as <UNK> (e.g., t2vec), it's not necessary to do this.
             While for models who rely on cell pretraining, we need to filter out those trajectories who are not (partially and fully) in the region.
             Otherwise  we couldn't obtain the cell embedding for the out-of-region points.
"""


# 1. Duplicate Filter (same location, same timestamp)
def filter_duplicate(df):
    result_df = pd.DataFrame.from_dict({"Traj_ID": [],
                                        "Obj_ID": [],
                                        "Timestamps": [],
                                        "Locations": [],
                                        "Type": [],
                                        "Missing_Data": []})
    duplicate_list = []
    for i in range(0, df.shape[0]):
        duplicate_num = 0
        timestamps = df.loc[i, "Timestamps"]
        if not isinstance(timestamps, list):
            timestamps = json.loads(timestamps)
        locations = df.loc[i, "Locations"]
        if not isinstance(locations, list):
            locations = json.loads(locations)
        new_timestamps = [timestamps[0]]
        new_locations = [locations[0]]
        for j in range(1, len(locations)):
            ts0 = timestamps[j - 1]
            ts1 = timestamps[j]
            location0 = locations[j - 1]
            location1 = locations[j]
            if ts0 == ts1 and location0 == location1:
                duplicate_num += 1
                continue
            else:
                new_timestamps.append(timestamps[j])
                new_locations.append(locations[j])
        duplicate_list.append(duplicate_num)
        result_df.loc[len(result_df.index)] = [df.loc[i, "Traj_ID"],
                                               df.loc[i, "Obj_ID"],
                                               new_timestamps,
                                               new_locations,
                                               df.loc[i, "Type"],
                                               df.loc[i, "Missing_Data"]]
    print(f"This dataset has {sum(duplicate_list)} duplicate points")
    return result_df


# 2. Missing Filter
def filter_missing(df, threshold):
    result_df = pd.DataFrame.from_dict({"Traj_ID": [],
                                        "Obj_ID": [],
                                        "Timestamps": [],
                                        "Locations": [],
                                        "Type": [],
                                        "Missing_Data": []})
    subtraj_list = []
    for i in range(0, df.shape[0]):
        subtraj_num = 0
        timestamps = df.loc[i, "Timestamps"]
        if not isinstance(timestamps, list):
            timestamps = json.loads(timestamps)
        locations = df.loc[i, "Locations"]
        if not isinstance(locations, list):
            locations = json.loads(locations)

        sub_list = []
        subtraj_temp = [0]
        for j in range(1, len(timestamps)):
            if timestamps[j] - timestamps[j - 1] > threshold:  # Missing, split
                sub_list.append(subtraj_temp)
                subtraj_temp = [j]
            else:
                subtraj_temp.append(j)
        if len(subtraj_temp) != 0:  # The tail
            sub_list.append(subtraj_temp)

        for sub_traj in sub_list:
            new_timestamps = []
            new_locations = []
            for sub_index in sub_traj:
                new_timestamps.append(timestamps[sub_index])
                new_locations.append(locations[sub_index])
            result_df.loc[len(result_df.index)] = [f"{df.loc[i, 'Traj_ID']}-{subtraj_num}",
                                                   df.loc[i, "Obj_ID"],
                                                   new_timestamps,
                                                   new_locations,
                                                   df.loc[i, "Type"],
                                                   df.loc[i, "Missing_Data"]]
            subtraj_num += 1
        subtraj_list.append(subtraj_num)
    print(f"This dataset has {sum(subtraj_list)} missing breaks")
    return result_df


# 3. Stop Filter
def filter_stop(df, threshold):
    result_df = pd.DataFrame.from_dict({"Traj_ID": [],
                                        "Obj_ID": [],
                                        "Timestamps": [],
                                        "Locations": [],
                                        "Type": [],
                                        "Missing_Data": []})
    subtraj_list = []
    for i in range(0, df.shape[0]):
        subtraj_num = 0
        timestamps = df.loc[i, "Timestamps"]
        if not isinstance(timestamps, list):
            timestamps = json.loads(timestamps)
        locations = df.loc[i, "Locations"]
        if not isinstance(locations, list):
            locations = json.loads(locations)

        stay_list = []
        stay_time = 0
        sub_list = []
        subtraj_temp = [0]
        for j in range(1, len(locations)):
            if locations[j] == locations[j - 1]:  # Stay points
                stay_time += timestamps[j] - timestamps[j - 1]
                stay_list.append(j)
            else:
                if stay_time == 0:  # Nothing, current trajectory continues
                    subtraj_temp.append(j)
                elif stay_time > threshold:  # Long stay period (stop), split traj
                    sub_list.append(subtraj_temp)
                    subtraj_temp = [j]
                    stay_time = 0
                    stay_list = []
                else:  # Short stay period, remain it in current trajectory and continue
                    for item in stay_list:
                        subtraj_temp.append(item)
                    subtraj_temp.append(j)
                    stay_time = 0
                    stay_list = []
        if len(stay_list) != 0:
            for item in stay_list:
                subtraj_temp.append(item)
        if len(subtraj_temp) != 0:
            sub_list.append(subtraj_temp)

        for sub_traj in sub_list:
            new_timestamps = []
            new_locations = []
            for sub_index in sub_traj:
                new_timestamps.append(timestamps[sub_index])
                new_locations.append(locations[sub_index])
            result_df.loc[len(result_df.index)] = [f"{df.loc[i, 'Traj_ID']}-{subtraj_num}",
                                                   df.loc[i, "Obj_ID"],
                                                   new_timestamps,
                                                   new_locations,
                                                   df.loc[i, "Type"],
                                                   df.loc[i, "Missing_Data"]]
            subtraj_num += 1
        subtraj_list.append(subtraj_num)
    print(f"This dataset has {sum(subtraj_list)} stay breaks")
    return result_df


def script_filter(dataset_name, filter_list=["duplicate", "stop", "missing"]):
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

        return df


# 4. Incomplete Filter
def filter_incomplete(df):
    return df[df.Missing_Data == False].reset_index(
        drop=True)  # Delete the incomplete traj. Note that len(traj)==0 doesn't mean MISSING_DATA==TRUE


# 5. Length Filter
def filter_length(df, length_min=30, length_max=2000):
    df["Length"] = df.Locations.apply(lambda traj: len(traj))

    # Length filter
    df = df[(df.Length >= length_min) & (df.Length <= length_max)]
    return df


# 6. Inregion Filter
def point_in_region(region_settings, lon, lat):
    if lon <= region_settings['min_lon'] or lon >= region_settings['max_lon'] \
            or lat <= region_settings['min_lat'] or lat >= region_settings['max_lat']:
        return False
    return True


def filter_inregion(df, region_settings):
    # Region filter  # For cell pretraining, all points need to be within the region
    df["inregion"] = df["Locations"].map(lambda traj: sum([point_in_region(region_settings, point[0], point[1]) for point in traj]) == len(traj))
    return df[df.inregion == True]


if __name__ == "__main__":
    configs = Config()
    configs.default_update(parse_args())
    configs.config_dataset()
    dataset_name = configs.dataset_name
    region_settings = {"min_lon": configs.min_lon,
                       "max_lon": configs.max_lon,
                       "min_lat": configs.min_lat,
                       "max_lat": configs.max_lat}

    t0 = time.time()

    # Unify
    if not os.path.exists(f"../data/{dataset_name}/{dataset_name}_unify.csv"):
        unify_func = unify(dataset_name)
        unify_func()
    t1 = time.time()
    print(f"Unify costs: {t1 - t0} s")

    # Filter
    if os.path.exists(f"../data/{dataset_name}/{dataset_name}_filter.csv"):
        df = pd.read_csv(f"../data/{dataset_name}/{dataset_name}_filter.csv")
    else:
        df = script_filter(dataset_name)
    t2 = time.time()
    print(f"Filter stage 1 costs {t2 - t1} s")
    df["Locations"] = df["Locations"].map(lambda traj_str: np.array(json.loads(traj_str)))
    df_full = df[["Traj_ID", "Obj_ID", "Timestamps", "Locations", "Type"]]
    df_full = df_full.reset_index(drop=True)
    with open(f"../data/{dataset_name}/{dataset_name}_full.pkl", "wb") as f:
        pickle.dump(df_full, f)

    df = filter_incomplete(df)
    df = filter_length(df)
    print(f"After filtering by length: {df.shape[0]}")
    df = filter_inregion(df, region_settings)
    t3 = time.time()
    print(f"Filter stage 2 costs {t3 - t2} s")
    print(f"We have {df.shape[0]} trajectories in total")
    # Save
    df = df[["Traj_ID", "Obj_ID", "Timestamps", "Locations", "Type", "Length"]].reset_index(drop=True)
    with open(f"../data/{dataset_name}/{dataset_name}.pkl", "wb") as f:
        pickle.dump(df, f)
    t4 = time.time()
    print(f"Preprocess for {dataset_name} costs {t4 - t0} s")


