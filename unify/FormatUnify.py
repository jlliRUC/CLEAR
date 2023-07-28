import sys
sys.path.append("..")
import pandas as pd
import os
import numpy as np
from utils import str2ts
import json
import glob


# Format unifying
# Different datasets may be stored in different formats.
# We'll unify the format of each dataset to dataframe with unified columns
# ["Traj_ID", "Obj_ID", "Timestamps", "Locations", "Type", "Missing_Data"].
# All data in the original files will be left for further filtering.
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


def rle_segment(seq):
    index = [[0]]
    for i in range(1, len(seq)):
        if seq[i] == seq[i-1]:
            index[-1].append(i)
        else:
            index.append([i])
    return index


def geolife_labeled_unify(path="../data/geolife/Data"):
    subfolders = os.listdir(path)
    subfolders.remove(".DS_Store")
    dfs = []
    mode_names = ['walk', 'bike', 'bus', 'car', 'subway', 'train', 'airplane', 'boat', 'run', 'motorcycle', 'taxi']
    mode_ids = {s: i + 1 for i, s in enumerate(mode_names)}
    traj_ID_list = []
    obj_ID_list = []
    timestamp_list = []
    location_list = []
    type_list = []
    missing_data_list = []

    for i, sf in enumerate(subfolders):
        print('[%d/%d] processing user %s' % (i + 1, len(subfolders), sf))
        user_folder = os.path.join(path, sf)
        plt_files = glob.glob(os.path.join(user_folder, 'Trajectory', '*.plt'))
        labels_file = os.path.join(user_folder, 'labels.txt')
        if os.path.exists(labels_file):
            labels = pd.read_csv(labels_file, skiprows=1, header=None, parse_dates=[[0, 1], [2, 3]],
                                 infer_datetime_format=True, delim_whitespace=True)
            # for clarity rename columns
            labels.columns = ['start_time', 'end_time', 'label']
            # replace 'label' column with integer encoding
            labels['label'] = [mode_ids[i] for i in labels['label']]
        else:
            labels = None

        for f in plt_files:
            # each file from each user can be mapped to multiple trajectories based on the corresponding vehicle
            # label df_points
            df_points = pd.read_csv(f, skiprows=6, header=None, parse_dates=[[5, 6]], infer_datetime_format=True)
            # for clarity rename columns
            df_points.rename(inplace=True, columns={'5_6': 'time', 0: 'lat', 1: 'lon', 3: 'alt'})
            # remove unused columns
            df_points.drop(inplace=True, columns=[2, 4])
            if labels is not None:
                indices = labels['start_time'].searchsorted(df_points['time'], side='right') - 1
                no_label = (indices < 0) | (df_points['time'].values > labels['end_time'].iloc[indices].values)
                df_points['label'] = labels['label'].iloc[indices].values
                df_points['label'][no_label] = 0
            else:
                df_points['label'] = 0
            # df_traj
            index_subtraj = rle_segment(df_points['label'])

            for id, index in enumerate(index_subtraj):
                df_subtraj = df_points.loc[index, :].reset_index(drop=True)
                df_subtraj["timestamps"] = df_subtraj.apply(lambda x: str2ts(f"{x.time}", "%Y-%m-%d %H:%M:%S"), axis=1)
                obj_ID = f.split('/')[-3]
                ts_list = df_subtraj['timestamps'].tolist()
                point_list = [[df_subtraj.loc[i, "lon"], df_subtraj.loc[i, "lat"]] for i in range(df_subtraj.shape[0])]
                traj_ID_list.append(f"{obj_ID}_{id}")
                obj_ID_list.append(obj_ID)
                timestamp_list.append(ts_list)
                location_list.append(point_list)
                missing_data_list.append("False")
                if df_subtraj.loc[0, 'label'] == 0:
                    type_list.append('Unknown')
                else:
                    type_list.append(mode_names[df_subtraj.loc[0, 'label']-1])

    dataset_dict = {"Traj_ID": traj_ID_list,
                    "Obj_ID": obj_ID_list,
                    "Timestamps": timestamp_list,
                    "Locations": location_list,
                    "Type": type_list,
                    "Missing_Data": missing_data_list}
    df = pd.DataFrame.from_dict(dataset_dict)
    df = df[df["Type"]!="Unknown"]
    print(df.shape)
    df.to_csv(f"../data/geolife-labeled/geolife-labeled_unify.csv", index=False)


def tdrive_unify(path="../data/tdrive/taxi_log_2008_by_id"):
    """
    T-Drive is originally stored in a folder, each .txt file in this folder stores a trajectory of one object,
    1. There is no type records or missing_data records.
    :return:
    """
    dir = path
    obj_ID_list = []
    traj_ID_list = []
    timestamp_list = []
    location_list = []
    type_list = []
    missing_data_list = []
    for path in os.listdir(dir):
        if path.endswith(".txt"):
            obj_ID = path.split(".")[0]
            full_path = os.path.join(dir, path)
            lines = open(full_path, "r").readlines()
            if len(lines) == 0:
                continue
            ts_list = []
            point_list = []
            for i in range(len(lines)):
                line = lines[i].removesuffix("\n").split(",")
                ts = str2ts(f"{line[1]}", "%Y-%m-%d %H:%M:%S")
                point = [json.loads(line[-2]), json.loads(line[-1])]
                ts_list.append(ts)
                point_list.append(point)
            traj_ID_list.append(obj_ID)  # Each file represents one trajectory, thus traj_ID = obj_ID
            obj_ID_list.append(obj_ID)
            timestamp_list.append(ts_list)
            location_list.append(point_list)
            missing_data_list.append("False")
            type_list.append("Unknown")

    dataset_dict = {"Traj_ID": traj_ID_list,
                    "Obj_ID": obj_ID_list,
                    "Timestamps": timestamp_list,
                    "Locations": location_list,
                    "Type": type_list,
                    "Missing_Data": missing_data_list}
    df = pd.DataFrame.from_dict(dataset_dict)
    print(df.shape)
    df.to_csv(f"../data/tdrive/tdrive_unify.csv", index=False)


def ais_unify(df, column_date, column_lon, column_lat, column_type, ts_format):
    result_df = pd.DataFrame.from_dict({"Traj_ID": [],
                                        "Obj_ID": [],
                                        "Timestamps": [],
                                        "Locations": [],
                                        "Type": [],
                                        "Missing_Data": []})

    grouped = df.sort_values(by=column_date).groupby('MMSI')
    print(f"We have {len(grouped)} vessels")
    for grouped_ID, grouped_df in grouped:
        grouped_df = grouped_df.reset_index(drop=True)
        obj_ID = grouped_df.loc[0]["MMSI"]
        locations = []
        timestamps = []

        for i in range(0, grouped_df.shape[0]):
            ts_current = str2ts(str(grouped_df.loc[i][column_date]), ts_format)
            timestamps.append(ts_current)
            locations.append([grouped_df.loc[i][column_lon], grouped_df.loc[i][column_lat]])

        missing_data = False
        type = grouped_df.loc[0][column_type]
        result_df.loc[len(result_df.index)] = [obj_ID, obj_ID, timestamps, locations, type, missing_data]
    return result_df


def aisdk_unify(path):
    result_df = ais_unify(df=pd.read_csv(path),
                          column_date="# Timestamp",
                          column_lon="Longitude",
                          column_lat="Latitude",
                          column_type="Ship type",
                          ts_format="%d/%m/%Y %H:%M:%S")
    result_df.to_csv(f"{path.split('.csv')[-2]}_unify.csv", index=False)
    return result_df


def aisus_unify(path):
    result_df = ais_unify(df=pd.read_csv(path),
                          column_date="BaseDateTime",
                          column_lon="LON",
                          column_lat="LAT",
                          column_type="VesselType",
                          ts_format="%Y-%m-%dT%H:%M:%S")
    result_df.to_csv(f"{path.split('.csv')[-2]}__unify.csv", index=False)
    return result_df
