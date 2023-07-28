import json
import numpy as np


def length_of_traj(df_location):
    len_list = []
    for traj in df_location:
        if not isinstance(traj, list):
            traj = json.loads(traj)
        len_list.append(len(traj))
    return len_list


def mean_time_interval(df):
    timestamps_list = df['Timestamps'].values
    sum_interval = 0
    num_interval = 0
    for timestamps in timestamps_list:
        if not isinstance(timestamps, list):
            timestamps = json.loads(timestamps)
            sum_temp = 0
            for i in range(1, len(timestamps)):
                sum_temp += (timestamps[i] - timestamps[i - 1])
            sum_interval += sum_temp
            num_interval += len(timestamps) - 1
    mean_interval = sum_interval / num_interval
    return mean_interval


def traj_stats(df):
    # Temporal information
    mean_interval = mean_time_interval(df)

    # Spatial information
    len_list = length_of_traj(df.Locations)
    data = np.array(len_list)
    print(f"We have {df.shape[0]} trajectories")
    print(f"We have {data.sum()} points")
    print(f"mean_length is {round(np.mean(data), 4)}")
    print(f"mean_time_interval is {round(mean_interval, 4)}")
    print(f"scale_point >= 30 is {data[data >= 30].sum()}")
    print(f"min_length is {np.min(data)}")
    print(f"max_length is {np.max(data)}")
    print(f"length>=40000 is {data[data >= 40000].shape[0]}")
    print(f"median length is {np.median(data)}")
    print(f"90% percentile is {np.percentile(data, 90, axis=0)}")
    print(f"95% percentile is {np.percentile(data, 95, axis=0)}")
    print(f"length<30 is {data[data < 30].shape[0]}, {data[data < 30].shape[0] / data.shape[0]:.2%}")
    print(f"length>=30 is {data[data >= 30].shape[0]}, mean_length is {round(np.mean(data[data >= 30]), 4)}")
    length1 = data.shape[0] - data[data < 30].shape[0] - data[data > 4000].shape[0]
    print(f"30<=length<=4000 is {length1}, {length1 / data.shape[0]:.2%}")
    length2 = data.shape[0] - data[data < 30].shape[0] - data[data > 100].shape[0]
    print(f"30<=length<=100 is {length2}, {length2 / data.shape[0]:.2%}")


def stats_run(df, min, max):
    len_list = length_of_traj(df.Locations)
    df["Length"] = len_list
    df_part = df[min <= df["length"] <= max]

    print(f"For whole dataset:")
    traj_stats(df)

    print(f"For partial dataset with length between {min} and {max}:")
    traj_stats(df_part)


