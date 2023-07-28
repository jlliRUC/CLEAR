import pandas as pd
import json
"""
Many datasets put all points of one object as a whole trajectory, which doesn't make sense.
Normally, a taxi or a pedestrian shouldn't move for more than one day. 
Their locations also shouldn't stop for a long time. 
For a taxi, when it finishes current business, it may stop and wait for the next customer, 
which implies another trajectory. Therefore, we define three rules to filter the trajectories in ori.csv:
1. Duplicate: same location, same timestamp. We only keep the first one.
2. Missing: The time interval between two continuous points is longer than the threshold, 
            we can't rely on existing points to depict the whole trajectory. 
            It's better to cut the whole trajectory between these two points.
3. Stop: same location, different timestamp. 
         The object might stay at one position for a preriod due to traffic jam. 
         But if the stay time is longer than the threshold, we argue it cannot be considered as a whole trajectory 
         since there are some stops. We'll keep the short stay and split the whole trajectory from those stops.
"""


# Filtering
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
