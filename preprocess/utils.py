import numpy as np
import random
import math
from datetime import datetime
from pytz import timezone
import json
random.seed(2023)


# Time convert
tz_utc = timezone("UTC")


def str2ts(timestring, format, tz=tz_utc):
    return datetime.timestamp(datetime.strptime(timestring, format).replace(tzinfo=tz))


def ts2str(timestamp, format, tz=tz_utc):
    return datetime.fromtimestamp(timestamp).astimezone(tz).strftime(format)


def lonlat2meters(lon, lat):
    """
    Convert location point from GPS coordinate to meters
    :param lon:
    :param lat:
    :return:
    """
    semimajoraxis = 6378137.0
    east = lon * 0.017453292519943295
    north = lat * 0.017453292519943295
    t = np.sin(north)
    return semimajoraxis * east, 3189068.5 * math.log((1 + t) / (1 - t))


def meters2lonlat(x, y):
    """
    Convert location point from meters to GPS coordinate
    :param lon:
    :param lat:
    :return:
    """
    semimajoraxis = 6378137.0
    lon = x / semimajoraxis / 0.017453292519943295
    t = math.exp(y / 3189068.5)
    lat = math.asin((t - 1) / (t + 1)) / 0.017453292519943295
    #import pyproj
    #proj = pyproj.Transformer.from_crs(3857, 4326, always_xy=True)
    #lon, lat = proj.transform(x, y)
    return lon, lat


def boundary(df):
    """
    For those datasets that not are not located in one city, we calculate the boundary of longitude and latitude.
    Note that there may be some outliers.
    :return:
    """
    df = df["Locations"]
    min_lon = [1000, 1000, 1000, 1000, 1000]
    min_lon_index = [0, 0, 0, 0, 0]
    min_lat = [1000, 1000, 1000, 1000, 1000]
    min_lat_index = [0, 0, 0, 0, 0]
    max_lon = [-1000, -1000, -1000, -1000, -1000]
    max_lon_index = [0, 0, 0, 0, 0]
    max_lat = [-1000, -1000, -1000, -1000, -1000]
    max_lat_index = [0, 0, 0, 0, 0]
    for i in range(0, df.shape[0]):
        traj = df.iloc[i]
        if not isinstance(traj, list):
            traj = json.loads(traj)
        for item in traj:
            if item[0] < max(min_lon):
                min_lon[min_lon.index(max(min_lon))] = item[0]
                min_lon_index[min_lon.index(max(min_lon))] = i
            if item[0] > min(max_lon):
                max_lon[max_lon.index(min(max_lon))] = item[0]
                max_lon_index[max_lon.index(min(max_lon))] = i
            if item[1] < max(min_lat):
                min_lat[min_lat.index(max(min_lat))] = item[1]
                min_lat_index[min_lat.index(max(min_lat))] = i
            if item[1] > min(max_lat):
                max_lat[max_lat.index(min(max_lat))] = item[1]
                max_lat_index[max_lat.index(min(max_lat))] = i

    return min_lon, min_lat, max_lon, max_lat





