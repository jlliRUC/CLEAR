import numpy as np
import random
import math
from datetime import datetime
from pytz import timezone
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






