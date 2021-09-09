import pandas as pd
import numpy as np
import datetime
import re
import polyline
import matplotlib.pyplot as plt
import pickle

from sklearn.cluster import KMeans
from geopy.geocoders import Nominatim
from tqdm import tqdm

import config

tqdm.pandas()


def format_date(df, column_name="start_date"):
    """
    :param df: dataframe that needs date column formatted 
    :param column_name: column name that needs date formatted
    :param date: date in format yy-mm-ddThh:mm:ssZ
    :return: date column in yy-mm-dd and time column in hh:mm:ss
    """

    def format_date_helper(date):
        date = date.split("T")
        day, time = date[0], date[1][:-1]
        return day, time

    df["temp"] = df["start_date"].apply(lambda x: format_date_helper(x))
    df["temp_local"] = df["start_date_local"].apply(lambda x: format_date_helper(x))
    df[["GMT_date", "GMT_time"]] = pd.DataFrame(df.temp.values.tolist(), index=df.index)
    df[["local_date", "local_time"]] = pd.DataFrame(
        df.temp_local.values.tolist(), index=df.index
    )
    df = df.drop(["temp", "temp_local", "start_date", "start_date_local"], axis=1)
    return df


def format_timezone(df, column_name="timezone"):
    """
    :param df: dataframe that needs timezone column formatted 
    :param column_name: column name that needs timezone formatted
    :param timezone: timezone to be stripped of (...) e.g. (GMT-05:00) America/New_York -> America/New_York  
    :return: formatted timezone 
    """

    def format_timezone_helper(timezone):
        timezone = re.sub(r"\([^)]*\) ", "", timezone)
        return timezone

    df[column_name] = df[column_name].apply(lambda x: format_timezone_helper(x))
    return df


def uk_awake_feature(df):
    """
    A binary feature to see if run was in U.K. awake hours or not
    """

    def is_awake(time_element):
        time_element = datetime.datetime.time(
            datetime.datetime.strptime(time_element, "%H:%M:%S")
        )
        wake = datetime.time(8, 00, 00)
        sleep = datetime.time(22, 59, 59)
        return int(time_element > wake and time_element < sleep)

    df["is_uk_awake"] = df["GMT_time"].apply(lambda x: is_awake(x))
    return df


def generate_time_features(df):
    # create time based features using date columns
    df.loc[:, "datetime"] = pd.to_datetime(df["local_date"] + " " + df["local_time"])
    df.loc[:, "year"] = df["datetime"].dt.year
    df.loc[:, "weekofyear"] = df["datetime"].dt.isocalendar().week
    df.loc[:, "month"] = df["datetime"].dt.month
    df.loc[:, "dayofweek"] = df["datetime"].dt.dayofweek
    df.loc[:, "weekend"] = (df.datetime.dt.weekday >= 5).astype(int)
    df.loc[:, "hour"] = df["datetime"].dt.hour
    df = uk_awake_feature(df)
    return df


def area_enclosed_by_run(poly, display=False):
    """
    Calculated the area enclosed by runs gps trace.
    :param poly: polyline from gps trace
    :param display: bool, True outputs a plot of gps trace
    :return: area enclosed 
    """

    def computeArea(pos):
        x, y = zip(*pos)
        return 0.5 * np.abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))

    if isinstance(poly, str):
        poly_decoded = polyline.decode(poly)
        lat, lon = list(zip(*poly_decoded))
        lat, lon = list(lat), list(lon)
        if display:
            fig = plt.figure(figsize=(12, 12))
            fig.suptitle("Strava Activity polymap")
            ax = plt.Axes(fig, [0.0, 0.0, 1.0, 1.0],)
            ax.set_aspect("equal")
            ax.set_axis_off()
            fig.add_axes(ax)
            ax.plot(lon, lat, lw=0.5, alpha=0.9)
            ax.fill(lon, lat)
            polygon = ax.fill(lon, lat)
        else:
            # pyplot.fill(a, b) will return a list of matplotlib.patches.Polygon.
            polygon = plt.fill(lon, lat)
            plt.close()
        area = computeArea(polygon[0].xy) * 100000

    else:
        area = 0

    return area


def get_poly_areas(df, column_name="map.summary_polyline"):
    """
    :param df: dataframe that needs timezone column formatted 
    :param column_name: column name that contains polylines 
    :return: run gps area
    """

    df["run_area"] = df[column_name].progress_apply(lambda x: area_enclosed_by_run(x))
    df = df.drop(column_name, axis=1)
    return df


def apply_clustering(df):
    """
    :param df: dataframe that needs lat, lng column clustered 
    :return: cluster
    """
    with open("models/cluster_latlng", "rb") as f:
        cluster_latlng = pickle.load(f)

    def apply_clustering_helper(lat, lng, cluster_model, num_clusters=6):
        """
        If there is a latitude and longitude predict the cluster. If not 
        assign to a new cluster.
        :param lat: start latitude of run
        :param lng: start longitude of run 
        :param cluster_model: KMeans clustering model fit on training data
        :param num_clusters: used to assign new cluster to missing data 
        :return: cluster assigned 
        """
        if not np.isnan(lat) and not np.isnan(
            lng
        ):  # and isinstance(lng, numbers.Number):
            val = np.array([lat, lng]).reshape(1, 2)
            cluster = cluster_model.predict(val)[0]
        else:
            cluster = num_clusters
        return cluster

    df["latlng_cluster"] = df.apply(
        lambda row: apply_clustering_helper(
            row["start_latitude"], row["start_longitude"], cluster_latlng
        ),
        axis=1,
    )
    return df


def reverse_geocoder(df):
    """
    reverse geocode the lat, lng coordinates to find the city where the run was
    :param df: dataframe to add new column too
    :return: new dataframe with column 'city'
    """

    geolocator = Nominatim(user_agent="myGeocoder")
    df["temp"] = df["start_latitude"].map(str) + "," + df["start_longitude"].map(str)

    def reverse_geocode_helper(geom):
        try:
            location = geolocator.reverse(geom)
            city = location.raw.get("address").get("city")
            if not city:
                city = location.raw.get("address").get("town")
        except:
            city = np.nan
        return city

    df["city"] = df["temp"].progress_apply(reverse_geocode_helper)
    df = df.drop("temp", axis=1)
    return df


if __name__ == "__main__":
    # load in initial data gathered from the Strava API
    data_path = config.STRAVA_DATA_PATH
    data_init = pd.read_csv(data_path, index_col=0)

    # filter out runs from activities and drop type column
    data_init = data_init.loc[data_init.type == "Run"]
    data_init = data_init.drop(columns=["type"])

    # remove private activities and drop older activities
    data_init = (
        data_init.loc[data_init.private == False].iloc[0:1125].drop("private", axis=1)
    )
    data_init = format_date(data_init)
    data_init = format_timezone(data_init)
    data_init = generate_time_features(data_init)
    print("Adding run area feature...")
    data_init = get_poly_areas(data_init)
    data_init = apply_clustering(data_init)
    print("Adding reverse geocoder feature...")
    data_init = reverse_geocoder(data_init)

    # convert distance in meters to kilometers
    data_init.loc[:, "distance"] = data_init.distance / 1000
    # convert moving in seconds time to minutes
    data_init.loc[:, "moving_time"] = data_init.moving_time / 60
    # convert average speed in meters per second to minutes per kilometer
    data_init.loc[:, "average_speed_mpk"] = 16.666 / data_init.average_speed
    data_init = data_init.drop("average_speed", axis=1)
    # remove runs with no kudos as it wasn't available to public
    data_init = data_init[data_init.kudos_count != 0]

    # drop unwanted/useless columns from the dataset
    cols_to_drop = [
        "resource_state",
        "id",
        "external_id",
        "upload_id",
        "utc_offset",
        "start_latlng",
        "end_latlng",
        "trainer",
        "commute",
        "visibility",
        "flagged",
        "gear_id",
        "from_accepted_tag",
        "upload_id_str",
        "average_cadence",
        "average_temp",
        "has_heartrate",
        "heartrate_opt_out",
        "display_hide_heartrate_option",
        "elev_high",
        "elev_low",
        "has_kudoed",
        "athlete.id",
        "athlete.resource_state",
        "map.id",
        "map.resource_state",
        "device_watts",
        "average_watts",
        "kilojoules",
        "photo_count",
        "athlete_count",
        "location_city",
        "location_state",
        "location_country",
        "comment_count",
        "elapsed_time",
        "start_latitude",
        "start_longitude",
    ]

    data_init = data_init.drop(cols_to_drop, axis=1)

    # split data into training and testing set
    test_frac = 0.25

    data_train = data_init[int(data_init.shape[0] * test_frac) :]
    data_test = data_init[: int(data_init.shape[0] * test_frac)]

    print("train: ", data_train.shape)
    print("test: ", data_test.shape)

    data_train.to_csv("input/data_train.csv")
    data_test.to_csv("input/data_test.csv")
    print("Data preprocessed and saved to file 'data_train.csv' and 'data_test.csv'")
