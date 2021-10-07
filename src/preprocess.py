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
            ax = plt.Axes(
                fig,
                [0.0, 0.0, 1.0, 1.0],
            )
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

    df["run_area"] = df[column_name].apply(lambda x: area_enclosed_by_run(x))
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


def label_name(row):
    default_names = [
        "Morning Run",
        "Lunch Run",
        "Afternoon Run",
        "Evening Run",
        "Night Run",
    ]
    if row["name"] in default_names:
        return 0
    else:
        return 1


def label_max(row, max_dist):
    if (
        row["distance"]
        == max_dist[max_dist.local_date == row["local_date"]].max_distance.values[0]
    ):
        return 1
    else:
        return 0


def race_heuristic(strava_caption, workout_type):
    strava_caption = str(strava_caption)
    if re.findall(r"\b\d+th|\d+st|\d+rd|\d+nd\b", strava_caption):
        return 1
    else:
        return workout_type


def map_time_of_day(hour):
    if hour in [6, 7, 8, 9, 10]:
        return "AM"

    elif hour in [11, 12, 13, 14, 15]:
        return "Mid"

    elif hour in [16, 17, 18, 19, 20]:
        return "PM"

    else:
        return "Night"


def preprocess_unseen(data):
    # extract only runs
    data = data.loc[data.type == "Run"]
    data = data.drop(columns=["type"])
    # add race heuristics
    data["workout_type"] = data.apply(
        lambda row: race_heuristic(row["name"], row["workout_type"]), axis=1
    )
    # remove private activities
    data = data.loc[data.private == False].drop("private", axis=1)
    # label names
    data.loc[:, "is_named"] = data.apply(lambda row: label_name(row), axis=1)
    # format date
    data = format_date(data)
    # get runs per day feature
    counts = data.groupby("local_date").size().reset_index(name="run_per_day")
    data = pd.merge(data, counts, on=["local_date"], how="inner")
    # label max run
    max_dist = (
        data.groupby("local_date")["distance"]
        .agg("max")
        .reset_index(name="max_distance")
    )
    data.loc[:, "max_run"] = data.apply(lambda row: label_max(row, max_dist), axis=1)
    # get run area feature
    data = get_poly_areas(data)
    # convert distance in meters to kilometers
    data.loc[:, "distance"] = data.distance / 1000
    # convert moving in seconds time to minutes
    data.loc[:, "moving_time"] = data.moving_time / 60
    # convert average speed in meters per second to minutes per kilometer
    data.loc[:, "average_speed_mpk"] = 16.666 / data.average_speed
    data = data.drop("average_speed", axis=1)
    # remove runs with no kudos as it wasn't available to public
    data = data[data.kudos_count != 0]
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
        "timezone",
        "achievement_count",
        "total_photo_count",
        "GMT_date",
        "GMT_time",
        "local_date",
        "local_time",
        "manual",
    ]
    for col in cols_to_drop:
        try:
            data = data.drop(col, axis=1)
        except:
            pass
    return data


if __name__ == "__main__":
    # load in initial data gathered from the Strava API
    data_path = config.STRAVA_DATA_PATH
    data_init = pd.read_csv(data_path, index_col=0)

    # filter out runs from activities and drop type column
    data_init = data_init.loc[data_init.type == "Run"]
    data_init = data_init.drop(columns=["type"])

    # add race heuristics
    data_init["workout_type"] = data_init.apply(
        lambda row: race_heuristic(row["name"], row["workout_type"]), axis=1
    )

    # remove private activities and drop older activities
    data_init = (
        data_init.loc[data_init.private == False].iloc[0:1125].drop("private", axis=1)
    )
    # map photo feature
    data_init["has_photo"] = data_init["total_photo_count"].map(
        lambda x: 0 if x == 0 else 1
    )

    # label names
    data_init.loc[:, "is_named"] = data_init.apply(lambda row: label_name(row), axis=1)

    data_init = format_date(data_init)
    data_init = format_timezone(data_init)

    # get runs per day feature
    counts = data_init.groupby("local_date").size().reset_index(name="run_per_day")
    data_init = pd.merge(data_init, counts, on=["local_date"], how="inner")

    # longest run of day
    max_dist = (
        data_init.groupby("local_date")["distance"]
        .agg("max")
        .reset_index(name="max_distance")
    )
    data_init.loc[:, "max_run"] = data_init.apply(
        lambda row: label_max(row, max_dist), axis=1
    )
    data_init = generate_time_features(data_init)
    data_init["hour_binned"] = data_init["hour"].apply(map_time_of_day)

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
