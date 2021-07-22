import pandas as pd
import numpy as np
import datetime
import re

import config


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

    df["temp"] = df[column_name].apply(lambda x: format_date_helper(x))
    df[["date", "time"]] = pd.DataFrame(df.temp.values.tolist(), index=df.index)
    df = df.drop(["temp", "start_date", "start_date_local"], axis=1)
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


if __name__ == "__main__":
    # load in initial data gathered from the Strava API
    data_path = config.STRAVA_DATA_PATH
    data_init = pd.read_csv(data_path, index_col=0)

    # drop unwanted/useless columns from the dataset
    cols_to_drop = [
        "resource_state",
        "id",
        "external_id",
        "upload_id",
        "utc_offset",
        "start_latitude",
        "start_longitude",
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
        "map.summary_polyline",
        "map.resource_state",
        "device_watts",
        "average_watts",
        "kilojoules",
        "message",
        "location_city",
        "location_state",
        "location_country",
        "comment_count",
        "elapsed_time",
    ]
    data_init = data_init.drop(cols_to_drop, axis=1)

    # filter out runs from activities and drop type column
    data_init = data_init.loc[data_init.type == "Run"]
    data_init = data_init.drop(columns=["type"])

    # remove private activities and drop older activities
    data_init = (
        data_init.loc[data_init.private == False].iloc[0:1001].drop("private", axis=1)
    )
    data_init = format_date(data_init)
    data_init = format_timezone(data_init)

    # convert distance in meters to kilometers
    data_init.loc[:, "distance"] = data_init.distance / 1000
    # convert moving in seconds time to minutes
    data_init.loc[:, "moving_time"] = data_init.moving_time / 60
    # convert average speed in meters per second to minutes per kilometer
    data_init.loc[:, "average_speed_mpk"] = 16.666 / data_init.average_speed
    data_init = data_init.drop("average_speed", axis=1)

    # split data into training and testing set
    test_frac = 0.25

    data_train = data_init[int(data_init.shape[0] * test_frac) :]
    data_test = data_init[: int(data_init.shape[0] * test_frac)]
    print("train: ", data_train.shape)
    print("test: ", data_test.shape)

    data_train.to_csv("input/data_train.csv")
    data_test.to_csv("input/data_test.csv")
    print("Data preprocessed and saved to file 'data_train.csv' and 'data_test.csv'")
