import pandas as pd
import numpy as np

import config

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
    ]
    data_init = data_init.drop(cols_to_drop, axis=1)

    # filter out runs from activities and drop type column
    data_init = data_init.loc[data_init.type == "Run"]
    data_init = data_init.drop(columns=["type"])

    # remove private activities and drop older activities
    data_init = (
        data_init.loc[data_init.private == False].iloc[0:1001].drop("private", axis=1)
    )

    # split data into training and testing set
    test_frac = 0.25
    data_train = data_init[int(data_init.shape[0] * test_frac) :]
    data_test = data_init[: int(data_init.shape[0] * test_frac)]
    print("train: ", data_train.shape)
    print("test: ", data_test.shape)

    data_train.to_csv("input/data_train.csv")
    data_test.to_csv("input/data_test.csv")
    print("Data preprocessed and saved to file 'data_train.csv' and 'data_test.csv'")
