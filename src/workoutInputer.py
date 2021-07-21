import numpy as np
import pandas as pd
import re

import config

pd.options.mode.chained_assignment = None  # default='warn'


def race_heuristic(strava_caption):
    strava_caption = str(strava_caption)
    if re.findall(r"\b\d+th|\d+st|\d+rd|\d+nd\b", strava_caption):
        return 1
    else:
        return np.nan


if __name__ == "__main__":

    # load in preprocessed data
    data = pd.read_csv("input/data_preprocessed.csv", index_col=0)

    # train and test set for workout classifier
    data_missing_ix = data[data["workout_type"].isnull()].index
    data_full = data[~data.index.isin(data_missing_ix)]
    data_missing = data[data.index.isin(data_missing_ix)]

    cols = [
        "workout_type",
        "name",
        "distance",
        "kudos_count",
        "photo_count",
        "average_speed",
    ]
    workout = workout[cols]

    nan_idx = workout[workout["workout_type"].isnull()].index
    missing_data = workout[workout.index.isin(nan_idx)]

    # print("\nWorkout columns: ", workout.columns.tolist())
    # print("Workout shape: ", workout.shape, "\n")
    # print("Missing columns:\n", workout.isnull().sum(axis=0), "\n")

    missing_data.loc[:, "workout_type"] = missing_data["name"].apply(race_heuristic)
