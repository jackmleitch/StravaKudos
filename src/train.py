import argparse
import os

import joblib
import pandas as pd
from sklearn import metrics

import config

# import model_dispatcher


def run(fold, model):
    # read training data with folds
    df = pd.read_csv(config.STRAVA_TRAIN_KFOLD_PATH)

    # list all numeric features
    num_cols = [
        "distance",
        "moving_time",
        "total_elevation_gain",
        "max_speed",
        "average_heartrate",
        "max_heartrate",
        "suffer_score",
        "run_area",
        "average_speed_mpk",
    ]

    cat_cols = [
        "workout_type",
        "timezone",
        "manual",
        "dayofweek",
        "weekend",
        "hour",
        "is_uk_awake",
        "latlng_cluster",
        "city",
        "pr_count",
        "total_photo_count",
    ]

    # training data is where kfold is not equal to fold
    df_train = df[df.kfold != fold].reset_index(drop=True)
    # validation data is where kfold = fold
    df_valid = df[df.kfold == fold].reset_index(drop=True)

    # drop target col from df and convert it to numpy array
    X_train = df_train.drop("kudos_count", axis=1)


if __name__ == "__main__":
    
