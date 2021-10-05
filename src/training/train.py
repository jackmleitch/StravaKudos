import os
import argparse
import pandas as pd
import numpy as np
import xgboost as xgb
import pickle

from sklearn import metrics
from sklearn import preprocessing
from sklearn import impute
from sklearn.pipeline import Pipeline
from scipy.sparse import hstack, vstack

import config
import model_dispatcher


def run(fold, model, scale_features=False):

    # read training data with folds
    df = pd.read_csv(config.STRAVA_TRAIN_KFOLD_PATH)

    # list all features
    num_cols = [
        "distance",
        "average_speed_mpk",
        "suffer_score",
        "max_speed",
        "moving_time",
        "max_heartrate",
        "total_elevation_gain",
        "run_area",
    ]

    cat_cols = ["max_run", "workout_type", "is_named", "run_per_day"]

    # all cols are features except for target and kfold
    features = num_cols + cat_cols

    # fill cat column NaN values with NONE
    for col in cat_cols:
        df.loc[:, col] = df[col].astype(str).fillna("NONE")

    # training data is where kfold is not equal to fold
    df_train = df[df.kfold != fold].reset_index(drop=True)
    y_train = df_train.kudos_count.values

    # validation data is where kfold = fold
    df_valid = df[df.kfold == fold].reset_index(drop=True)
    y_valid = df_valid.kudos_count.values

    # pipelines for model transformation
    if scale_features:
        num_pipeline = Pipeline(
            [
                ("imputer", impute.SimpleImputer(strategy="median")),
                ("std_scaler", preprocessing.StandardScaler()),
            ]
        )
    else:
        num_pipeline = Pipeline([("imputer", impute.SimpleImputer(strategy="median"))])

    cat_pipeline = Pipeline(
        [("cat", preprocessing.OneHotEncoder(handle_unknown="ignore"))]
    )

    # transforms columns and drops columns not specified
    x_train_num = num_pipeline.fit_transform(df_train[num_cols])
    x_train_cat = cat_pipeline.fit_transform(df_train[cat_cols])
    x_valid_num = num_pipeline.transform(df_valid[num_cols])
    x_valid_cat = cat_pipeline.transform(df_valid[cat_cols])

    # check shapes are the same
    assert (
        x_train_num.shape[0] == y_train.shape[0]
    ), "training data (numeric) and label dimension are not equal"

    assert (
        x_train_cat.shape[0] == y_train.shape[0]
    ), "training data (categorical) and label dimension are not equal"

    assert (
        x_valid_num.shape[0] == y_valid.shape[0]
    ), "validation data (numeric) and label dimension are not equal"

    assert (
        x_valid_cat.shape[0] == y_valid.shape[0]
    ), "validation data (categorical) and label dimension are not equal"

    # join numeric data and categorical data
    x_train = hstack((x_train_num, x_train_cat), format="csr")
    x_valid = hstack((x_valid_num, x_valid_cat), format="csr")

    # initialize model
    model = model_dispatcher.models[model]

    # fit model on training data
    model.fit(x_train, y_train)

    # predict on validation data
    valid_preds = model.predict(x_valid)

    # get rmse, and mape
    rmse = metrics.mean_squared_error(y_valid, valid_preds, squared=False)
    max_error = metrics.max_error(y_valid, valid_preds)
    print(f"Fold = {fold}, rmse = {rmse}, max error = {max_error}")

    data = [x_train, y_train, x_valid, y_valid]

    return rmse, model, data


if __name__ == "__main__":
    # initialize ArgumentParser class of argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str)
    parser.add_argument("--scale", type=bool)
    args = parser.parse_args()

    scores = []
    print(f"\nTraining {args.model} model")
    for fold_ in range(3):
        rmse, _, _ = run(fold_, model=args.model, scale_features=args.scale)
        scores.append(rmse)
    print(f"Average rmse for {args.model} = {sum(scores) / len(scores)}")

