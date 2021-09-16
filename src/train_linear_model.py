import argparse
import collections
import os
from src import workoutImputer

import joblib
import pandas as pd
import numpy as np
from sklearn import metrics
from sklearn import preprocessing
from sklearn import impute
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn import neighbors

from sklearn.linear_model import LinearRegression

import config


def run(fold, transform_target=False, remove_outliers=False):

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
        "is_uk_awake",
        "latlng_cluster",
        "city",
    ]

    ordinal_cols = ["hour", "total_photo_count", "pr_count"]

    # all cols are features except for target and kfold
    features = num_cols + cat_cols + ordinal_cols

    # fill cat column NaN values with NONE
    for col in cat_cols + ordinal_cols:
        df.loc[:, col] = df[col].astype(str).fillna("NONE")

    # training data is where kfold is not equal to fold
    df_train = df[df.kfold != fold].reset_index(drop=True)
    y_train = df_train.kudos_count.values
    # validation data is where kfold = fold
    df_valid = df[df.kfold == fold].reset_index(drop=True)
    y_valid = df_valid.kudos_count.values

    # pipelines for model transformation
    num_pipeline = Pipeline(
        [
            ("imputer", impute.SimpleImputer(strategy="median")),
            ("std_scaler", preprocessing.StandardScaler()),
        ]
    )

    full_pipeline = ColumnTransformer(
        [
            ("num", num_pipeline, num_cols),
            (
                "cat",
                preprocessing.OneHotEncoder(handle_unknown="ignore"),
                cat_cols + ordinal_cols,
            ),
        ]
    )

    # transforms columns and drops columns not specified
    x_train = full_pipeline.fit_transform(df_train)
    x_valid = full_pipeline.transform(df_valid)

    # transform target variable
    if transform_target:
        boxcox = preprocessing.PowerTransformer()
        y_train = boxcox.fit_transform(y_train.reshape(-1, 1))
        y_train = np.squeeze(y_train)

    if remove_outliers:
        # initialize and train outlier model
        outlier_model = neighbors.LocalOutlierFactor(
            n_neighbors=2, metric="manhattan", contamination=0.02
        )
        outlier_train = np.append(x_train.toarray(), y_train.reshape(-1, 1), axis=1)
        # remove outliers from training data
        outliers = outlier_model.fit_predict(outlier_train)
        x_train = x_train[outliers > 0, :]
        y_train = y_train[outliers > 0]

    # check shapes are the same
    # print("Training data shape: ", x_train.shape)
    # print("Training target shape: ", y_train.shape, "\n")
    assert (
        x_train.shape[0] == y_train.shape[0]
    ), "training data and label dimension are not equal"

    # print("Validation data shape: ", x_valid.shape)
    # print("Validation target shape: ", y_valid.shape, "\n")
    assert (
        x_valid.shape[0] == y_valid.shape[0]
    ), "validation data and label dimension are not equal"

    # initialize the model
    model = LinearRegression()
    # fit model on training data
    model.fit(x_train, y_train)

    # predict on validation data
    valid_preds = model.predict(x_valid)
    if transform_target:
        valid_preds = boxcox.inverse_transform(valid_preds.reshape(-1, 1))
        valid_preds = np.squeeze(valid_preds)

    # get rmse, and mape
    rmse = metrics.mean_squared_error(y_valid, valid_preds, squared=False)
    max_error = metrics.max_error(y_valid, valid_preds)
    print(f"\nFold = {fold}, rmse = {rmse}, max error = {max_error}")


if __name__ == "__main__":
    for fold_ in range(3):
        run(fold_, remove_outliers=True)
