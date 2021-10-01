import argparse
import copy
import pandas as pd
import numpy as np
import xgboost as xgb

from sklearn import metrics
from sklearn import preprocessing
from sklearn import impute
from sklearn import ensemble
from sklearn.pipeline import Pipeline
from scipy.sparse import hstack, vstack

import config
import model_dispatcher


def mean_target_encoding(data):

    # make copy of dataframe
    df = copy.deepcopy(data)

    # list of numerical columns
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

    # select only needed cols
    df = df[features + ["kudos_count", "kfold"]]

    # fill in NAs with NONE
    for col in cat_cols:
        df.loc[:, col] = df[col].astype(str).fillna("NONE")

    # we label encode all the features
    for col in cat_cols:
        # initialise label encoder
        lbl = preprocessing.LabelEncoder()
        # fit label encoder
        lbl.fit(df[col])
        # transform all of the data
        df.loc[:, col] = lbl.transform(df[col])

    # a list to store 5 validation dataframes
    encoded_dfs = []

    # loop over every fold
    for fold in range(3):
        # get training and validation data
        df_train = df[df.kfold != fold].reset_index(drop=True)
        df_valid = df[df.kfold == fold].reset_index(drop=True)
        # for all cat columns
        for column in cat_cols:
            # create dict of category:mean target
            mapping_dict = dict(df_train.groupby(column)["kudos_count"].mean())
            # column_enc is the new column we have with mean encodings
            df_valid.loc[:, column + "_enc"] = df_valid[column].map(mapping_dict)
        # append to our list of encoded validation dfs
        encoded_dfs.append(df_valid)
    # create full dataframe again and return
    encoded_df = pd.concat(encoded_dfs, axis=0)
    return encoded_df


def run(df, fold):

    # training data is where kfold is not equal to fold
    df_train = df[df.kfold != fold].reset_index(drop=True)
    # validation data is where kfold = fold
    df_valid = df[df.kfold == fold].reset_index(drop=True)

    features = [f for f in df.columns if f not in ("kfold", "kudos_count")]

    x_train = df_train[features].values
    x_valid = df_valid[features].values

    model = xgb.XGBRegressor(n_jobs=-1)
    model.fit(x_train, df_train.kudos_count.values)

    # predict on validation data
    valid_preds = model.predict(x_valid)

    # get rmse, and mape
    rmse = metrics.mean_squared_error(
        df_valid.kudos_count.values, valid_preds, squared=False
    )
    max_error = metrics.max_error(df_valid.kudos_count.values, valid_preds)
    print(f"Fold = {fold}, rmse = {rmse}, max error = {max_error}")
    return rmse


if __name__ == "__main__":
    # read training data with folds
    df = pd.read_csv(config.STRAVA_TRAIN_KFOLD_PATH)
    # create mean target encoded categories and munge data
    df = mean_target_encoding(df)

    scores = []
    print("\nTraining xgb model")
    for fold_ in range(3):
        rmse = run(df, fold_)
        scores.append(rmse)
    print(f"Average rmse = {sum(scores) / len(scores)}")
