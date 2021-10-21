import pandas as pd
import numpy as np
import xgboost as xgb
import pickle

from sklearn.metrics import mean_squared_error


def warn(*args, **kwargs):
    pass


import warnings

warnings.warn = warn

import config


def process(df):
    # get features we need
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
    # list of categorical columns
    cat_cols = ["max_run", "workout_type", "run_per_day", "uk_awake"]
    # all cols are features except for target
    features = num_cols + cat_cols
    # select only needed cols
    df = df[features]
    # fill in NAs in cat columns with NONE
    for col in cat_cols:
        df.loc[:, col] = df[col].astype(str).fillna("NONE")

    # we encode all the features
    for col in cat_cols:
        # load in label encoder
        with open(f"models/production/label_encoders/lbl_enc_{col}.pickle", "rb") as f:
            lbl = pickle.load(f)
        # transform the column data
        df.loc[:, col] = df[col].apply(lambda x: lbl.get(x, np.nan))

    # load in target encodings
    target_encoding_cols = ["workout_type", "max_run", "run_per_day"]
    with open("models/production/target_encodings/target_enc.pickle", "rb") as f:
        target_encodings = pickle.load(f)
    # loop over each cat column
    for col in target_encoding_cols:
        # get mapping
        mapping_dict = target_encodings[col]
        # column_enc is the new column we have with mean encodings
        df.loc[:, col + "_enc"] = df[col].map(mapping_dict)

    # impute missing values in numeric data
    # load in imputer
    with open("models/production/imputer/numeric_imputer.pickle", "rb") as f:
        imp = pickle.load(f)
    # transform data
    df[num_cols] = imp.transform(df[num_cols])
    return df


def predict(df, model=False):

    # get features we need
    df = process(df)
    df = df.values

    # load model
    if not model:
        with open("models/production/xgb_model.pickle", "rb") as f:
            model = pickle.load(f)

    # predict on data and return
    preds = model.predict(df)
    preds = [round(pred) for pred in preds]
    return preds


def score_preds(preds, y_true):
    rmse = mean_squared_error(preds, y_true, squared=False)
    print(f"RMSE on data = {rmse}")


if __name__ == "__main__":

    # read in data
    test_data = pd.read_csv(config.STRAVA_TEST_PATH)

    # make predictions
    preds = predict(test_data)

    # score preds
    y_true = test_data.kudos_count.values
    score_preds(preds, y_true)
