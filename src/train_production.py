import pandas as pd
import numpy as np
import xgboost as xgb
import pickle


def warn(*args, **kwargs):
    pass


import warnings

warnings.warn = warn

from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import RandomForestRegressor


def uk_awake(hour):
    if hour >= 6 and hour <= 19:
        return 1
    else:
        return 0


def train():

    # read in the data
    STRAVA_TRAIN_PATH = "input/data_train.csv"
    df = pd.read_csv(STRAVA_TRAIN_PATH)
    # create uk awake feature
    df.loc[:, "datetime"] = pd.to_datetime(df["GMT_date"] + " " + df["GMT_time"])
    df.loc[:, "hour"] = df["datetime"].dt.hour
    df.loc[:, "uk_awake"] = df.hour.apply(uk_awake)
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
    cat_cols = ["max_run", "workout_type", "is_named", "run_per_day", "uk_awake"]
    # all cols are features except for target
    features = num_cols + cat_cols
    # select only needed cols
    df = df[["kudos_count"] + features]

    # fill in NAs in cat columns with NONE
    for col in cat_cols:
        df.loc[:, col] = df[col].astype(str).fillna("NONE")

    # split data into test and training
    # randomize the rows of the data
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    # Sturge's rule to calc bins
    num_bins = int(np.floor(1 + np.log2(len(df))))
    # bin targets
    df.loc[:, "bins"] = pd.cut(df["kudos_count"], bins=num_bins, labels=False)
    # initiate kfold class, 5 splits gives valid size 1/4 = 25%
    kf = StratifiedKFold(n_splits=4, random_state=42)
    # get indicies for train and valid set
    train_idx, valid_idx = next(iter(kf.split(X=df, y=df.bins.values)))
    # drop bins col
    df = df.drop("bins", axis=1)
    # set train and valid dataset
    df_train = df.loc[train_idx, :]
    df_valid = df.loc[valid_idx, :]
    print(f"Training shape: {df_train.shape} \nValidation shape: {df_valid.shape}")

    # we label encode all the features
    for col in cat_cols:
        # initialise label encoder
        lbl = LabelEncoder()
        # fit label encoder on training data
        lbl.fit(df_train[col])
        encodings = dict(zip(lbl.classes_, lbl.transform(lbl.classes_)))
        # transform all of the data
        df_train.loc[:, col] = lbl.transform(df_train[col])
        df_valid.loc[:, col] = lbl.transform(df_valid[col])
        # save each encoder
        with open(f"models/production/label_encoders/lbl_enc_{col}.pickle", "wb") as f:
            pickle.dump(encodings, f)

    # get target encodings
    target_encodings = {}
    # loop over each cat column
    for col in cat_cols:
        # create dict of category:mean_target
        mapping_dict = dict(df_train.groupby(col)["kudos_count"].mean())
        target_encodings[col] = mapping_dict
        # column_enc is the new column we have with mean encodings
        df_train.loc[:, col + "_enc"] = df_train[col].map(mapping_dict)
        df_valid.loc[:, col + "_enc"] = df_valid[col].map(mapping_dict)
    # save target encodings
    with open("models/production/target_encodings/target_enc.pickle", "wb") as f:
        pickle.dump(target_encodings, f)

    # impute missing values in numeric data
    # initialize imputer with stratergy median
    imp = SimpleImputer(strategy="median")
    # fit on training data
    imp = imp.fit(df_train[num_cols])
    # transform training and validation data
    df_train[num_cols] = imp.transform(df_train[num_cols])
    df_valid[num_cols] = imp.transform(df_valid[num_cols])
    # save imputer
    with open("models/production/imputer/numeric_imputer.pickle", "wb") as f:
        pickle.dump(imp, f)

    # get training matrix and y vectors
    features = [f for f in df_train.columns if f not in ("kudos_count")]
    x_train = df_train[features].values
    y_train = df_train.kudos_count.values
    x_valid = df_valid[features].values
    y_valid = df_valid.kudos_count.values

    # load best hyperparams from model_tuning.py
    with open("models/production/xgb_params.pickle", "rb") as f:
        params = pickle.load(f)

    # initialize model
    model = xgb.XGBRegressor(**params)
    # train
    model.fit(
        x_train,
        y_train,
        early_stopping_rounds=10,
        eval_metric="rmse",
        eval_set=[(x_valid, y_valid)],
        verbose=False,
    )

    # predict on validation data
    valid_preds = model.predict(x_valid)
    # get rmse, and max_error
    rmse = mean_squared_error(y_valid, valid_preds, squared=False)
    rmse_train = mean_squared_error(y_train, model.predict(x_train), squared=False)
    print(f"Rmse valid = {rmse}")
    print(f"Rmse train = {rmse_train}")

    with open("models/production/xgb_model.pickle", "wb") as f:
        params = pickle.dump(model, f)


if __name__ == "__main__":
    train()
