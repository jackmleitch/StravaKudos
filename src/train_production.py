import pandas as pd
import numpy as np
import pickle


def warn(*args, **kwargs):
    pass


import warnings

warnings.warn = warn

from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder

# import config


def train():

    # read in the data
    STRAVA_TRAIN_PATH = "input/data_train.csv"
    df = pd.read_csv(STRAVA_TRAIN_PATH)

    # split data into test and training
    # randomize the rows of the data
    df = df.sample(frac=1).reset_index(drop=True)
    # Sturge's rule to calc bins
    num_bins = int(np.floor(1 + np.log2(len(df))))
    # bin targets
    df.loc[:, "bins"] = pd.cut(df["kudos_count"], bins=num_bins, labels=False)
    # initiate kfold class, 5 splits gives valid size 1/5 = 20%
    kf = StratifiedKFold(n_splits=5)
    # get indicies for train and valid set
    train_idx, valid_idx = next(iter(kf.split(X=df, y=df.bins.values)))
    # drop bins col
    df = df.drop("bins", axis=1)
    # set train and valid dataset
    df_train = df.loc[train_idx, :]
    df_valid = df.loc[valid_idx, :]
    print(f"Training shape: {df_train.shape} \nValidation shape: {df_valid.shape}")

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
    cat_cols = ["max_run", "workout_type", "is_named", "run_per_day"]
    # all cols are features except for target
    features = num_cols + cat_cols

    # select only needed cols
    df = df[["kudos_count"] + features]

    # fill in NAs in cat columns with NONE
    for col in cat_cols:
        df.loc[:, col] = df[col].astype(str).fillna("NONE")

    # we label encode all the features
    for col in cat_cols:
        # initialise label encoder
        lbl = LabelEncoder()
        # fit label encoder
        lbl.fit(df[col])
        # transform all of the data
        df.loc[:, col] = lbl.transform(df[col])
        # save each encoder
        with open(f"models/production/label_encoders/lbl_enc_{col}.pickle", "wb") as f:
            pickle.dump(lbl, f)

    # get target encodings
    target_encodings = {}
    # loop over each cat column
    for col in cat_cols:
        # create dict of category:mean_target
        mapping_dict = dict(df_train.groupby(col)["kudos_count"].mean())
        target_encodings[col] = mapping_dict
    # save target encodings
    with open("models/production/target_encodings/target_enc.pickle", "wb") as f:
        pickle.dump(target_encodings, f)
    # column_enc is the new column we have with mean encodings
    # df_valid.loc[:, col + "_enc"] = df_valid[col].map(mapping_dict)


if __name__ == "__main__":
    train()
