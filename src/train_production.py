import pandas as pd
import xgboost as xgb
import pickle


def warn(*args, **kwargs):
    pass


import warnings

warnings.warn = warn

from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_squared_error

from predict import process


def feature_engineer():
    # read in the data
    STRAVA_TRAIN_PATH = "input/data_train.csv"
    df = pd.read_csv(STRAVA_TRAIN_PATH)

    # print training shape
    print(f"Training shape: {df.shape}")

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
    df = df[["kudos_count"] + features]

    # fill in NAs in cat columns with NONE
    for col in cat_cols:
        df.loc[:, col] = df[col].astype(str).fillna("NONE")

    # we label encode all the features
    for col in cat_cols:
        # initialise label encoder
        lbl = LabelEncoder()
        # fit label encoder on training data
        lbl.fit(df[col])
        encodings = dict(zip(lbl.classes_, lbl.transform(lbl.classes_)))
        # transform all of the data
        df.loc[:, col] = lbl.transform(df[col])
        # save each encoder
        with open(f"models/production/label_encoders/lbl_enc_{col}.pickle", "wb") as f:
            pickle.dump(encodings, f)

    # get target encodings
    target_encodings = {}
    # loop over each cat column
    target_encoding_cols = ["workout_type", "max_run", "run_per_day"]
    for col in target_encoding_cols:
        # create dict of category:mean_target
        mapping_dict = dict(df.groupby(col)["kudos_count"].mean())
        target_encodings[col] = mapping_dict
        # column_enc is the new column we have with mean encodings
        df.loc[:, col + "_enc"] = df[col].map(mapping_dict)
    # save target encodings
    with open("models/production/target_encodings/target_enc.pickle", "wb") as f:
        pickle.dump(target_encodings, f)

    # impute missing values in numeric data
    # initialize imputer with stratergy median
    imp = SimpleImputer(strategy="median")
    # fit on training data
    imp = imp.fit(df[num_cols])
    # transform training data
    df[num_cols] = imp.transform(df[num_cols])
    # save imputer
    with open("models/production/imputer/numeric_imputer.pickle", "wb") as f:
        pickle.dump(imp, f)

    # get training matrix and y vectors
    features = [f for f in df.columns if f not in ("kudos_count")]
    x_train = df[features].values
    y_train = df.kudos_count.values
    return x_train, y_train


def train(x_train, y_train, x_valid, y_valid):

    # load best hyperparams from model_tuning.py
    with open("models/production/xgb_params.pickle", "rb") as f:
        params = pickle.load(f)

    # initialize model
    model = xgb.XGBRegressor(**params)
    # train
    model.fit(
        x_train,
        y_train,
        eval_set=[(x_valid, y_valid)],
        early_stopping_rounds=50,
        verbose=False,
    )

    with open("models/production/xgb_model.pickle", "wb") as f:
        params = pickle.dump(model, f)

    print("Training done and model saved to models/production/xgb_model.pickle")
    print(
        f"Training RMSE: {mean_squared_error(model.predict(x_train), y_train, squared=False)}"
    )
    print(
        f"Validation RMSE: {mean_squared_error(model.predict(x_valid), y_valid, squared=False)}"
    )


if __name__ == "__main__":
    # engineer features
    x_train, y_train = feature_engineer()
    # valid set
    df_valid = pd.read_csv("input/data_test.csv")
    x_valid, y_valid = process(df_valid).values, df_valid.kudos_count.values
    # train xgb model
    train(x_train, y_train, x_valid, y_valid)
