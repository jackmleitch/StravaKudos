import pandas as pd
import numpy as np
import xgboost as xgb
import pickle

from optuna import Trial
from optuna import create_study
from optuna.samplers import TPESampler
from optuna.integration import XGBoostPruningCallback
from optuna.visualization import plot_optimization_history


def warn(*args, **kwargs):
    pass


import warnings

warnings.warn = warn

from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_squared_error

from predict import process


def objective(trial, n_jobs=-1, random_state=42):

    # read in the data
    STRAVA_TRAIN_PATH = "input/data_train.csv"
    data = pd.read_csv(STRAVA_TRAIN_PATH)
    # prepare data
    df = process(data)
    df.loc[:, "kudos_count"] = data.kudos_count

    # stratisfied kfold
    # create a new col kfold and fill with -1
    df["kfold"] = -1
    # randomize the rows of the data
    df = df.sample(frac=1).reset_index(drop=True)
    # Sturge's rule to calc bins
    num_bins = int(np.floor(1 + np.log2(len(df))))
    # bin targets
    df.loc[:, "bins"] = pd.cut(df["kudos_count"], bins=num_bins, labels=False)
    # initiate kfold class
    kf = StratifiedKFold(n_splits=5)
    # fill the new kfold col
    for f, (t_, v_) in enumerate(kf.split(X=df, y=df.bins.values)):
        df.loc[v_, "kfold"] = f
    # drop bins col
    df = df.drop("bins", axis=1)

    params = {
        "tree_method": "gpu_hist",
        "verbosity": 0,  # 0 (silent) - 3 (debug)
        "objective": "reg:squarederror",
        "n_estimators": trial.suggest_int("n_estimators", 100, 1000),
        "max_depth": trial.suggest_int("max_depth", 4, 12),
        "learning_rate": trial.suggest_loguniform("learning_rate", 0.005, 0.05),
        "colsample_bytree": trial.suggest_loguniform("colsample_bytree", 0.2, 0.6),
        "subsample": trial.suggest_loguniform("subsample", 0.4, 0.8),
        "alpha": trial.suggest_loguniform("alpha", 0.01, 10.0),
        "lambda": trial.suggest_loguniform("lambda", 1e-8, 10.0),
        "gamma": trial.suggest_loguniform("lambda", 1e-8, 10.0),
        "min_child_weight": trial.suggest_loguniform("min_child_weight", 10, 1000),
        "seed": random_state,
        "n_jobs": n_jobs,
    }

    # add pruning callback
    pruning_callback = XGBoostPruningCallback(trial, "validation_0-rmse")

    scores = []
    features = [f for f in df.columns if f not in ("kudos_count", "kfold")]
    for fold in range(5):

        # training data is where kfold is not equal to fold
        x_train = df[df.kfold != fold].reset_index(drop=True)
        y_train = x_train.kudos_count.values
        x_train = x_train[features]
        # validation data is where kfold = fold
        x_valid = df[df.kfold == fold].reset_index(drop=True)
        y_valid = x_valid.kudos_count.values
        x_valid = x_valid[features]

        model = xgb.XGBRegressor(**params)
        model.fit(
            x_train,
            y_train,
            eval_set=[(x_valid, y_valid)],
            early_stopping_rounds=50,
            verbose=False,
            callbacks=[pruning_callback],
        )

        preds = model.predict(x_valid)
        scores.append(mean_squared_error(y_valid, preds, squared=False))

    score = sum(scores) / len(scores)
    return score


if __name__ == "__main__":

    sampler = TPESampler()
    study = create_study(direction="minimize", sampler=sampler)
    study.optimize(objective, n_trials=100)

    # display params
    best = study.best_params
    for key, value in best.items():
        print(f"{key:>20s} : {value}")
    print(f"{'best objective value':>20s} : {study.best_value}")

    with open("models/production/xgb_params.pickle", "wb") as f:
        pickle.dump(best, f)
