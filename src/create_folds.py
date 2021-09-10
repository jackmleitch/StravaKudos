# stratified-kfold for regression
import numpy as np
import pandas as pd

from sklearn import datasets
from sklearn import model_selection

import config


def create_folds(data):
    # create a new col kfold and fill with -1
    data["kfold"] = -1

    # randomize the rows of the data
    data = data.sample(frac=1).reset_index(drop=True)

    # Sturge's rule to calc bins
    num_bins = int(np.floor(1 + np.log2(len(data))))

    # bin targets
    data.loc[:, "bins"] = pd.cut(data["kudos_count"], bins=num_bins, labels=False)

    # initiate kfold class
    kf = model_selection.StratifiedKFold(n_splits=3)

    # fill the new kfold col
    for f, (t_, v_) in enumerate(kf.split(X=data, y=data.bins.values)):
        data.loc[v_, "kfold"] = f

    # drop bins col
    data = data.drop("bins", axis=1)

    # return df with folds
    return data


if __name__ == "__main__":
    # data path
    data_train_path = config.STRAVA_TRAIN_PATH
    data_train = pd.read_csv(data_train_path, index_col=0)
    # create folds
    df = create_folds(data_train)
    # save new df with folds
    df.to_csv("input/data_train_kfold.csv")
    print("Folds created")
