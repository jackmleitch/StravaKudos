from os import XATTR_REPLACE
import pandas as pd
import numpy as np
import pickle
import re

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted

from pycaret.classification import predict_model, load_model

import config


def race_heuristic(strava_caption, workout_type):
    """
    Filters through titles and if it contains a result from a race then labels it as race.
    :param strava_caption: string, caption from strava workout
    :param workout_type: int, type of workout
    :return: new workout_type
    """
    strava_caption = str(strava_caption)
    if re.findall(r"\b\d+th|\d+st|\d+rd|\d+nd\b", strava_caption):
        return 1
    else:
        return workout_type


class WorkoutImputer(BaseEstimator, TransformerMixin):
    """
    Class used for imputing missing values in a pd.DataFrame using either mean or median of a group.
    
    Parameters
    ----------    
    target : str
        The name of the column to impute

    Returns
    -------
    X : array-like
        The array with imputed values in the target column
    """

    def __init__(self, target):

        assert type(target) == str, "target should be a string"
        self.target = target

    def fit(self, X, y=None):

        model = load_model("models/workoutImputer", verbose=False)
        self.model = model

        return self

    def transform(self, X, y=None):

        # make sure that the imputer was fitted
        check_is_fitted(self, "model")
        data = X.copy()

        # apply race heuristics
        data[self.target] = data.apply(
            lambda row: race_heuristic(row["name"], row[self.target]), axis=1
        )

        for index, row in data.iterrows():
            if row[self.target] not in [0, 1, 2, 3]:
                val = row.to_frame().T
                # data = data.drop("workout_type", axis=1)
                pred = predict_model(self.model, data=val)
                pred = pred.reset_index(drop=True)
                pred.loc[0, "workout_type"] = pred.loc[0, "Label"]
                data.loc[index, :] = pred.values[0][:-2]

        return data.values


if __name__ == "__main__":
    # load in preprocessed data
    data_path = config.STRAVA_TRAIN_PATH
    data = pd.read_csv(data_path, index_col=0)
    data = data.reset_index(drop=True)

    # workout inputer
    imp = WorkoutImputer(target="workout_type")
    imputed = imp.fit_transform(data)
    data_imp = pd.DataFrame(imputed, columns=data.columns)

