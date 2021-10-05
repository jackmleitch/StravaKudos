import xgboost as xgb
import pickle

from sklearn import ensemble
from sklearn import tree
from sklearn import linear_model
from sklearn import svm


with open("models/production/xgb_params.pickle", "rb") as f:
    params = pickle.load(f)

models = {
    "rf": ensemble.RandomForestRegressor(),
    "xgb": xgb.XGBRegressor(**params),
    "linear": linear_model.LinearRegression(),
    "ridge": linear_model.Ridge(),
}

# "decision_tree": tree.DecisionTreeRegressor(),
# "lasso": linear_model.Lasso(),
# "svm": svm.SVR(),
# "svm_linear": svm.SVR(kernel="linear"),
# "svm_poly": svm.SVR(kernel="poly"),
# "svm_sigmoid": svm.SVR(kernel="sigmoid"),
