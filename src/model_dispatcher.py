import xgboost as xgb

from sklearn import ensemble
from sklearn import tree
from sklearn import linear_model
from sklearn import svm

models = {
    "decision_tree": tree.DecisionTreeRegressor(),
    "rf": ensemble.RandomForestRegressor(),
    "xgb": xgb.XGBRegressor(n_jobs=-1),
    "linear": linear_model.LinearRegression(),
    "lasso": linear_model.Lasso(),
    "ridge": linear_model.Ridge(),
    "svm": svm.SVR(),
    "svm_linear": svm.SVR(kernel="linear"),
    "svm_poly": svm.SVR(kernel="poly"),
    "svm_sigmoid": svm.SVR(kernel="sigmoid"),
}
