import xgboost as xgb

from sklearn import ensemble
from sklearn import tree
from sklearn import linear_model
from sklearn import svm

models = {
    "rf": ensemble.RandomForestRegressor(),
    "xgb": xgb.XGBRegressor(n_jobs=-1),
    "linear": linear_model.LinearRegression(),
    "ridge": linear_model.Ridge(),
}

# "decision_tree": tree.DecisionTreeRegressor(),
# "lasso": linear_model.Lasso(),
# "svm": svm.SVR(),
# "svm_linear": svm.SVR(kernel="linear"),
# "svm_poly": svm.SVR(kernel="poly"),
# "svm_sigmoid": svm.SVR(kernel="sigmoid"),
