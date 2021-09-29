import xgboost as xgb

from sklearn import ensemble
from sklearn import tree

tree_models = {
    "decision_tree": tree.DecisionTreeRegressor(),
    "rf": ensemble.RandomForestRegressor(),
    "xgb": xgb.XGBRegressor(n_jobs=-1),
}
