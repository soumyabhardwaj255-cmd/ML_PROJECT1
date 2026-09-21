"""Compatibility factories; all active settings live in config/modeling."""
from .modeling import make_model

def make_logistic_regression():
    return make_model("logistic_regression", [0, 1])

def make_random_forest():
    return make_model("random_forest", [0, 1])

def make_xgboost(y_train):
    return make_model("xgboost", y_train)

def get_models(y_train):
    return {name: make_model(name, y_train) for name in
            ("logistic_regression", "random_forest", "xgboost")}
