import catboost as cat
import lightgbm as lgbm
import pandas as pd
import xgboost as xgb
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier


def Predict(model, df):
    return model.predict(df)
