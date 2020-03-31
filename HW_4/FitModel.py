import catboost as catb
import lightgbm as lgbm
import pandas as pd
import xgboost as xgb
import ReportModel
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier


def ModelFit(model, x_train, y_train, x_test, y_test):
    model.fit(x_train, y_train)
    y_train_predict = model.predict(x_train)
    y_test_predict = model.predict(x_test)
    ReportModel.get_classification_report(y_train, y_train_predict, 'TRAIN')
    ReportModel.get_classification_report(y_test, y_test_predict, 'TEST')
    return model


def LogisticRegressionFit(x_train, y_train, x_test, y_test):
    model_lr = LogisticRegression()
    return ModelFit(model_lr, x_train, y_train, x_test, y_test)


def KNeighborsClassifierFit(x_train, y_train, x_test, y_test):
    model_knn = KNeighborsClassifier()
    return ModelFit(model_knn, x_train, y_train, x_test, y_test)


def XGBClassifierFit(x_train, y_train, x_test, y_test):
    model_xgb = xgb.XGBClassifier(random_state=42)
    return ModelFit(model_xgb, x_train, y_train, x_test, y_test)


def LGBMClassifierFit(x_train, y_train, x_test, y_test):
    model_lgbm = lgbm.LGBMClassifier(random_state=42)
    return ModelFit(model_lgbm, x_train, y_train, x_test, y_test)


def CatBoostFit(x_train, y_train, x_test, y_test):
    model_catb = catb.CatBoostClassifier(n_estimators=200, max_depth=3,
                                         silent=True, random_state=42)
    return ModelFit(model_catb, x_train, y_train, x_test, y_test)
