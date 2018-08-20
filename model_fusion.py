import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression,LogisticRegression
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split, cross_val_score, cross_val_predict
from mlxtend.regressor import StackingRegressor, StackingCVRegressor
from mlxtend.classifier import StackingClassifier, StackingCVClassifier
from models import Models

model = Models()


# ---------------------------------------定义模型融合类----------------------------------------
class RegressorBlender:
    def __init__(self, x_train, x_test, y_train, y_test=None):
        x_train.drop(['Unnamed: 0', 'Id'], axis=1, inplace=True)
        x_test.drop(['Unnamed: 0', 'Id'], axis=1, inplace=True)
        self.x_train = x_train
        self.x_test = x_test
        self.y_train = y_train['y'].values
        if self.y_train is not None:
            self.y_test = y_test['y'].values

    def reg_blend(self):
        mete_reg = LinearRegression()
        reg1 = model.svm_regressor()
        reg2 = model.randomforest_regressor()
        reg3 = model.xgb_regressor()
        self.blend = StackingRegressor(regressors=[reg1, reg2, reg3], meta_regressor=mete_reg)
        self.blend.fit(self.x_train, self.y_train)
        return self.blend

    def score(self):
        scores = cross_val_score(self.blend, X=self.x_train, y=self.y_train, cv=10,
                                 verbose=2)
        return scores

    def prediction(self):
        y_pred = self.blend.predict(self.x_test)
        return y_pred

import matplotlib
class ClassifierBlender:
    def __init__(self, x_train, x_test, y_train, y_test=None):
        x_train.drop(['Unnamed: 0', 'Id'], axis=1, inplace=True)
        x_test.drop(['Unnamed: 0', 'Id'], axis=1, inplace=True)
        self.x_train = x_train
        self.x_test = x_test
        self.y_train = y_train['y'].values
        if self.y_train is not None:
            self.y_test = y_test['y'].values

    def clf_blend(self):
        mete_clf = LinearRegression()
        clf1 = model.svm_regressor()
        clf2 = model.randomforest_regressor()
        clf3 = model.xgb_regressor()
        self.blend = StackingClassifier(classifiers=[clf1, clf2, clf3], meta_classifier=mete_clf)
        self.blend.fit(self.x_train, self.y_train)
        return self.blend

    def score(self):
        scores = cross_val_score(self.blend, X=self.x_train, y=self.y_train, cv=10,
                                 verbose=2)
        return scores

    def prediction(self):
        y_pred = self.blend.predict(self.x_test)
        return y_pred
