import pandas as pd
import numpy as np
from sklearn.svm import SVR, SVC
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, PolynomialFeatures
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier, AdaBoostRegressor
from xgboost import XGBRegressor, XGBClassifier


# --------------------------定义模型类---------------------------

class Models:
    # --------------------------定义超参数------------------------
    def __init__(self):
        self.cv = 5
        self.n_jobs = 4

    # --------------------------支持向量机------------------------
    def svm_regressor(self):
        self.svr_param_grid = [{

        }]
        self.svm_reg = SVR()
        svm_grid_search = GridSearchCV(self.svm_reg, self.svr_param_grid, cv=self.cv, n_jobs=self.n_jobs,
                                       verbose=1)
        return svm_grid_search

    def svm_classifier(self):
        self.svc_param_grid = [{

        }]
        self.svc_clf = SVC()
        svc_grid_search = GridSearchCV(self.svc_clf, self.svc_param_grid, cv=self.cv, n_jobs=self.n_jobs,
                                       verbose=1)
        return svc_grid_search

    # --------------------------决策树------------------------
    def dt_regressor(self):
        self.dtr_param_grid = [{

        }]
        self.df_reg = DecisionTreeRegressor()
        dtr_grid_search = GridSearchCV(self.df_reg, self.dtr_param_grid, cv=self.cv, n_jobs=self.n_jobs,
                                       verbose=1)
        return dtr_grid_search

    def dt_classifier(self):
        self.dtc_param_grid = [{

        }]
        self.df_clf = DecisionTreeClassifier()
        dtc_grid_search = GridSearchCV(self.df_clf, self.dtc_param_grid, cv=self.cv, n_jobs=self.n_jobs,
                                       verbose=1)
        return dtc_grid_search

    # --------------------------KNN------------------------
    def knn_regressor(self):
        self.knnr_param_grid = [{

        }]
        self.knn_reg = KNeighborsRegressor()
        knnr_grid_search = GridSearchCV(self.knn_reg, self.knnr_param_grid, cv=self.cv, n_jobs=self.n_jobs,
                                        verbose=1)
        return knnr_grid_search

    def knn_classifier(self):
        self.knnc_param_grid = [{

        }]
        self.knn_clf = KNeighborsClassifier()
        knnc_grid_search = GridSearchCV(self.knn_clf, self.knnc_param_grid, cv=self.cv, n_jobs=self.n_jobs,
                                        verbose=1)
        return knnc_grid_search

    # -------------------------线性回归和逻辑回归---------------------
    def polynomial_linear_regressor(self):
        self.linear_param_grid = [{
            'polynomial__degree': [i for i in range(1, 10)]
        }]
        self.linear_reg = Pipeline([('polynomial', PolynomialFeatures()),
                                          ('linear_regression', LinearRegression())])
        linear_grid_search = GridSearchCV(self.linear_reg, self.linear_param_grid, cv=self.cv, n_jobs=self.n_jobs,
                                          verbose=1)
        return linear_grid_search

    def logistic_regression_classifier(self):
        self.logistic_param_grid = [{

        }]
        self.logistic_clf = LogisticRegression()
        logistic_grid_search = GridSearchCV(self.logistic_clf, self.logistic_param_grid, cv=self.cv,
                                            n_jobs=self.n_jobs,
                                            verbose=1)
        return logistic_grid_search

    # -------------------------随机森林---------------------
    def randomforest_regressor(self):
        self.rfr_param_grid = [{

        }]
        self.rf_reg = RandomForestRegressor()
        rfr_grid_search = GridSearchCV(self.rf_reg, self.rfr_param_grid, cv=self.cv, n_jobs=self.n_jobs,
                                       verbose=1)
        return rfr_grid_search

    def randomforest_classifier(self):
        self.rfc_param_grid = [{

        }]
        self.rf_clf = RandomForestClassifier()
        rfc_grid_search = GridSearchCV(self.rf_clf, self.rfc_param_grid, cv=self.cv, n_jobs=self.n_jobs,
                                       verbose=1)
        return rfc_grid_search

    # ------------------------Adaboost---------------------
    def adaboost_regressor(self):
        self.adar_param_grid = [{

        }]
        self.ada_reg = AdaBoostRegressor()
        adar_grid_search = GridSearchCV(self.ada_reg, self.adar_param_grid, cv=self.cv, n_jobs=self.n_jobs,
                                        verbose=1)
        return adar_grid_search

    def adaboost_classifier(self):
        self.adac_param_grid = [{

        }]
        self.ada_clf = AdaBoostClassifier()
        adac_grid_search = GridSearchCV(self.ada_clf, self.adac_param_grid, cv=self.cv, n_jobs=self.n_jobs,
                                        verbose=1)
        return adac_grid_search

    # ------------------------xgboost---------------------

    def xgb_regressor(self):
        self.xgbr_param_grid = [{

        }]
        self.xgb_reg= XGBRegressor()
        xgbr_grid_search = GridSearchCV(self.xgb_reg, self.xgbr_param_grid, cv=self.cv, n_jobs=self.n_jobs,
                                        verbose=1)
        return xgbr_grid_search

    def xgb_classifier(self):
        self.xgbc_param_grid = [{

        }]
        self.xgb_clf= XGBClassifier()
        xgbc_grid_search = GridSearchCV(self.xgb_clf, self.xgbc_param_grid, cv=self.cv, n_jobs=self.n_jobs,
                                        verbose=1)
        return xgbc_grid_search





