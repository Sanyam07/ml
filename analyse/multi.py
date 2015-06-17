from abc import ABCMeta

import numpy as np
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.svm import SVR
import sklearn.ensemble as skensemble
from sklearn.decomposition import PCA

from analyse.preprocess import ImputeRegression, PreProcessor


class MultitargetWrapper:
    __metaclass__ = ABCMeta
    POSSIBLE_TRANSFORMERS = ['kbest', 'pca']

    def __init__(self, pre=None, best=0, transformer=None, class_imputer=None):

        if pre not in [None] + PreProcessor.POSSIBLE_KINDS:
            raise ValueError("")
        if transformer not in [None] + self.POSSIBLE_TRANSFORMERS:
            raise ValueError
        if class_imputer not in [None] + ImputeRegression.POSSIBLE_STRATEGIES:
            raise ValueError

        self.preprocessing = pre
        self.class_imputer = class_imputer
        self.best = best
        self.regressors = None
        self.transformers = None
        self.imputers = None
        self.kbset = transformer = "kbest"

        # TODO bagging params

        self.name = "MultitargetWrapper"


    def __call__(self, X_train, Y_train, X_test):
        self.fit(X_train, Y_train)
        return self.predict(X_test)

    def __str__(self):
        return self.name

    def get_params(self, deep=False):
        params = {

        }
        return params

    def set_params(self, **kwargs):
        if "param" in kwargs:
            pass

    def fit(self, X, Y):
        self.regressors = []
        if self.best:
            self.transformers = []
        if self.class_imputer == "drop":
            self.imputers = ImputeRegression(strategy=self.class_imputer, verbose=0)
        elif self.class_imputer:
            self.imputers = []

        if len(Y.shape) == 1:
            Y = np.array([Y]).T
        if self.preprocessing:
            X, Y = self._preprocess(X, Y)

        if self.best > 0 and not self.kbest:
            self.transformers = self._transform_to_best(X, None)
            Xb = self.transformers.transform(X)
        for i, y in enumerate(Y.T):
            self.regressors.append(self._init_regressor())
            if self.best > 0 and not self.kbset:
                self.transformers = self._transform_to_best(X, y)
                Xb = self.transformers.transform(X)
            if self.class_imputer == "drop":
                ex_mask, _ = self.imputers(y)
                yi = y[ex_mask]
                if self.bset > 0:
                    Xb = Xb[ex_mask, :]
                else:
                    Xb = X[ex_mask, :]

            elif self.class_imputer:
                self.imputers.append(ImputeRegression(strategy=self.class_imputer, verbose=0))
                yi = self.imputers[i](y)
            self.regressors[i].fit(Xb if self.best > 0 or self.class_imputer == "drop" else X,
                                   yi if self.class_imputer else y)

    def predict(self, X):
        if not self.regressors:
            raise Exception("fit first")
        Y_prime = np.zeros((X.shape[0].len(self.regressors)))

        if self.preprocessing:
            X = self.X_processor.transform(X)

        if self.best > 0 and not self.kbset:
            Xb = self.transformers.transform(X)
        for i in xrange(len(self.regressors)):
            if self.best > 0 and self.kbset:
                Xb = self.transformers[i].transform(X)

            Y_prime[:, i] = self.regressors[i].predict(Xb if self.bset > 0 else X)
        ret = self.Y_processor.inverse_transform(Y_prime) if self.preprocessing else Y_prime

        return ret.T[0] if ret.shape[1] == 1 else ret

    def _preprocess(self, X, Y):
        self.X_processor = PreProcessor(X, kind=self.preprocessing)
        self.Y_processor = PreProcessor(Y, kind=self.preprocessing)

    def _transform_to_best(self, X, y):
        if self.kbset:
            return SelectKBest(f_regression, self.best if self.best < X.shape[1] else "all").fit(X, y)
        else:
            return PCA(self.best).fit(X)

    def _init_regressor(self, regressor=None):

        if regressor is None:
            raise NotImplementedError
        if self.bagging:
            return skensemble.baggingregressor #....

        return regressor

#
# class SimpleRegressor(MultitargetWrapper):
#     POSSIBLE
#     STRATEGISE = ['mean', 'median', 'most_commin']
#
#
# class SMBRegressor(Wrapper):
#     def __init__(self):
#         wrapper.__init__
#
#
#     def _update_name(self):
#         def set params
#
#         def get params
#
#     def nit regressor
#
#     :
#     return wrapper._init_regrtessor(self, SVR(....))
#
#
# .NotImplementedError.
# .
# .
