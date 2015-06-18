from abc import ABCMeta, abstractmethod

import numpy as np
from sklearn.feature_selection import SelectKBest, f_regression, f_classif
from sklearn.svm import SVR
import sklearn.ensemble as skensemble
from sklearn.decomposition import PCA
import copy
from analyse.preprocess import ImputeRegression, ImputeClassification, PreProcessor, Drop
from sklearn.dummy import DummyRegressor

class TechniqueWrapper:
    __metaclass__ = ABCMeta
    POSSIBLE_TRANSFORMERS = ['kbest', 'pca']

    @staticmethod
    def _validate_inputs(preprocessor, transformer, transform_n, class_imputer):
        if preprocessor not in [None] + PreProcessor.POSSIBLE_KINDS:
            raise ValueError("preprocessor not in PreProcessor.POSSIBLE_KINDS")
        if transformer not in [None] + TechniqueWrapper.POSSIBLE_TRANSFORMERS:
            raise ValueError("transformer not in POSSIBLE_TRANSFORMERS")
        if transformer and transform_n is None:
            raise ValueError("transformer requires a number or 'all' best to be set")
        if class_imputer not in [None, 'drop'] + ImputeRegression.POSSIBLE_STRATEGIES:
            raise ValueError("class_imputer not in ImputeRegression.POSSIBLE_STRATEGIES")

    def __init__(self, technique, preprocessor=None, transform_n=0, transformer=None, class_imputer=None, **kwargs):
        TechniqueWrapper._validate_inputs(preprocessor, transformer, transform_n, class_imputer)

        self.technique = technique
        self.preprocessor = preprocessor
        self.class_imputer = class_imputer
        self.transformer = transformer
        self.transform_n = transform_n

        self._technique_instances = None
        self._transformers = None
        self._imputers = None

        self.kwargs = kwargs

        self.name = "MultitargetWrapper"

    def fit_predict(self, X_train, Y_train, X_test):
        self.fit(X_train, Y_train)
        return self.predict(X_test)

    def __call__(self, X_train, Y_train, X_test):
        return self.fit_predict(X_train, Y_train, X_test)

    def __str__(self):
        return self.name

    def get_params(self, deep=False):
        params = {
            'technique': self.technique,
            'preprocessor': self.preprocessor,
            'class_imputer': self.class_imputer,
            'transformer': self.transformer,
            'transform_n': self.transform_n,
        }
        params.update(**self.kwargs)
        return params

    def set_params(self, **kwargs):
        # TODO: test if destroyed, deepcopy if true
        # copy.deepcopy(kwargs)

        self.technique = kwargs.pop('technique', None)
        self.preprocessor = kwargs.pop('preprocessor', None)
        self.class_imputer = kwargs.pop('class_imputer', None)
        self.transformer = kwargs.pop('transformer', None)
        self.transform_n = kwargs.pop('transform_n', None)

        self.kwargs = kwargs

        TechniqueWrapper._validate_inputs(self.preprocessor, self.transformer, self.transform_n, self.class_imputer)
        self._technique_instances = None
        self._transformers = None
        self._imputers = None

    def fit(self, X, Y):
        self._singletarget = False
        self._technique_instances = []
        if self.transformer:
            self._transformers = []
        if self.class_imputer == "drop":
            self._imputers = Drop(drop_threshold=self.kwargs.get('drop_treshold', .5), verbose=0)
        elif self.class_imputer:
            self._imputers = []

        # if single target
        if len(Y.shape) == 1:
            self._singletarget = True
            Y = np.array([Y]).T

        if self.preprocessor:
            X, Y = self._preprocess(X, Y)

        if self.transformer == 'pca':
            self._transformers = self._transform_pca(X)
            X_transformed = self._transformers.transform(X)

        for i, y in enumerate(Y.T):
            self._technique_instances.append(self._init_technique())
            if self.transformer == 'kbest':
                self._transformers.append(self._transform_kbest(X, y))
                X_transformed = self._transformers[i].transform(X)

            if self.class_imputer == "drop":
                self._imputers.fit(y)
                y_transformed = y[self._imputers.row_mask]
                if self.transformer:
                    X_transformed = X_transformed[self._imputers.row_mask, :]
                else:
                    X_transformed = X[self._imputers.row_mask, :]
            elif self.class_imputer:
                self._imputers.append(self._get_imputer_instance())
                y_transformed = self._imputers[i](y)

            self._technique_instances[i].fit(X_transformed if self.transformer or self.class_imputer == "drop" else X,
                                             y_transformed if self.class_imputer else y)
        return self

    def predict(self, X):
        if not self._technique_instances:
            raise Exception("The technique needs to be fitted before predicting (call .fit)")
        Y_prime = np.zeros((X.shape[0], len(self._technique_instances)))

        if self.preprocessor:
            X = self._X_processor.transform(X)

        if self.transformer == 'pca':
            X_transformed = self._transformers.transform(X)

        for i in xrange(Y_prime.shape[1]):
            if self.transformer == 'kbest':
                X_transformed = self._transformers[i].transform(X)

            Y_prime[:, i] = self._technique_instances[i].predict(X_transformed if self.transformer else X)

        Y_return = self._Y_processor.inverse_transform(Y_prime) if self.preprocessor else Y_prime
        return Y_return.T[0] if self._singletarget else Y_return

    def _preprocess(self, X, Y):
        self._X_processor = PreProcessor(X, kind=self.preprocessor)
        self._Y_processor = PreProcessor(Y, kind=self.preprocessor)
        return self._X_processor.transform(X), self._Y_processor.transform(Y)

    def _transform_pca(self, X):
        pca = PCA(self.transform_n)
        pca.fit(X)
        return pca

    @abstractmethod
    def _transform_kbest(self, X, y):
        pass

    @abstractmethod
    def _get_imputer_instance(self):
        pass

    def _init_technique(self):
        return self.technique(**self.kwargs)


class ClassifierWrapper(TechniqueWrapper):
    def _get_imputer_instance(self):
        return ImputeClassification(strategy=self.class_imputer)

    def _transform_kbest(self, X, y):
        kbest = SelectKBest(f_classif, self.best if self.best < X.shape[1] else "all")
        kbest.fit(X, y)
        return kbest


class RegressionWrapper(TechniqueWrapper):
    def _get_imputer_instance(self):
        return ImputeRegression(strategy=self.class_imputer)

    def _transform_kbest(self, X, y):
        kbest = SelectKBest(f_regression, self.best if self.best < X.shape[1] else "all")
        kbest.fit(X, y)
        return kbest


