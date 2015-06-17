from abc import ABCMeta
from collections import Counter

import numpy as np
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier

from analyse.test import score_mse


def _ensure_masked(array):
    if not np.ma.isMaskedArray(array) or not np.ma.is_masked(array):
        raise ValueError("Impute works with a masked array containing missing values")


class Drop:
    """
    Removes rows and columns with any missing values, columns first
    """

    def __init__(self, drop_threshold=.5, verbose=0):
        self.drop_threshold = drop_threshold
        self.verbose = verbose
        self.row_mask = None
        self.column_mask = None

    def fit(self, data):
        _ensure_masked(data)
        self.column_mask = None
        # masks columns with more than drop_threshold missing values, abort if all would  be dropped
        if len(data.shape) > 1:
            self.column_mask = data.mask.sum(axis=0) < self.drop_threshold * data.shape[0]
            if self.column_mask.sum() == 0:
                raise Exception("all columns would be dropped")

        if self.column_mask is None:
            self.row_mask = ~data.mask
        else:
            self.row_mask = data[:, self.column_mask].mask.sum(axis=1) == 0

        if self.verbose > 0:
            print "removed %d columns and %d rows" % (
                (~self.column_mask).sum() if self.column_mask is not None else -1, (~self.row_mask).sum())
        if self.verbose > 1:
            print "column indices:", np.where(~self.row_mask)[0] if self.row_mask is not None else -1
            print "row indices:", np.where(~self.row_mask)[0]

    def transform(self, data):
        if self.column_mask is not None:
            return data[np.where(self.row_mask)[0], :][:, np.where(self.column_mask)[0]]
        return data[np.where(self.row_mask)[0]]

    def fit_transform(self, data):
        self.fit(data)
        return self.transform(data)

    def __call__(self, data):
        return self.fit_transform(data)


class Impute:
    __metaclass__ = ABCMeta
    POSSIBLE_STRATEGIES = {}

    def __init__(self, strategy):
        if strategy not in self.POSSIBLE_STRATEGIES:
            raise ValueError("strategy not in POSSIBLE_STRATEGIES")
        self.strategy = strategy

    def fit(self, data, **kwargs):
        _ensure_masked(data)
        self.fitters = {}
        if len(data.shape) > 1:
            for idx, column in enumerate(data.T):
                if column.mask.sum() > 0:
                    self.fitters[idx] = getattr(self, "_fit_%s" % self.strategy)(data, column, idx, **kwargs)
        else:
            if self.strategy in [s for s, cls in self.POSSIBLE_STRATEGIES.iteritems() if cls is None]:
                self.fitters = getattr(self, "_fit_%s" % self.strategy)(None, data, None, **kwargs)
            else:
                raise ValueError("Only basic imputation techniques work on single column data")

    def transform(self, data):
        _ensure_masked(data)
        if not hasattr(self, "fitters"):
            raise Exception("fit first")

        ret = data.copy()
        if len(data.shape) > 1:
            for idx, column in enumerate(data.T):
                if column.mask.sum() > 0:
                    ret[column.mask, idx] = getattr(self, "_apply_%s" % self.strategy)(data, column, idx,
                                                                                       self.fitters[idx])
        else:
            if data.mask.sum() > 0:
                ret[data.mask] = getattr(self, "_apply_%s" % self.strategy)(None, data, None, self.fitters)
        return ret

    def fit_transform(self, data, **kwargs):
        self.fit(data, **kwargs)
        return self.transform(data)

    def __call__(self, data, *_, **kwargs):
        return self.fit_transform(data, **kwargs)

    def _setup_fitters(self):
        for name, fitter_cls in self.POSSIBLE_STRATEGIES.iteritems():
            if fitter_cls is not None:
                def fit_fun(self, data, column, idx, **kwargs):
                    train_columns = np.ones(data.shape[1], dtype=bool)
                    train_columns[idx] = False
                    fitter = fitter_cls(**kwargs)
                    fitter.fit(data[~column.mask, :][:, train_columns], column[~column.mask])
                    return fitter

                setattr(self, '_fit_%s' % name, fit_fun)

                def apply_fun(self, data, column, idx, fitter):
                    train_columns = np.ones(data.shape[1], dtype=bool)
                    train_columns[idx] = False
                    return fitter.predict(data[column.mask, :][:, train_columns])

                setattr(self, '_apply_%s' % name, apply_fun)


class ImputeClassification(Impute):
    POSSIBLE_STRATEGIES = {
        'majority': None,
        'tree': DecisionTreeClassifier,
        'knn': KNeighborsClassifier,
        'rf': RandomForestClassifier
    }

    def __init__(self, strategy='majority'):
        Impute.__init__(self, strategy=strategy)

        self._setup_fitters()

    def _fit_majority(self, _data, column, _idx):
        return Counter(column[~column.mask]).most_common(1)[0][0]

    def _apply_most_common(self, _data, _column, _idx, values):
        return values


class ImputeRegression(Impute):
    POSSIBLE_STRATEGIES = {
        'mean': None,
        'median': None,
        'tree': DecisionTreeRegressor,
        'knn': KNeighborsRegressor,
        'rf': RandomForestRegressor
    }

    def __init__(self, strategy='mean'):
        Impute.__init__(self, strategy=strategy)

        self._setup_fitters()

    def _fit_median(self, _data, column, _idx):
        return np.median(column[~column.mask])

    def _apply_median(self, _data, _column, _idx, median):
        return median

    def _fit_mean(self, _data, column, _idx):
        return column[~column.mask].mean()

    def _apply_mean(self, _data, _column, _idx, mean):
        return mean


class PreProcessor:
    POSSIBLE_KINDS = ['norm', 'std']

    def __init__(self, data=None, kind='norm'):
        if kind not in self.POSSIBLE_KINDS:
            raise ValueError("kind not in POSSIBLE_KINDS")
        self.norm = kind == 'norm'

        if data is not None:
            self.__call__(data)

    def __call__(self, data):
        self.fit(data)

    def fit(self, data):
        if self.norm:
            self.min = np.asarray(data.min(axis=0), dtype=float)
            self.max = np.asarray(data.max(axis=0), dtype=float)
        else:
            self.mean = data.mean(axis=0, dtype=float)
            self.std = np.std(data, axis=0, dtype=float)

    def transform(self, data):
        if self.norm:
            return np.nan_to_num((data - self.min) / (self.max - self.min))
        else:
            return np.nan_to_num((data - self.mean) / self.std)

    def inverse_transform(self, data):
        if self.norm:
            return np.nan_to_num(data * (self.max - self.min) + self.min)
        else:
            return np.nan_to_num(data * self.std + self.mean)

    def fit_transform(self, data):
        self.fit(data)
        return self.transform(data)
