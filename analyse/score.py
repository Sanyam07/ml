from itertools import izip
import numpy as np


def _check_masked(y_true, y_pred):
    if np.ma.isMaskedArray(y_true) and np.ma.is_masked(y_true):
        y_pred = y_pred[~y_true.mask]
        y_true = y_true[~y_true.mask]
    return y_true, y_pred


def score_mae(y_true, y_pred):
    y_true, y_pred = _check_masked(y_true, y_pred)
    return np.abs(y_pred - y_true).mean()


def score_ca(y_true, y_pred):
    y_true, y_pred = _check_masked(y_true, y_pred)
    return np.sum(y_true == y_pred) / float(y_true.size)


def score_r2(y_true, y_pred):
    y_true, y_pred = _check_masked(y_true, y_pred)
    numerator = ((y_true - y_pred) ** 2).sum(dtype=np.float64)
    denominator = ((y_true - np.average(y_true, axis=0)) ** 2).sum(dtype=np.float64)

    if denominator == 0:
        if numerator == 0:
            return 1.0
        else:
            return 0.0
    return 1. - numerator / denominator


def score_mse(y_true, y_pred):
    y_true, y_pred = _check_masked(y_true, y_pred)
    return ((y_pred - y_true) ** 2).mean()


def score_mean_mt_r2(Y_true, Y_pred):
    #: because envelope can't be pickled
    return np.array(
        [score_r2(y, yp) for y, yp in izip(Y_true.T, Y_pred.T)]).mean()


def score_mean_mt_mse(Y_true, Y_pred):
    #: because envelope can't be pickled
    return np.array(
        [score_mse(y, yp) for y, yp in izip(Y_true.T, Y_pred.T)]).mean()


def score_mean_mt_envelope(scorer):
    def fun(Y_true, Y_pred):
        return (np.array([scorer(y, yp) for y, yp in izip(Y_true.T, Y_pred.T)])).mean()

    return fun


def score_median_mt_envelope(scorer):
    def fun(Y_true, Y_pred):
        return np.median(np.array([scorer(y, yp) for y, yp in izip(Y_true.T, Y_pred.T)]))

    return fun

# TODO: add flat
# TODO: auto switch to MT
