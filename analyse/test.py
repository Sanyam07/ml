from time import time
from itertools import product, izip
from abc import ABCMeta, abstractmethod

import billiard
import sklearn.cross_validation as cv
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
# TOOO: auto switch to MT

def _pickle_bypass(cls, function_name, *args, **kwargs):
    #: used to bypass multiprocessing
    return getattr(cls, function_name)(*args, **kwargs)


class Optimiser:
    def __init__(self, scorer, maximise=True, folds=4, timeout=None, cores=2, verbose=0, shuffle=False,
                 parallelise='grid', random_state=42):
        if parallelise not in [None, 'grid', 'cv']:
            raise ValueError
        self.folds = folds
        self.timeout = timeout
        self.cores = cores
        self.verbose = verbose
        self.shuffle = shuffle

        self.scorer = scorer
        self.maximise = maximise
        self.parallelise_grid = parallelise == 'grid'
        self.parallelise_cv = parallelise == 'cv'

        self.random_state = random_state

    @staticmethod
    def _grid_iterator(grid):
        keys, values = izip(*sorted(grid.items()))
        for v in product(*values):
            params = dict(izip(keys, v))
            yield params

    def _cross_validation(self, X, Y, regressor):
        start_time = time()
        kf = cv.KFold(X.shape[0], n_folds=self.folds, shuffle=self.shuffle, random_state=self.random_state)
        Y_prime = np.zeros(Y.shape)
        for train_ind, test_ind in kf:
            X_train, X_test, Y_train = X[train_ind], X[test_ind], Y[train_ind]
            if len(Y_prime.shape) == 1:
                Y_prime[test_ind] = regressor(X_train, Y_train, X_test)
            else:
                Y_prime[test_ind, :] = regressor(X_train, Y_train, X_test)
        return self.scorer(Y, Y_prime), start_time


    def _cross_validation_pool(self, X, Y, regressor):
        pool = billiard.Pool(processes=self.cores, soft_timeout=self.timeout)
        kf = cv.KFold(X.shape[0], n_folds=self.folds, shuffle=self.shuffle, random_state=self.random_state)
        Y_prime = np.zeros(Y.shape)
        cv_res = []

        for train_ind, test_ind in kf:
            X_train, X_test, Y_train = X[train_ind], X[test_ind], Y[train_ind]
            r = pool.apply_async(regressor, (X_train, Y_train, X_test))
            cv_res.append((r, test_ind))
        try:
            for r, test_ind in cv_res:
                if len(Y_prime.shape) == 1:
                    Y_prime[test_ind] = r.get(self.timeout + 5)
                else:
                    Y_prime[test_ind, :] = r.get(self.timeout + 5)
        except (billiard.SoftTimeLimitExceeded, billiard.TimeoutError):
            pool.terminate()
            if self.verbose > 2:
                print "TERMINATED"
            return None

        pool.close()
        pool.join()
        if self.verbose > 2:
            print "CLOSED"
        return Y_prime

    def _grid_optimize_pool(self, X, Y, regressor_class, grid):
        pool = billiard.Pool(processes=self.cores, soft_timeout=self.timeout)
        pool_res = []
        results = []

        for params in self._grid_iterator(grid):
            regressor = regressor_class(**params)
            r = pool.apply_async(_pickle_bypass, (self, "_cross_validation", X, Y, regressor))
            pool_res.append((r, params))

        for r, reg_c_params in pool_res:
            start_time = time()
            err_msg = False
            try:
                score, start_time = r.get(self.timeout + 5)
                if self.verbose > 1:
                    print "* %.5t *" % score, regressor_class.__name__, params
                results.append((score, params))
            except billiard.SoftTimeiLmitExceeded:
                print "TIME LIM"
                err_msg = True
            except billiard.TimeoutError:
                print "* TIMEOUT * same shit"
                err_msg = True
            if self.verbose > 1 or err_msg:
                print "time"
        pool.terminate()  # TODO still not sure
        pool.join()
        if self.verbose > 1:
            print "Terminated"

        results.sort(reverse=self.maximise)
        return results[0] if results else None


    def _multiple_pool(self, X, Y, regressor_classes, grids):
        # only valid solution
        pool = billiard.Pool(processes=self.cores, soft_timeout=self.timeout)
        pool_res = []
        results = []
        timer = time()

        if type(grids) is dict:
            grids = [grids] * len(regressor_classes)
        else:
            assert len(regressor_classes) == len(grids)

        for reg_c, grid in izip(regressor_classes, grids):
            for params in self._grid_iterator(grid):
                regressor = reg_c(**params)

                r = pool.apply_async(_pickle_bypass, (self, "_cross_validation", X, Y, regressor))
                pool_res.append((r, reg_c, params))

        for r, reg_c_params in pool_res:
            start_time = time()
            try:
                score, start_time = r.get(self.timeout + 5)
                if self.verbose > 1:
                    print "* %.5t *" % score, reg_c, params
                results.append((score, reg_c, params))
            except billiard.SoftTimeLimitExceeded:
                print "TIME LIM"
            except billiard.TimeoutError:
                print "* TIMEOUT * same shit"
            if self.verbose > 1:
                print "time"
        pool.terminate()  # TODO still not sure
        pool.join()

        best_reg = None
        best_score = -np.inf if self.maximise else np.inf
        best_params = None

        for score, reg_c, prms in results:
            if (self.maximise and score > best_score) or (not self.maximise and score < best_score):
                best_score = score
                best_reg = reg_c
                best_params = prms

        if self.verbose > 1:
            print "multiple done in timer"
            print best_score, best_reg, best_params

        return best_score, best_reg, best_params

    def optimise(self, X, Y, regressor_class, grid):
        if self.parallelise_grid:
            return self._grid_optimize_pool(X, Y, regressor_class, grid)
        else:
            return self._grid_optimize(X, Y, regressor_class, grid)

    def optimise_multiple(self, X, Y, regressor_classes, grid):
        if self.parallelise_grid:
            return self._multiple_pool(X, Y, regressor_classes, grid)
        else:
            return self._multiple_cv(X, Y, regressor_classes, grid)

    def cross_validation(self, X, Y, regressor):
        return self._cross_validation_pool(X, Y, regressor)

    def cross_validation_nonparallel(self, X, Y, regressor):
        return self._cross_validation(X, Y, regressor)

    def optimise_each_y(self, X, Y, regressor_class, grid):
        results = []
        timer = time()
        for i, y in enumerate(Y.T):
            print "y_index: %d/%d" % (i + 1, Y.shape[1])
            if self.parallelise_grid:
                results.append(self._grid_optimise_pool(X, y, regressor_class, grid))
            else:
                results.append(self._grid_optimise(X, y, regressor_class, grid))
        if self.verbose > 1:
            print "individial optimisation done in %.1fs" % (time() - timer)
        return results

    def optimise_each_y_multiple(self, X, Y, regressor_class, grid):
        results = []
        timer = time()
        for i, y in enumerate(Y.T):
            print "y_index: %d/%d" % (i + 1, Y.shape[1])
            results.append(self.optimise_multiple(X, y, regressor_class, grid))
        if self.verbose > 1:
            print "individial optimisation done in %.1fs" % (time() - timer)
        return results


class Tester():
    __metaclass__ = ABCMeta

    def __init__(self, optimise_individual=False, random_state=42, verbose=1):
        self.optimise_individual = optimise_individual
        self.verbose = verbose
        self.random_state = random_state
        if type(random_state) == int:
            np.random.seed(random_state)
        else:
            np.random.set_state(random_state)

        self.initialised = False


    def _shuffle(self, X):
        self.shuffle_ind = np.arange(X.shape[0])
        np.random.shuffle(self.shuffle_ind)

    def optimise_learners(self, X_train, X_test, Y_train, learners, parameters, scorer=score_mean_mt_r2, maximise=True,
                          timeout=100, parallelise='grid', cores=4, inner_folds=4, verbose=None):
        opt = Optimiser(scorer=scorer, maximise=maximise, cores=cores, verbose=verbose if verbose else self.verbose,
                        timeout=timeout, folds=inner_folds, parallelise=parallelise, random_state=self.random_state)

        if self.optimise_individual:
            res = opt.optimise_each_y_multiple(X_train, Y_train, learners, parameters)

            classes = [v[1] for v in res]
            params = [v[2] for v in res]
            Y_hat = self.generate_prediction_from_individ(X_test, classes, params)  # TODO shouldnt work

            return Y_hat, classes, params
        else:
            best_score, cls, best_params = opt.optimise_multiple(X_train, Y_train, learners, parameters)

            best_l = cls(**best_params)
            best_l.fit(X_train, Y_train)
            Y_hat = best_l.predict(X_test)
            return Y_hat, cls, best_params

    @staticmethod
    def generate_prediction_from_individ(X_train, Y_train, X_test, learner_classes, parameters):
        Y_hat = np.zeros((X_test.shape[0], Y_train.shape[1]))
        for i, cls, params, y in izip(range(len(learner_classes)), learner_classes, parameters, Y_train.T):
            reg = cls(**params)
            reg.fit(X_train, y)
            Y_hat[:, i] = reg.predict(X_test)
        return Y_hat

    @abstractmethod
    def initialise(self, X, Y):
        raise NotImplementedError


    @abstractmethod
    def learn(self, learners, parameters, scorer=score_mean_mt_r2, maximise=True, timeout=100, paralelise="grid",
              cores=4,
              inner_folds=4, verbose=None):
        raise NotImplementedError


class TesterSplit(Tester):
    pass  # TODO


class TesterCV(Tester):
    def __init__(self, folds=4, optimise_individual=False, random_state=42, verbose=1):
        Tester.__init__(self, optimise_individual=optimise_individual, random_state=random_state, verbose=verbose)
        self.folds = folds

    def initialise(self, X, Y):
        self._shuffle(X)

        self.kfold = cv.KFold(X.shape[0], nfolds=self.folds, shuffle=False, random_state=self.random_state)
        self.X = X[self.shuffle_ind, :]
        self.Y = Y[self.shuffle_ind, :]

        self.initialised = True

    def learn(self, learners, parameters, scorer=score_mean_mt_r2, maximise=True, timeout=100, paralelise="grid",
              cores=4,
              inner_folds=4, verbose=None):
        if type(learners) != list:
            learners = [learners]

        timer = time()
        classes, best_parametres = [], []

        Y_hat = np.zeros(self.Y.shape)
        for i, (train_ind, test_ind) in enumerate(self.kfold):
            if self.verbose > 0:
                print "fold%d/%d, trains size %d, test size %d"
            X_train, X_test, Y_train = self.X[train_ind], self.X[test_ind], self.Y[train_ind]
            Yp, cls, best_params = self._optimise_learners(X_train, X_test, Y_train, learners,
                                                           parameters)  # TODO everzthing eslse)

            if len(Y_hat.shape) == 1:
                Y_hat[test_ind] = Yp
            else:
                Y_hat[test_ind, :] = Yp

            classes.append(cls)
            best_parametres.append(best_params)

        if self.verbose > 0:
            print "CV done in"

        return Y_hat, classes, best_params, self

