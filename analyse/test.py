from itertools import product, izip
from timeit import default_timer as time

import billiard
import sklearn.cross_validation as cv

from analyse import *


def _pickle_bypass(cls, function_name, *args, **kwargs):
    #: used to bypass multiprocessing
    return getattr(cls, function_name)(*args, **kwargs)


def _output_time(timer, simple=False):
    print ("%.2f" if simple else "** time taken: %.2f **") % (time() - timer)


class Optimiser:
    def __init__(self, scorer, maximise=True, folds=4, timeout=None, cores=2, verbose=1, shuffle=False,
                 random_state=42):
        self.folds = folds
        self.timeout = timeout
        self.cores = cores
        self.verbose = verbose
        self.shuffle = shuffle

        self.scorer = scorer
        self.maximise = maximise

        self.random_state = random_state

    @staticmethod
    def _grid_iterator(grid):
        keys, values = izip(*sorted(grid.items()))
        for v in product(*values):
            params = dict(izip(keys, v))
            yield params

    def _cross_validation(self, X, Y, technique_object, return_predictions=False):
        start_time = time()
        kf = cv.KFold(X.shape[0], n_folds=self.folds, shuffle=self.shuffle, random_state=self.random_state)
        Y_prime = np.zeros(Y.shape)

        for train_ind, test_ind in kf:
            X_train, X_test, Y_train = X[train_ind], X[test_ind], Y[train_ind]
            if len(Y_prime.shape) == 1:
                Y_prime[test_ind] = technique_object(X_train, Y_train, X_test)
            else:
                Y_prime[test_ind, :] = technique_object(X_train, Y_train, X_test)

        if return_predictions:
            return Y_prime, start_time
        else:
            return self.scorer(Y, Y_prime), start_time

    def _cross_validation_pool(self, X, Y, technique_object):
        start_time = time()
        pool = billiard.Pool(processes=self.cores, soft_timeout=self.timeout)
        kf = cv.KFold(X.shape[0], n_folds=self.folds, shuffle=self.shuffle, random_state=self.random_state)
        Y_prime = np.zeros(Y.shape)
        cv_res = []

        for train_ind, test_ind in kf:
            X_train, X_test, Y_train = X[train_ind], X[test_ind], Y[train_ind]
            r = pool.apply_async(technique_object, (X_train, Y_train, X_test))
            cv_res.append((r, test_ind))
        try:
            for r, test_ind in cv_res:
                if len(Y_prime.shape) == 1:
                    Y_prime[test_ind] = r.get(self.timeout + 5 if self.timeout else None)
                else:
                    Y_prime[test_ind, :] = r.get(self.timeout + 5 if self.timeout else None)
        except (billiard.SoftTimeLimitExceeded, billiard.TimeoutError):
            # invalidate whole run upon exception
            pool.terminate()
            if self.verbose > 3:
                print "** TERMINATED **"
            return None, start_time

        pool.close()
        pool.join()
        return Y_prime, start_time

    def _multiple_pool(self, X, Y, technique_classes, grids, return_best=True):
        pool = billiard.Pool(processes=self.cores, soft_timeout=self.timeout)
        pool_res = []
        results = []
        global_timer = time()

        for tech_c, grid in izip(technique_classes, grids):
            for params in self._grid_iterator(grid):
                technique_object = tech_c(**params)
                r = pool.apply_async(_pickle_bypass, (self, "_cross_validation", X, Y, technique_object))
                pool_res.append((r, tech_c, params))

        for r, reg_c_params in pool_res:
            start_time = time()
            err = False
            try:
                score, start_time = r.get(self.timeout + 5 if self.timeout else None)
                if self.verbose > 2:
                    print "** %.5f ** (%s, %s)" % (score, tech_c, params),
                results.append((score, tech_c, params))
            except billiard.SoftTimeLimitExceeded:
                print "** TIME LIMIT ** (%s, %s)" % (tech_c, params),
                err = True
            except billiard.TimeoutError:
                print "** TIMEOUT ** (%s, %s)" % (tech_c, params),
                err = True
            if self.verbose > 2 or err:
                _output_time(start_time, simple=True)
        pool.close()
        pool.join()

        if self.verbose > 1:
            print "** Optimisation done in %s **" % _output_time(global_timer)

        if return_best:
            best_tech = None
            best_score = -np.inf if self.maximise else np.inf
            best_params = None

            for score, tech_c, prms in results:
                if (self.maximise and score > best_score) or (not self.maximise and score < best_score):
                    best_score = score
                    best_tech = tech_c
                    best_params = prms
            if self.verbose > 1:
                print "** BEST: %.5f ** (%s, %s)" % (best_score, best_tech, best_params)

            return best_score, best_tech, best_params
        else:
            return results

    def optimise(self, X, Y, technique_class, grid, return_best=True):
        return self.optimise_multiple(X, Y, [technique_class], [grid], return_best=return_best)

    def optimise_multiple(self, X, Y, technique_classes, grids, return_best=True):
        self.verbose += 1
        if type(technique_classes) != list:
            technique_classes = [technique_classes]
        if type(grids) is dict:
            grids = [grids] * len(technique_classes)
        else:
            assert len(technique_classes) == len(grids), "The length of classes and grids must match"

        ret = self._multiple_pool(X, Y, technique_classes, grids, return_best=return_best)
        self.verbose -= 1
        return ret

    def cross_validation(self, X, Y, technique_object, parallel=True):
        self.verbose += 3
        if parallel:
            Y_prime, start_time = self._cross_validation_pool(X, Y, technique_object)
        else:
            Y_prime, start_time = self._cross_validation(X, Y, technique_object, return_predictions=True)
        self.verbose -= 3
        if self.verbose > 0:
            _output_time(start_time)
        return Y_prime

    def optimise_each_y(self, X, Y, technique_class, grid, return_best=True):
        return self.optimise_each_y_multiple(X, Y, [technique_class], [grid], return_best=return_best)

    def optimise_each_y_multiple(self, X, Y, regressor_class, grid, return_best=True):
        results = []
        global_timer = time()
        for i, y in enumerate(Y.T):
            if self.verbose > 0:
                print "** y_index: %d/%d **" % (i + 1, Y.shape[1])
            results.append(self.optimise_multiple(X, y, regressor_class, grid, return_best=return_best))
        if self.verbose > 0:
            print "** Individual optimisation done in %s **" % _output_time(global_timer, simple=True)
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
        self.best_learners = None
        self.best_parameters = None

    def _shuffle(self, X):
        self.shuffle_ind = np.arange(X.shape[0])
        np.random.shuffle(self.shuffle_ind)

    def _optimise_learners(self, X_train, Y_train, learners, parameters, scorer=score.score_mean_mt_r2,
                           maximise=True, timeout=100, cores=4, inner_folds=4, return_best=True, verbose=None):
        opt = Optimiser(scorer=scorer, maximise=maximise, cores=cores, verbose=verbose if verbose else self.verbose,
                        timeout=timeout, folds=inner_folds, random_state=self.random_state)

        if self.optimise_individual:
            return opt.optimise_each_y_multiple(X_train, Y_train, learners, parameters, return_best=return_best)
        else:
            return opt.optimise_multiple(X_train, Y_train, learners, parameters, return_best=return_best)

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
    def fit(self, learners, parameters, scorer=score.score_mean_mt_r2, maximise=True, timeout=100,
            cores=4, inner_folds=4, verbose=None):
        raise NotImplementedError

    @abstractmethod
    def predict(self):
        raise NotImplementedError

    def fit_predict(self, learners, parameters, **kwargs):
        self.fit(learners, parameters, **kwargs)
        return self.predict()


class TesterSplit(Tester):
    def __init__(self, train_ratio=.7, optimise_individual=False, random_state=42, verbose=1):
        Tester.__init__(self, optimise_individual=optimise_individual, random_state=random_state, verbose=verbose)

        self.train_ratio = train_ratio

    def initialise(self, X, Y):
        self._shuffle(X)

        N = int(X.shape[0] * self.train_ratio)
        self.X_train, self.X_test = X[self.shuffle_ind[:N], :], X[self.shuffle_ind[N:], :]
        self.Y_train, self.Y_test = Y[self.shuffle_ind[:N], :], Y[self.shuffle_ind[N:], :]

        self.initialised = True

        return self.X_train, self.X_test, self.Y_train, self.Y_test

    def fit(self, learners, parameters, scorer=score.score_mean_mt_r2, maximise=True, timeout=100, paralelise='grid',
            cores=4, inner_folds=4, verbose=None):
        if not self.initialised:
            raise Exception("Initialise first")
        if type(learners) != list:
            learners = [learners]
        results = self._optimise_learners(self.X_train, self.Y_train, learners, parameters, maximise=maximise,
                                          scorer=scorer, timeout=timeout, cores=cores, inner_folds=inner_folds,
                                          verbose=verbose)
        if self.optimise_individual:
            self.best_learners = [v[1] for v in results]
            self.best_parameters = [v[2] for v in results]

        else:
            self.best_learners = results[1]
            self.best_parameters = results[2]
        return self

    def predict(self):
        if self.optimise_individual:
            Y_hat = self.generate_prediction_from_individ(self.X_test, self.best_learners, self.best_parameters)
        else:
            best_l = self.best_learners(**self.best_parameters)
            best_l.fit(self.X_train, self.Y_train)
            Y_hat = best_l.predict(self.X_test)
        return Y_hat


class TesterCV(Tester):
    def __init__(self, folds=4, best_single_learner=False, optimise_individual=False, random_state=42, verbose=1):
        Tester.__init__(self, optimise_individual=optimise_individual, random_state=random_state, verbose=verbose)

        self.best_single_learner = best_single_learner
        self.folds = folds

    def initialise(self, X, Y):
        self._shuffle(X)

        self.kfold = cv.KFold(X.shape[0], n_folds=self.folds, shuffle=False, random_state=self.random_state)
        self.X = X[self.shuffle_ind, :]
        self.Y = Y[self.shuffle_ind, :]

        self.initialised = True

    def fit(self, learners, parameters, scorer=score.score_mean_mt_r2, maximise=True, timeout=100, cores=4,
            inner_folds=4, verbose=None):
        if not self.initialised:
            raise Exception("Initialise first")
        if type(learners) != list:
            learners = [learners]

        global_timer = time()
        classes, best_parametres = [], []

        Y_hat = np.zeros(self.Y.shape)
        for i, (train_ind, test_ind) in enumerate(self.kfold):
            if self.verbose > 0:
                print "## fold %d/%d (trains size %d, test size %d) ##"
            X_train, X_test, Y_train = self.X[train_ind], self.X[test_ind], self.Y[train_ind]
            results = self._optimise_learners(X_train, Y_train, learners, parameters, maximise=maximise,
                                              scorer=scorer, timeout=timeout, cores=cores,
                                              inner_folds=inner_folds, verbose=verbose, return_best=False)

            classes.append(cls)
            best_parametres.append(best_params)

            if self.optimise_individual:
                self.best_learners = [v[1] for v in results]
                self.best_parameters = [v[2] for v in results]

            else:
                self.best_learners = results[1]
                self.best_parameters = results[2]
            return self

        if self.verbose > 0:
            print "## CV done in %s ##" % _output_time(global_timer, simple=True)

        return Y_hat, classes, best_parametres, self

    def predict(self):
        if self.optimise_individual:
            Y_hat = self.generate_prediction_from_individ(self.X_test, self.best_learners, self.best_parameters)

            if len(Y_hat.shape) == 1:
                Y_hat[test_ind] = Yp
            else:
                Y_hat[test_ind, :] = Yp

        else:
            best_l = self.best_learners(**self.best_parameters)
            best_l.fit(self.X_train, self.Y_train)
            Y_hat = best_l.predict(self.X_test)
        return Y_hat
