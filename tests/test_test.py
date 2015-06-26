from tests import *
from tests.utils import DummyLearner, _test_params

import analyse.test as test
import analyse.score as score


def test_grid_iterator():
    parameter_grid = {
        'p1': range(4),
        'p2': range(3),
        'p3': [True, False],
    }

    parameters_list = [p for p in test.Optimiser._grid_iterator(parameter_grid)]
    for params in parameters_list:
        assert params['p1'] in range(4)
        assert params['p2'] in range(3)
        assert params['p3'] in [True, False]
    assert len(parameters_list) == 24

    with pytest.raises(TypeError) as e:
        _ = [_ for _ in test.Optimiser._grid_iterator({'not_range': None})]


class _PickleTest:
    def fun(self, *args, **kwargs):
        return args, kwargs


def test_pickle_bypass():
    t = _PickleTest()
    pool = test.billiard.Pool(1)
    r = pool.apply_async(test._pickle_bypass, (t, 'fun', 'arg1', 'arg2'), {'kwarg': 'val'})
    pool.close()
    pool.join()
    arg, kwargs = r.get()
    assert arg == ('arg1', 'arg2')
    assert kwargs == {'kwarg': 'val'}


def test_optimiser_crossvalidation():
    default_params = {
        'maximise': True,
        'folds': 4,
        'timeout': None,
        'cores': 2,
        'verbose': 1,
        'shuffle': False,
        'random_state': 42
    }
    optimiser = test.Optimiser(None)
    _test_params(optimiser, default_params)

    X = np.arange(200).reshape((20, 10))
    Y_mt = np.arange(40).reshape((20, 2))
    Y_st = np.arange(20)
    Y_mt_correct = np.vstack(np.arange(10).reshape(5, 2) for _ in xrange(4))
    Y_st_correct = np.hstack(np.arange(5) for _ in xrange(4))

    Y_hat = optimiser.cross_validation(X, Y_mt, DummyLearner(), parallel=False)
    assert np.all(Y_hat == Y_mt_correct)
    Y_hat = optimiser.cross_validation(X, Y_mt, DummyLearner(), parallel=True)
    assert np.all(Y_hat == Y_mt_correct)

    Y_hat = optimiser.cross_validation(X, Y_st, DummyLearner(), parallel=False)
    assert np.all(Y_hat == Y_st_correct)
    Y_hat = optimiser.cross_validation(X, Y_st, DummyLearner(), parallel=True)
    assert np.all(Y_hat == Y_st_correct)

    optimiser.timeout = 1
    Y_hat = optimiser.cross_validation(X, Y_mt, DummyLearner(sleep=2), parallel=True)
    assert Y_hat is None


def test_optimiser():
    optimiser = test.Optimiser(score.score_mean_mt_mse, maximise=False)

    X = np.arange(200).reshape((20, 10))
    Y_mt = np.arange(40).reshape((20, 2))
    Y_st = np.arange(20)
    Y_mt_correct = np.vstack(np.arange(10).reshape(5, 2) for _ in xrange(4))
    Y_st_correct = np.hstack(np.arange(5) for _ in xrange(4))

    params = {
        'dummy1': [True, False],
        'dummy2': [0, 1]
    }
    optimiser.verbose = 5
    Y_hat = optimiser.optimise(X, Y_mt, DummyLearner, params)
    print Y_hat


if __name__ == "__main__":
    test_pickle_bypass()
    test_optimiser_crossvalidation()
    test_optimiser()
