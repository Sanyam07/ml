from tests import *
from tests.utils import DummyLearner, _test_params

import analyse.test as test


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


def test_optimiser():
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
    Y_st = np.arange(40)

    Yp = optimiser.cross_validation(X, Y_mt, DummyLearner(), parallel=False)
    print Yp

    Yp = optimiser.cross_validation(X, Y_mt, DummyLearner(), parallel=True)
    print Yp

