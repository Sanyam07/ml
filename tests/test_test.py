import analyse.test as test
import numpy as np

np.random.seed(42)


def _test_params(cls, params):
    for k, v in params.iteritems():
        assert hasattr(cls, k)
        assert getattr(cls, k) == v


class DummyLearner:
    def __init__(self, **_):
        self.shape = None
        pass

    def fit(self, _, Y):
        self.shape = Y.shape
        return self

    def predict(self, X):
        if len(self.shape) > 1:
            return np.zeros((X.shape[0], self.shape[1]))
        return np.zeros((X.shape[0], ))


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
