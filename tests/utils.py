from tests import *
from time import sleep


def _test_params(cls, params):
    for k, v in params.iteritems():
        assert hasattr(cls, k)
        assert getattr(cls, k) == v


class DummyLearner:
    def __init__(self, sleep=0, **_):
        self.shape = None
        self.sleep = sleep

    def fit_predict(self, X_train, Y_train, X_test):
        self.fit(X_train, Y_train)
        return self.predict(X_test)

    def __call__(self, X_train, Y_train, X_test):
        return self.fit_predict(X_train, Y_train, X_test)

    def fit(self, _, Y):
        if self.sleep:
            sleep(self.sleep)
        self.shape = Y.shape
        return self

    def predict(self, X):
        if len(self.shape) > 1:
            return np.arange(X.shape[0] * self.shape[1]).reshape((X.shape[0], self.shape[1]))
        return np.arange(X.shape[0])
