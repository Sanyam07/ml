import analyse.score as score
import numpy as np

np.random.seed(42)


def test_check_masked():
    y_true = np.arange(0, 10)
    y_pred = np.arange(1, 11)
    ret_true, ret_pred = score._check_masked(y_true, y_pred)
    assert y_true is ret_true
    assert y_pred is ret_pred

    y_true_ma = np.ma.arange(0, 10)
    assert y_true_ma.mask == False
    ret_true, ret_pred = score._check_masked(y_true_ma, y_pred)
    assert y_true_ma is ret_true

    mask = np.ones(10)
    mask[5:] = 0
    y_true_ma.mask = mask
    ret_true, ret_pred = score._check_masked(y_true_ma, y_pred)
    assert y_pred is not ret_pred
    assert np.all(ret_pred == np.arange(6, 11))
    assert np.all(y_pred == np.arange(1, 11))
    assert y_true_ma is not ret_true
    assert np.all(y_true_ma.data == np.arange(0, 10))
    assert np.all(y_true_ma.mask == mask)
    assert np.all(ret_true.data == np.arange(5, 10))
    assert np.all(ret_true.mask == np.zeros(5))


def test_score_mse():
    y_true = np.ma.arange(0, 10)
    y_true[:5] = 1
    y_pred = np.arange(1, 11)
    assert score.score_mse(y_true, y_pred) == 3.5

    y_true.mask = np.zeros(10)
    y_true.mask[:5] = 1
    assert score.score_mse(y_true, y_pred) == 1

    assert score.score_mse(np.arange(0, 10), y_pred) == 1
    assert score.score_mse(np.arange(0, 5), np.arange(3, 8)) == 9


def test_score_mae():
    y_true = np.ma.arange(0, 10)
    y_true[:5] = 1
    y_pred = np.arange(1, 11)
    assert score.score_mae(y_true, y_pred) == 1.5

    y_true.mask = np.zeros(10)
    y_true.mask[:5] = 1
    assert score.score_mae(y_true, y_pred) == 1

    assert score.score_mae(np.arange(0, 10), y_pred) == 1
    assert score.score_mae(np.arange(0, 5), np.arange(3, 8)) == 3


def test_score_r2():
    assert score.score_r2(np.zeros(5), np.zeros(5)) == 1.0
    assert score.score_r2(np.arange(5), np.arange(5)) == 1.0

    y_true = np.ma.arange(0, 10)
    y_true[:5] = 1
    y_pred = np.arange(1, 11)
    assert score.score_r2(y_true, y_pred) - 0.65 < 1e-5

    y_true.mask = np.zeros(10)
    y_true.mask[:5] = 1
    assert score.score_r2(y_true, y_pred) == 0.5

    assert score.score_r2(np.arange(0, 10), y_pred) - 0.87878 < 1e-5
    assert score.score_r2(np.arange(0, 5), np.arange(3, 8)) == -3.5


def test_score_mean_mt_mse():
    V3_true = np.vander(np.arange(3))
    V3_pred = np.vander(np.arange(1, 4))
    assert score.score_mean_mt_mse(V3_true, V3_pred) - 4.22222 < 1e-5

    V3_true_ma = np.ma.MaskedArray(V3_true)
    V3_true_ma.mask = np.zeros((3, 3))
    V3_true_ma.mask[2, :] = 1
    assert score.score_mean_mt_mse(V3_true_ma, V3_pred) == 2


def test_score_ca():
    assert score.score_ca(np.zeros(5), np.zeros(5)) == 1.0
    assert score.score_ca(np.zeros(5), np.ones(5)) == 0.0

    y_true = np.ma.arange(0, 10)
    y_true.mask = np.zeros(10)
    y_true.mask[:5] = 1
    y_pred = np.arange(0, 10)
    assert score.score_ca(y_true, y_pred) == 1.0
    y_pred[-1] = 1
    assert score.score_ca(y_true, y_pred) == 0.8


# TODO: score_median_mt_envelope, score_mean_mt_envelope,
# TODO: score_mean_mt_r2, Optimiser, pickling, Tester
