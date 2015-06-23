import analyse.preprocess as pprocess
import numpy as np
import pytest

np.random.seed(42)


def test_preprocessor():
    with pytest.raises(ValueError) as e:
        pprocess.PreProcessor(kind=None)
    assert "kind not in POSSIBLE_KINDS" in str(e.value)
    with pytest.raises(ValueError) as e:
        pprocess.PreProcessor(kind="not in kind")
    assert "kind not in POSSIBLE_KINDS" in str(e.value)

    pp = pprocess.PreProcessor()
    assert pp.norm == True

    X = np.vander(np.arange(4))
    pp.fit(X)
    pp2 = pprocess.PreProcessor(data=X)
    assert np.allclose(pp.min, pp2.min) and np.allclose(pp.max, pp2.max)

    pp3 = pprocess.PreProcessor()
    XT = pp3.fit_transform(X)
    XT2 = pp.transform(X)
    assert np.allclose(pp.min, pp3.min) and np.allclose(pp.max, pp3.max)
    assert np.allclose(XT, XT2)


def test_preprocessor_norm():
    X = np.vander(np.arange(10))
    pp = pprocess.PreProcessor(kind='norm', data=X)
    assert np.allclose(pp.min, X.min(axis=0))
    assert np.allclose(pp.max, X.max(axis=0))

    X_transform = pp.transform(X)
    assert np.allclose(X_transform, np.nan_to_num((X - pp.min) / (pp.max - pp.min)))

    X_reversed = pp.inverse_transform(X_transform)
    assert np.allclose(X_reversed, X)
    assert np.allclose(X_reversed, np.nan_to_num(X_transform * (pp.max - pp.min) + pp.min))


def test_preprocessor_std():
    X = np.vander(np.arange(10))
    pp = pprocess.PreProcessor(kind='std', data=X)
    assert np.allclose(pp.std, np.std(X, axis=0))
    assert np.allclose(pp.mean, X.mean(axis=0))

    X_transform = pp.transform(X)
    assert np.allclose(X_transform, np.nan_to_num(((X - pp.mean) / pp.std)))

    X_reversed = pp.inverse_transform(X_transform)
    assert np.allclose(X_reversed, X)
    assert np.allclose(X_reversed, np.nan_to_num(X_transform * pp.std + pp.mean))


def basic_imputer_test(imputer_class, strategy, **kwargs):
    with pytest.raises(ValueError) as e:
        imputer_class(strategy=None)
    assert "strategy not in POSSIBLE_STRATEGIES" in str(e.value)
    with pytest.raises(ValueError) as e:
        imputer_class(strategy='invalid')
    assert "strategy not in POSSIBLE_STRATEGIES" in str(e.value)

    imputer = imputer_class(strategy=strategy)
    assert imputer.strategy == strategy
    data = np.ma.MaskedArray(np.ones((10, 10)))

    with pytest.raises(ValueError) as e:
        imputer.fit(data)
    assert "Impute works with a masked array containing missing values" in str(e.value)
    data.mask = np.zeros(data.shape)
    with pytest.raises(ValueError) as e:
        imputer.fit(data)
    assert "Impute works with a masked array containing missing values" in str(e.value)

    data.mask = np.zeros(data.shape)
    for i in range(3, 8):
        data.mask[:i, i] = True
        data.data[:i, i] = 0

    imputed_data = imputer.fit_transform(data, **kwargs)
    assert np.all(imputed_data.data == np.ones(data.shape))
    assert np.all(imputed_data.mask == np.zeros(data.shape))


def test_impute_regression_mean_median():
    basic_imputer_test(pprocess.ImputeRegression, 'mean')
    basic_imputer_test(pprocess.ImputeRegression, 'median')

    imputer_mean = pprocess.ImputeRegression()
    assert imputer_mean.strategy == 'mean'
    imputer_median = pprocess.ImputeRegression(strategy='median')

    # single column test
    data = np.ma.MaskedArray(np.arange(50, dtype=float))
    data[40:] = 100
    data.mask = np.zeros(50)
    data.mask[:10] = 1
    imputed_data = imputer_mean.fit_transform(data)
    assert np.allclose(imputed_data.data[:10], np.ones(10) * 43.375)
    assert np.all(imputed_data.mask == np.zeros(data.shape))

    imputed_data = imputer_median.fit_transform(data)
    assert np.allclose(imputed_data.data[:10], np.ones(10) * 29.5)
    assert np.all(imputed_data.mask == np.zeros(data.shape))


def test_impute_classification_majority():
    basic_imputer_test(pprocess.ImputeClassification, 'majority')

    imputer_majority = pprocess.ImputeClassification()
    assert imputer_majority.strategy == 'majority'

    # single column test
    data = np.ma.MaskedArray(np.arange(50))
    data[40:] = 100
    data.mask = np.zeros(50)
    data.mask[:10] = 1
    imputed_data = imputer_majority.fit_transform(data)
    assert np.allclose(imputed_data.data[:10], np.ones(10) * 100)
    assert np.all(imputed_data.mask == np.zeros(data.shape))


def test_impute_regression():
    basic_imputer_test(pprocess.ImputeRegression, 'tree')
    basic_imputer_test(pprocess.ImputeRegression, 'rf', n_estimators=5)
    basic_imputer_test(pprocess.ImputeRegression, 'knn', n_neighbors=3)

    # TODO: expand to specific tests


def test_impute_clasification():
    basic_imputer_test(pprocess.ImputeClassification, 'tree')
    basic_imputer_test(pprocess.ImputeClassification, 'rf', n_estimators=5)
    basic_imputer_test(pprocess.ImputeClassification, 'knn', n_neighbors=3)

    # TODO: expand to specific tests


def test_drop():
    dropper = pprocess.Drop()
    assert dropper.drop_threshold == .5
    with pytest.raises(ValueError) as e:
        dropper.fit(np.ones((10, 10)))
    assert "Impute works with a masked array containing missing values" in str(e.value)
    data = np.ma.MaskedArray(np.ones((10, 10)))
    with pytest.raises(ValueError) as e:
        dropper.fit(data)
    assert "Impute works with a masked array containing missing values" in str(e.value)
    data.mask = np.zeros(data.shape)
    with pytest.raises(ValueError) as e:
        dropper.fit(data)
    assert "Impute works with a masked array containing missing values" in str(e.value)

    data.mask = np.zeros(data.shape)
    for i in range(3, 8):
        data.mask[:i, i] = True
    dropper.fit(data)
    test_mask = np.ones(10, dtype=bool)
    test_mask[5:8] = 0
    assert np.all(dropper.column_mask == test_mask)
    test_mask = np.ones(10, dtype=bool)
    test_mask[:4] = 0
    assert np.all(dropper.row_mask == test_mask)
    imputed_data = dropper.transform(data)
    assert imputed_data.shape == (6, 7)

    dropper.drop_threshold = .2
    imputed_data = dropper.fit_transform(data)
    assert imputed_data.shape == (10, 5)

    dropper.drop_threshold = .9
    imputed_data = dropper.fit_transform(data)
    assert imputed_data.shape == (3, 10)

    data.mask = np.ones(data.shape)
    with pytest.raises(Exception) as e:
        dropper.fit(data)
    assert "all columns would be dropped" in str(e.value)

    # single column test
    data = np.ma.MaskedArray(np.ones(10))
    data.mask = np.zeros(10)
    data.mask[:2] = 1
    imputed_data = dropper.fit_transform(data)
    assert imputed_data.shape == (8, )
    data.mask[:6] = 1
    imputed_data = dropper.fit_transform(data)
    assert imputed_data.shape == (4, )
    data.mask[:10] = 1
    imputed_data = dropper.fit_transform(data)
    assert imputed_data.shape == (0, )
