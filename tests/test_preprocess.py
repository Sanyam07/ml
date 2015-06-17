import analyse.preprocess as pprocess
import numpy as np

np.random.seed(42)


def test_preprocessor():
    try:
        pprocess.PreProcessor(kind=None)
    except ValueError as e:
        assert e.message == "kind not in POSSIBLE_KINDS"
    try:
        pprocess.PreProcessor(kind="not in kind")
    except ValueError as e:
        assert e.message == "kind not in POSSIBLE_KINDS"
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


def test_impute_regression_mean():
    try:
        pprocess.ImputeRegression(strategy=None)
    except ValueError as e:
        assert e.message == "strategy not in POSSIBLE_STRATEGIES"
    try:
        pprocess.ImputeRegression(strategy='invalid')
    except ValueError as e:
        assert e.message == "strategy not in POSSIBLE_STRATEGIES"

    imputer = pprocess.ImputeRegression()
    assert imputer.strategy == 'mean'
    data = np.ma.MaskedArray(np.ones((10, 10)))
    try:
        imputer.fit(data)
    except ValueError as e:
        assert e.message == "Impute works with a masked array containing missing values"
    data.mask = np.zeros(data.shape)
    try:
        imputer.fit(data)
    except ValueError as e:
        assert e.message == "Impute works with a masked array containing missing values"

    data.mask = np.zeros(data.shape)
    for i in range(3, 8):
        data.mask[:i, i] = True
        data.data[:i, i] = 0

    imputed_data = imputer.fit_transform(data)
    assert np.all(imputed_data.data == np.ones(data.shape))
    assert np.all(imputed_data.mask == np.zeros(data.shape))




def test_drop():
    dropper = pprocess.Drop()
    assert dropper.drop_threshold == .5
    try:
        dropper.fit(np.ones((10, 10)))
    except ValueError as e:
        assert e.message == "Impute works with a masked array containing missing values"
    data = np.ma.MaskedArray(np.ones((10, 10)))
    try:
        dropper.fit(data)
    except ValueError as e:
        assert e.message == "Impute works with a masked array containing missing values"
    data.mask = np.zeros(data.shape)
    try:
        dropper.fit(data)
    except ValueError as e:
        assert e.message == "Impute works with a masked array containing missing values"

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
    try:
        dropper.fit(data)
    except Exception as e:
        assert e.message == "all columns would be dropped"

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
