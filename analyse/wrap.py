from sklearn.feature_selection import SelectKBest, f_regression, f_classif
from sklearn.decomposition import PCA

from analyse import *


class _AbstractSingleWrapper:
    __metaclass__ = ABCMeta
    POSSIBLE_TRANSFORMERS = ['kbest', 'pca']
    DEFAULT_NAME = "_AbstractSingleWrapper"

    @staticmethod
    def _validate_inputs(preprocessor, transformer, transform_n, class_imputer):
        if preprocessor not in [None] + preprocess.PreProcessor.POSSIBLE_KINDS:
            raise ValueError("preprocessor not in preprocess.PreProcessor.POSSIBLE_KINDS")
        if transformer not in [None] + _AbstractSingleWrapper.POSSIBLE_TRANSFORMERS:
            raise ValueError("transformer not in POSSIBLE_TRANSFORMERS")
        if transformer and transform_n is None:
            raise ValueError("transformer requires a number or 'all' best to be set")
        if class_imputer not in [None, 'drop'] + preprocess.ImputeRegression.POSSIBLE_STRATEGIES.keys():
            raise ValueError("class_imputer not in preprocess.ImputeRegression.POSSIBLE_STRATEGIES")

    def __init__(self, technique, preprocessor=None, transform_n=0, transformer=None, class_imputer=None, **kwargs):
        _AbstractSingleWrapper._validate_inputs(preprocessor, transformer, transform_n, class_imputer)

        self.technique = technique
        self.preprocessor = preprocessor
        self.class_imputer = class_imputer
        self.transformer = transformer
        self.transform_n = transform_n
        self.name = kwargs.pop('name', self.DEFAULT_NAME)

        self.kwargs = kwargs
        self._initialise_containers()

    def _initialise_containers(self):
        self._technique_instance = None
        self._transformer = None
        self._imputer = None

    def fit_predict(self, X_train, Y_train, X_test):
        self.fit(X_train, Y_train)
        return self.predict(X_test)

    def __call__(self, X_train, Y_train, X_test):
        return self.fit_predict(X_train, Y_train, X_test)

    def __str__(self):
        return self.name

    def get_params(self, deep=False):
        params = {
            'technique': self.technique,
            'preprocessor': self.preprocessor,
            'class_imputer': self.class_imputer,
            'transformer': self.transformer,
            'transform_n': self.transform_n,
        }
        params.update(**self.kwargs)
        return params

    def set_params(self, **kwargs):
        # TODO: test if destroyed, deepcopy if true
        # copy.deepcopy(kwargs)

        self.technique = kwargs.pop('technique', None)
        self.preprocessor = kwargs.pop('preprocessor', None)
        self.class_imputer = kwargs.pop('class_imputer', None)
        self.transformer = kwargs.pop('transformer', None)
        self.transform_n = kwargs.pop('transform_n', None)

        self.kwargs = kwargs

        _AbstractSingleWrapper._validate_inputs(self.preprocessor, self.transformer, self.transform_n,
                                                self.class_imputer)
        self._initialise_containers()

    def fit(self, X, y):
        self._2d_output = False
        if self.class_imputer == "drop":
            self._imputer = preprocess.Drop(drop_threshold=self.kwargs.get('drop_treshold', .5), verbose=0)

        # if 2d output
        if len(y.shape) == 2:
            self._2d_output = True
            y = y.T[0]

        if self.preprocessor:
            X, y = self._preprocess(X, y)

        if self.transformer:
            self._transformer = self._transform_pca(X) if self.transformer == 'pca' else self._transform_kbest(X, y)
            X_transformed = self._transformer.transform(X)

        self._technique_instance = self._init_technique()
        if self.class_imputer == "drop":
            self._imputer.fit(y)
            y_transformed = y[self._imputers.row_mask]
            if self.transformer:
                X_transformed = X_transformed[self._imputer.row_mask, :]
            else:
                X_transformed = X[self._imputer.row_mask, :]
        elif self.class_imputer:
            self._imputer = self._get_imputer_instance()
            y_transformed = self._imputers(y)

        self._technique_instance.fit(X_transformed if self.transformer or self.class_imputer == "drop" else X,
                                     y_transformed if self.class_imputer else y)
        return self

    def predict(self, X):
        if self._technique_instance:  # TODO: add class check
            raise Exception("The technique needs to be fitted before predicting (call .fit)")
        Y_prime = np.zeros((X.shape[0], ))

        if self.preprocessor:
            X = self._X_processor.transform(X)

        if self.transformer:
            X_transformed = self._transformers.transform(X)

        Y_prime[:] = self._technique_instance.predict(X_transformed if self.transformer else X)
        Y_return = self._Y_processor.inverse_transform(Y_prime) if self.preprocessor else Y_prime
        return np.asarray([Y_return]).T if self._2d_output else Y_return

    def _preprocess(self, X, Y):
        self._X_processor = preprocess.PreProcessor(X, kind=self.preprocessor)
        self._Y_processor = preprocess.PreProcessor(Y, kind=self.preprocessor)
        return self._X_processor.transform(X), self._Y_processor.transform(Y)

    def _transform_pca(self, X):
        pca = PCA(self.transform_n)
        pca.fit(X)
        return pca

    @abstractmethod
    def _transform_kbest(self, X, y):
        raise NotImplementedError()

    @abstractmethod
    def _get_imputer_instance(self):
        raise NotImplementedError()

    def _init_technique(self):
        return self.technique(**self.kwargs)


class _AbstractClassifier():
    __metaclass__ = ABCMeta

    def _get_imputer_instance(self):
        return preprocess.ImputeClassification(strategy=self.class_imputer)

    def _transform_kbest(self, X, y):
        kbest = SelectKBest(f_classif, self.transform_n if self.transform_n < X.shape[1] else "all")
        kbest.fit(X, y)
        return kbest


class _AbstractRegression():
    __metaclass__ = ABCMeta

    def _get_imputer_instance(self):
        return preprocess.ImputeRegression(strategy=self.class_imputer)

    def _transform_kbest(self, X, y):
        kbest = SelectKBest(f_regression, self.transform_n if self.transform_n < X.shape[1] else "all")
        kbest.fit(X, y)
        return kbest


class _AbstractMultiWrapper(_AbstractSingleWrapper):
    __metaclass__ = ABCMeta
    DEFAULT_NAME = "_AbstractMultiWrapper"

    def _initialise_containers(self):
        self._technique_instances = None
        self._transformers = None
        self._imputers = None

    def fit(self, X, Y):
        self._singletarget = False
        self._technique_instances = []
        if self.transformer:
            self._transformers = []
        if self.class_imputer == "drop":
            self._imputers = preprocess.Drop(drop_threshold=self.kwargs.get('drop_treshold', .5), verbose=0)
        elif self.class_imputer:
            self._imputers = []

        # if single target
        if len(Y.shape) == 1:
            self._singletarget = True
            Y = np.asarray([Y]).T

        if self.preprocessor:
            X, Y = self._preprocess(X, Y)

        if self.transformer == 'pca':
            self._transformers = self._transform_pca(X)
            X_transformed = self._transformers.transform(X)

        for i, y in enumerate(Y.T):
            self._technique_instances.append(self._init_technique())
            if self.transformer == 'kbest':
                self._transformers.append(self._transform_kbest(X, y))
                X_transformed = self._transformers[i].transform(X)

            if self.class_imputer == "drop":
                self._imputers.fit(y)
                y_transformed = y[self._imputers.row_mask]
                if self.transformer:
                    X_transformed = X_transformed[self._imputers.row_mask, :]
                else:
                    X_transformed = X[self._imputers.row_mask, :]
            elif self.class_imputer:
                self._imputers.append(self._get_imputer_instance())
                y_transformed = self._imputers[i](y)

            self._technique_instances[i].fit(X_transformed if self.transformer or self.class_imputer == "drop" else X,
                                             y_transformed if self.class_imputer else y)
        return self

    def predict(self, X):
        if not self._technique_instances:
            raise Exception("The technique needs to be fitted before predicting (call .fit)")
        Y_prime = np.zeros((X.shape[0], len(self._technique_instances)))

        if self.preprocessor:
            X = self._X_processor.transform(X)

        if self.transformer == 'pca':
            X_transformed = self._transformers.transform(X)

        for i in xrange(Y_prime.shape[1]):
            if self.transformer == 'kbest':
                X_transformed = self._transformers[i].transform(X)

            Y_prime[:, i] = self._technique_instances[i].predict(X_transformed if self.transformer else X)

        Y_return = self._Y_processor.inverse_transform(Y_prime) if self.preprocessor else Y_prime
        return Y_return.T[0] if self._singletarget else Y_return


class ClassificationSingleWrapper(_AbstractClassifier, _AbstractSingleWrapper):
    DEFAULT_NAME = "ClassificationSingleWrapper"


class RegressionSingleWrapper(_AbstractRegression, _AbstractSingleWrapper):
    DEFAULT_NAME = "RegressionSingleWrapper"


class ClassificationMultiWrapper(_AbstractClassifier, _AbstractMultiWrapper):
    DEFAULT_NAME = "ClassificationMultiWrapper"


class RegressionMultiWrapper(_AbstractMultiWrapper):
    DEFAULT_NAME = "RegressionMultiWrapper"

