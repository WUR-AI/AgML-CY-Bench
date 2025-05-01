import pickle
import numpy as np
import logging
from collections.abc import Iterable
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge, Lasso
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import SelectFromModel
from sklearn.model_selection import GridSearchCV, GroupKFold
from sklearn.pipeline import Pipeline

from cybench.models.model import BaseModel
from cybench.datasets.dataset import Dataset
from cybench.util.data import data_to_pandas
from cybench.util.features import (
    unpack_time_series,
    design_features,
)

from cybench.config import (
    KEY_LOC,
    KEY_YEAR,
    KEY_TARGET,
    KEY_DATES,
    SOIL_PROPERTIES,
    YIELD_TREND_FEATURES,
    TIME_SERIES_INPUTS,
)


class BaseSklearnModel(BaseModel):
    """Base class for wrappers around scikit learn estimators."""

    def __init__(self, **kwargs):
        super(BaseModel, self).__init__()
        self._feature_cols = None
        self._predesigned_features = False
        if ("feature_cols" in kwargs) and kwargs["feature_cols"] is not None:
            self._feature_cols = kwargs["feature_cols"]
            self._predesigned_features = True

        assert "estimator" in kwargs and kwargs["estimator"] is not None
        estimator = kwargs["estimator"]

        if ("scaler" in kwargs) and kwargs["scaler"] is not None:
            scaler = kwargs["scaler"]
        else:
            scaler = StandardScaler()

        if ("ft_selector" in kwargs) and kwargs["ft_selector"] is not None:
            ft_selector = kwargs["ft_selector"]
            self._est = Pipeline(
                [
                    ("scaler", scaler),
                    ("selector", ft_selector),
                    ("estimator", estimator),
                ]
            )
        else:
            self._est = Pipeline([("scaler", scaler), ("estimator", estimator)])

        self._logger = logging.getLogger(__name__)

    def fit(
        self,
        dataset: Dataset,
        optimize_hyperparameters=False,
        select_features=False,
        **fit_params,
    ) -> tuple:
        """Fit or train the model.

        Args:
          dataset (Dataset): training dataset
          optimize_hyperparameters (bool): flag to optimize hyperparameters
          select_features (bool): flat to select features
          **fit_params: Additional parameters.

        Returns:
          A tuple containing the fitted model and a dict with additional information.
        """
        # NOTE: We want to support a dataset with pre-designed features.
        if self._predesigned_features:
            train_data = data_to_pandas(dataset)
            train_years = dataset.years
        else:
            train_features = self._design_features(dataset.crop, dataset)
            train_labels = data_to_pandas(
                dataset, data_cols=[KEY_LOC, KEY_YEAR, KEY_TARGET]
            )
            self._feature_cols = [
                ft for ft in train_features.columns if ft not in [KEY_LOC, KEY_YEAR]
            ]
            train_data = train_features.merge(train_labels, on=[KEY_LOC, KEY_YEAR])
            train_years = sorted(train_data[KEY_YEAR].unique())

        X = train_data[self._feature_cols].values
        y = train_data[KEY_TARGET].values
        if (len(train_years) > 1) and (optimize_hyperparameters or select_features):
            assert "param_space" in fit_params, "Parameter space not provided"
            param_space = fit_params["param_space"]
            assert param_space, "Parameter space is empty"

            # check select_features and pipeline are set correctly
            if select_features:
                assert "selector" in self._est.named_steps
            else:
                assert "selector" not in self._est.named_steps

            if select_features and ("max_features" in fit_params):
                assert fit_params["max_features"] is not None
                param_space["selector__max_features"] = fit_params["max_features"]
                # NOTE: Update max_features to have valid values.
                param_space["selector__max_features"] = [
                    v
                    for v in param_space["selector__max_features"]
                    if (len(self._feature_cols) > v)
                ]

            # NOTE: 1. optimize hyperparameters refits the estimator
            #          with the optimal hyperparameter values.
            #       2. use kfolds=len(train_years) for leave-one-out
            #
            cv_groups = train_data[KEY_YEAR].values
            self._est = self._optimize_hyperparameters(
                X,
                y,
                param_space,
                groups=cv_groups,
                kfolds=min(5, len(train_years)),
            )

        else:
            self._est.fit(X, y)

        return self, {}

    def _optimize_hyperparameters(
        self,
        X: np.ndarray,
        y: np.ndarray,
        param_space: dict,
        groups: np.ndarray = None,
        kfolds=5,
    ):
        """Optimize hyperparameters

        Args:
          X (np.ndarray): training features
          y (np.ndarray): training labels
          param_space (dict): hyperparameters to optimize
          groups (np.ndarray): group values (e.g year values) for each row in X and y
          kfolds (int): number of splits in cross validation

        Returns:
          A sklearn pipeline refitted with the optimal hyperparameters.
        """
        # regular k-fold cv
        cv = kfolds

        # GroupKFold to split by years
        if groups is not None:
            group_kfold = GroupKFold(n_splits=kfolds)
            # cv is here a list of tuples (train split, validation split)
            cv = group_kfold.split(X, y, groups)

        # Search for optimal value of hyperparameters
        grid_search = GridSearchCV(self._est, param_grid=param_space, cv=cv)
        grid_search.fit(X, y)
        best_params = grid_search.best_params_
        est = grid_search.best_estimator_

        self._logger.debug(est)
        if "selector" in est.named_steps:
            selector = est.named_steps["selector"]
            indices = selector.get_support(indices=True)
            self._logger.debug(f"Selected {len(indices)} Features")
            self._logger.debug([self._feature_cols[i] for i in indices])

        self._logger.debug("Optimal Hyperparameters")
        self._logger.debug(best_params)

        return est

    def _design_features(self, crop: str, data_items: Iterable):
        """Design features using data samples.

        Args:
          crop (str): crop name
          data_items (Iterable): a Dataset or list of data items.

        Returns:
          A pandas dataframe with KEY_LOC, KEY_YEAR and features.
        """
        # static data is repeated for every year. Drop duplicates.
        assert len(SOIL_PROPERTIES) > 0
        soil_df = data_to_pandas(data_items, data_cols=[KEY_LOC] + SOIL_PROPERTIES)
        soil_df = soil_df.drop_duplicates()

        dfs_x = {"soil": soil_df}
        for x, ts_cols in TIME_SERIES_INPUTS.items():
            df_ts = data_to_pandas(
                data_items, data_cols=[KEY_LOC, KEY_YEAR] + [KEY_DATES] + ts_cols
            )
            df_ts = unpack_time_series(df_ts, ts_cols)
            # fill in NAs
            df_ts = df_ts.astype({k: "float" for k in ts_cols})
            df_ts = (
                df_ts.set_index([KEY_LOC, KEY_YEAR, "date"])
                .sort_index()
                .interpolate(method="linear")
            )
            dfs_x[x] = df_ts.reset_index()

        features = design_features(crop, dfs_x)
        trend_features = data_to_pandas(data_items,
                                        data_cols=[KEY_LOC, KEY_YEAR] + YIELD_TREND_FEATURES)
        features = features.merge(trend_features, on=[KEY_LOC, KEY_YEAR])

        return features

    def _predict(self, crop: str, data_items: Iterable):
        """Utility method called by both `predict_items` and `predict`.

        Args:
          crop (str): crop name
          data_items (Iterable): a Dataset or a list of data items

        Returns:
          A tuple containing a np.ndarray and a dict with additional information.
        """
        if self._predesigned_features:
            test_data = data_to_pandas(data_items)
        else:
            test_features = self._design_features(crop, data_items)
            test_labels = data_to_pandas(
                data_items, data_cols=[KEY_LOC, KEY_YEAR, KEY_TARGET]
            )
            # Check features are the same for training and test data
            ft_cols = list(test_features.columns)[len([KEY_LOC, KEY_YEAR]) :]
            missing_features = [ft for ft in self._feature_cols if ft not in ft_cols]
            for ft in missing_features:
                test_features[ft] = 0.0

            test_features = test_features[[KEY_LOC, KEY_YEAR] + self._feature_cols]
            test_data = test_features.merge(test_labels, on=[KEY_LOC, KEY_YEAR])

        X_test = test_data[self._feature_cols].values

        return self._est.predict(X_test), {}

    def predict(self, dataset: Dataset, **predict_params):
        """Run fitted model on batched data items.

        Args:
          dataset (Dataset): test dataset
          **predict_params: Additional parameters.

        Returns:
          A tuple containing a np.ndarray and a dict with additional information.
        """
        return self._predict(dataset.crop, dataset)

    def predict_items(self, X: list, crop=None, **predict_params):
        """Run fitted model on a list of data items.

        Args:
          X (list): a list of data items, each of which is a dict
          crop (str): crop name
          **predict_params: Additional parameters.

        Returns:
          A tuple containing a np.ndarray and a dict with additional information.
        """
        assert crop is not None
        return self._predict(crop, X)

    def save(self, model_name: str):
        """Save model, e.g. using pickle.
        Check here for options to save and load scikit-learn models:
        https://scikit-learn.org/stable/model_persistence.html

        Args:
          model_name (str): Filename that will be used to save the model.
        """
        with open(model_name, "wb") as f:
            pickle.dump(self, f)

    def load(cls, model_name: str):
        """Deserialize a saved model.

        Args:
          model_name (str): Filename that was used to save the model.

        Returns:
          The deserialized model.
        """
        with open(model_name, "rb") as f:
            saved_model = pickle.load(f)

        return saved_model


class SklearnRidge(BaseSklearnModel):
    def __init__(self, feature_cols: list = None):
        ridge = Ridge(alpha=0.5)
        lasso_selector = SelectFromModel(Lasso(), threshold="median")
        kwargs = {
            "feature_cols": feature_cols,
            "estimator": ridge,
            "ft_selector": lasso_selector,
        }

        super().__init__(**kwargs)

    def fit(self, train_dataset: Dataset, **fit_params):
        """Fit or train the model.

        Args:
          train_dataset (Dataset): training dataset
          **fit_params: Additional parameters.

        Returns:
          A tuple containing the fitted model and a dict with additional information.
        """
        fit_params["select_features"] = True
        fit_params["optimize_hyperparameters"] = True
        fit_params["param_space"] = {
            "estimator__alpha": [0.01, 0.1, 1.0, 5.0, 10.0],
            "selector__estimator__alpha": [0.1, 1.0, 5.0],
            "selector__max_features": [20, 25, 30],
        }

        super().fit(train_dataset, **fit_params)


class SklearnRandomForest(BaseSklearnModel):
    def __init__(self, feature_cols: list = None):
        random_forest = RandomForestRegressor(
            oob_score=True, n_estimators=100, min_samples_leaf=5
        )

        kwargs = {
            "feature_cols": feature_cols,
            "estimator": random_forest,
        }

        super().__init__(**kwargs)

    def fit(self, train_dataset: Dataset, **fit_params):
        """Fit or train the model.

        Args:
          train_dataset (Dataset): training dataset
          **fit_params: Additional parameters.

        Returns:
          A tuple containing the fitted model and a dict with additional information.
        """
        fit_params["optimize_hyperparameters"] = True
        fit_params["param_space"] = {
            "estimator__n_estimators": [50, 100, 500],
        }

        super().fit(train_dataset, **fit_params)

class SklearnKNN(BaseSklearnModel):
    def __init__(self, feature_cols: list = None):
        knn = KNeighborsRegressor(weights="distance")

        kwargs = {
            "feature_cols": feature_cols,
            "estimator": knn,
        }

        super().__init__(**kwargs)

    def fit(self, train_dataset: Dataset, **fit_params):
        """Fit or train the model.

        Args:
          train_dataset (Dataset): training dataset
          **fit_params: Additional parameters.

        Returns:
          A tuple containing the fitted model and a dict with additional information.
        """
        fit_params["optimize_hyperparameters"] = True
        fit_params["param_space"] = {
            "estimator__n_neighbors": [3, 5, 7, 9],
        }

        super().fit(train_dataset, **fit_params)
