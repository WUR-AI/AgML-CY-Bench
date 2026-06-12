import logging
from collections.abc import Iterable

from sklearn.svm import SVR

from cybench.models.sklearn_models import BaseSklearnModel
from cybench.datasets.dataset import Dataset


class SVRModel(BaseSklearnModel):
    """Support Vector Regression baseline using the BaseSklearnModel feature pipeline."""

    def __init__(self, feature_cols: list = None):
        """
        Args:
            feature_cols (list, optional): If provided, use these columns directly.
                If None, BaseSklearnModel will design features from the Dataset.
        """
        svr = SVR(
            kernel="rbf",
            C=1.0,
            epsilon=0.1,
        )

        kwargs = {
            "feature_cols": feature_cols,
            "estimator": svr,
        }

        super().__init__(**kwargs)

    def fit(
        self,
        train_dataset: Dataset,
        **fit_params,
    ) -> tuple:
        """Fit the SVR model with a simple hyperparameter search.

        Args:
            train_dataset (Dataset): training dataset
            **fit_params: Additional parameters passed to BaseSklearnModel.fit.
        """
        fit_params["optimize_hyperparameters"] = True
        fit_params["param_space"] = {
            "estimator__C": [0.1, 1.0, 10.0],
            "estimator__epsilon": [0.01, 0.1, 0.5],
        }

        super().fit(train_dataset, **fit_params)
