import logging
from collections.abc import Iterable

from xgboost import XGBRegressor

from cybench.models.sklearn_models import BaseSklearnModel
from cybench.datasets.dataset import Dataset


class XGBoostModel(BaseSklearnModel):
    """XGBoost baseline using the BaseSklearnModel feature pipeline."""

    def __init__(self, feature_cols: list = None):
        """
        Args:
            feature_cols (list, optional): If provided, use these columns directly.
                If None, BaseSklearnModel will design features from the Dataset.
        """
        xgb = XGBRegressor(
            objective="reg:squarederror",
            n_estimators=300,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            tree_method="hist",
            random_state=42,
            n_jobs=-1,
        )

        kwargs = {
            "feature_cols": feature_cols,
            "estimator": xgb,
        }

        super().__init__(**kwargs)

    def fit(
        self,
        train_dataset: Dataset,
        **fit_params,
    ) -> tuple:
        """Fit the XGBoost model with a simple hyperparameter search.

        Args:
            train_dataset (Dataset): training dataset
            **fit_params: Additional parameters passed to BaseSklearnModel.fit.

        Returns:
            tuple: (fitted model, info dict)
        """
        fit_params["optimize_hyperparameters"] = True
        fit_params["param_space"] = {
            "estimator__n_estimators": [50, 100, 500],
            "estimator__learning_rate": [0.01, 0.05, 0.1],
        }

        super().fit(train_dataset, **fit_params)    
