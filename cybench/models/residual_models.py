import numpy as np
import pandas as pd
import pickle

from cybench.datasets.dataset import Dataset
from cybench.datasets.modified_dataset import ModifiedTargetsDataset
from cybench.models.model import BaseModel
from cybench.models.trend_models import TrendModel
from cybench.models.sklearn_models import SklearnRidge, SklearnRandomForest
from cybench.models.xgboost_model import XGBoostModel
from cybench.models.nn_models import (
    BaselineLSTM,
    BaselineInceptionTime,
    BaselineTransformer,
)
from cybench.util.data import data_to_pandas

from cybench.config import (
    KEY_LOC,
    KEY_YEAR,
    KEY_TARGET,
)


class ResidualModel(BaseModel):
    def __init__(self, baseline_model: BaseModel):
        """Ridge model that predicts residuals from the trend.

        It leverages `TrendModel` to estimate the trend.
        Then it detrends targets and predicts the residuals.
        During inference, residual predictions are added to trend to
        produce the target predictions.
        """
        self._trend_model = TrendModel()
        self._baseline_model = baseline_model

    def fit(self, dataset: Dataset, **fit_params):
        """Fit or train the model.

        Args:
          train_dataset (Dataset): training dataset
          **fit_params: Additional parameters.

        Returns:
          A tuple containing the fitted model and a dict with additional information.
        """
        train_df = data_to_pandas(dataset, data_cols=[KEY_LOC, KEY_YEAR, KEY_TARGET])

        train_residuals = []
        for item in dataset:
            trend_train_df = train_df[
                (train_df[KEY_LOC] == item[KEY_LOC])
                & (train_df[KEY_YEAR] != item[KEY_YEAR])
            ].copy()
            # only one year of training data for this location
            if trend_train_df.empty:
                trend_pred = train_df[KEY_TARGET].mean()
            else:
                trend_train_df.set_index([KEY_LOC, KEY_YEAR], inplace=True)
                trend_train_dataset = Dataset(dataset.crop, data_target=trend_train_df)
                self._trend_model.fit(trend_train_dataset)
                trend_pred, _ = self._trend_model.predict_items([item])
                if not isinstance(trend_pred, float):
                    trend_pred = trend_pred[0]

            res_row = {
                KEY_LOC: item[KEY_LOC],
                KEY_YEAR: item[KEY_YEAR],
                KEY_TARGET: item[KEY_TARGET] - trend_pred,
            }
            train_residuals.append(res_row)

        residuals_df = pd.DataFrame(train_residuals)
        residuals_df.set_index([KEY_LOC, KEY_YEAR], inplace=True)
        residuals_dataset = ModifiedTargetsDataset(
            dataset, modified_targets=residuals_df
        )
        self._baseline_model.fit(residuals_dataset, **fit_params)

        # Fit the trend model on the entire training data
        self._trend_model.fit(dataset)

        return self, {}

    def predict(self, dataset: Dataset, **predict_params):
        """Run fitted model on batched data items.

        Args:
          dataset (Dataset): test dataset
          **predict_params: Additional parameters.

        Returns:
          A tuple containing a np.ndarray and a dict with additional information.
        """
        res_preds, _ = self._baseline_model.predict(dataset, **predict_params)
        trend_preds, _ = self._trend_model.predict(dataset, **predict_params)

        return np.add(trend_preds, res_preds), {}

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
        res_preds, _ = self._baseline_model.predict_items(
            X, crop=crop, **predict_params
        )
        trend_preds, _ = self._trend_model.predict_items(X, **predict_params)

        return np.add(trend_preds, res_preds), {}

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


class RidgeRes(ResidualModel):
    def __init__(self, feature_cols: list = None):
        """Ridge model that predicts residuals from the trend."""
        super().__init__(SklearnRidge(feature_cols=feature_cols))


class RandomForestRes(ResidualModel):
    def __init__(self, feature_cols: list = None):
        """RandomForest model that predicts residuals from the trend."""
        super().__init__(SklearnRandomForest(feature_cols=feature_cols))


class XGBoostRes(ResidualModel):
    def __init__(self, **kwargs):
        """XGBoost model that predicts residuals from the trend."""
        super().__init__(XGBoostModel(**kwargs))


class LSTMRes(ResidualModel):
    def __init__(self, **kwargs):
        """LSTM model that predicts residuals from the trend."""
        super().__init__(BaselineLSTM(**kwargs))


class InceptionTimeRes(ResidualModel):
    def __init__(self, **kwargs):
        """InceptionTime model that predicts residuals from the trend."""
        super().__init__(BaselineInceptionTime(**kwargs))


class TransformerRes(ResidualModel):
    def __init__(self, **kwargs):
        """Transformer model that predicts residuals from the trend."""
        super().__init__(BaselineTransformer(**kwargs))
