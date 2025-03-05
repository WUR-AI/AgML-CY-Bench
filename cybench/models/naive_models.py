import pickle
import numpy as np
import logging

from cybench.models.model import BaseModel
from cybench.datasets.dataset import Dataset
from cybench.util.data import data_to_pandas
from cybench.config import KEY_LOC, KEY_YEAR, KEY_TARGET


class AverageYieldModel(BaseModel):
    """A naive yield prediction model.

    Predicts the average of the training set by location.
    If the location is not in the training data, then predicts the global average.
    """

    def __init__(self, group_by=[KEY_LOC]):
        self._group_by = group_by
        self._train_df = None
        self._logger = logging.getLogger(__name__)

    def fit(self, dataset: Dataset, **fit_params) -> tuple:
        """Fit or train the model.

        Args:
          dataset: Dataset

          **fit_params: Additional parameters.

        Returns:
          A tuple containing the fitted model and a dict with additional information.
        """
        sel_cols = [KEY_LOC, KEY_YEAR, KEY_TARGET]
        self._train_df = data_to_pandas(dataset, data_cols=sel_cols)
        # check group by columns are in the dataframe
        assert set(self._group_by).intersection(set(self._train_df.columns)) == set(
            self._group_by
        )
        self._averages = (
            self._train_df.groupby(self._group_by, observed=True)
            .agg(GROUP_AVG=(KEY_TARGET, "mean"))
            .reset_index()
        )

        return self, {}

    def predict_items(self, X: list):
        """Run fitted model on a list of data items.

        Args:
          X: a list of data items, each of which is a dict

        Returns:
          A tuple containing a np.ndarray and a dict with additional information.
        """
        predictions = np.zeros((len(X), 1))
        for i, item in enumerate(X):
            filter_condition = None
            for g in self._group_by:
                if filter_condition is None:
                    filter_condition = self._averages[g] == item[g]
                else:
                    filter_condition &= self._averages[g] == item[g]

            filtered = self._averages[filter_condition]
            # If there is no matching group in training data,
            # predict the global average
            if filtered.empty:
                self._logger.warning(
                    "No matching group found; predicting global average"
                )
                y_pred = self._train_df[KEY_TARGET].mean()
            else:
                y_pred = filtered["GROUP_AVG"].values[0]

            predictions[i] = y_pred

        return predictions.flatten(), {}

    def save(self, model_name):
        """Save model, e.g. using pickle.

        Args:
          model_name: Filename that will be used to save the model.
        """
        with open(model_name, "wb") as f:
            pickle.dump(self, f)

    def load(cls, model_name):
        """Deserialize a saved model.

        Args:
          model_name: Filename that was used to save the model.

        Returns:
          The deserialized model.
        """
        with open(model_name, "rb") as f:
            saved_model = pickle.load(f)

        return saved_model
