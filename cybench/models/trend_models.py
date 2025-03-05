import pickle
import numpy as np
from collections.abc import Iterable
from statsmodels.regression.linear_model import OLS
from statsmodels.tools.tools import add_constant
import pymannkendall as trend_mk

from cybench.models.model import BaseModel
from cybench.datasets.dataset import Dataset
from cybench.util.data import data_to_pandas

from cybench.config import KEY_LOC, KEY_YEAR, KEY_TARGET


class TrendModel(BaseModel):
    """Default trend estimator.

    Trend is estimated using years as features.
    """

    MIN_TREND_WINDOW_SIZE = 5
    MAX_TREND_WINDOW_SIZE = 10

    def __init__(self):
        self._train_df = None

    def fit(self, dataset: Dataset, **fit_params) -> tuple:
        """Fit or train the model.
        Args:
          dataset: Dataset
          **fit_params: Additional parameters.

        Returns:
          A tuple containing the fitted model and a dict with additional information.
        """
        self._train_df = data_to_pandas(
            dataset, data_cols=[KEY_LOC, KEY_YEAR, KEY_TARGET]
        )
        # NOTE: We save training data and do trend estimation during inference.

        return self, {}

    def _estimate_trend(self, trend_x: list, trend_y: list, test_x: int):
        """Implements a linear trend.
        From @mmeronijrc: Small sample sizes and the use of quadratic or loess trend
        can lead to strange results.

        Args:
          trend_x (list): year in the trend window.
          trend_y (list): values (e.g. yields) in the trend window
          test_x (int): test year

        Returns:
          estimated trend (float)
        """
        assert len(trend_y) >= self.MIN_TREND_WINDOW_SIZE
        trend_x = add_constant(trend_x)
        linear_trend_est = OLS(trend_y, trend_x).fit()
        pred_x = np.array([test_x]).reshape((1, 1))
        pred_x = add_constant(pred_x, has_constant="add")

        return linear_trend_est.predict(pred_x)[0]

    def _find_optimal_trend_window(
        self, train_labels: np.ndarray, window_years: list, extend_forward: bool = False
    ):
        """Find the optimal trend window based on pymannkendall statistical test.

        Args:
          train_labels (np.ndarray): years and values for a specific location.
          window_years (list): years to consider in a window
          extend_forward (bool): if true, extend trend window forward, else backward

        Returns:
          a list of years representing the optimal trend window
        """
        min_p = float("inf")
        opt_trend_years = None
        for i in range(
            self.MIN_TREND_WINDOW_SIZE,
            min(self.MAX_TREND_WINDOW_SIZE, len(window_years)) + 1,
        ):
            # should the search window be extended forward, i.e. towards later years
            if extend_forward:
                trend_x = window_years[:i]
            else:
                trend_x = window_years[-i:]

            trend_y = train_labels[np.in1d(train_labels[:, 0], trend_x)][:, 1]
            result = trend_mk.original_test(trend_y)

            # select based on p-value, lower the better
            if result.h and (result.p < min_p):
                min_p = result.p
                opt_trend_years = trend_x

        return opt_trend_years

    def _predict_trend(self, test_data: Iterable):
        """Predict the trend for each data item in test data.

        Args:
          test_data (Iterable): Dataset or a list of data items

        Returns:
          np.ndarray of predictions
        """
        trend_predictions = np.zeros((len(test_data), 1))
        for i, item in enumerate(test_data):
            loc = item[KEY_LOC]
            test_year = item[KEY_YEAR]

            sel_train_df = self._train_df[self._train_df[KEY_LOC] == loc]
            train_labels = sel_train_df[[KEY_YEAR, KEY_TARGET]].values
            train_years = sorted(sel_train_df[KEY_YEAR].unique())
            assert test_year not in train_years

            # Case 1: no training data for location
            if sel_train_df.empty:
                trend = self._train_df[KEY_TARGET].mean()
            else:
                lt_test_yr = [yr for yr in train_years if yr < test_year]
                gt_test_yr = [yr for yr in train_years if yr > test_year]

                # Case 2: Not enough years to estimate trend
                if (len(lt_test_yr) < self.MIN_TREND_WINDOW_SIZE) and (
                    len(gt_test_yr) < self.MIN_TREND_WINDOW_SIZE
                ):
                    trend = sel_train_df[KEY_TARGET].mean()
                else:
                    trend = None
                    # Case 3: Estimate trend using years before
                    if len(lt_test_yr) >= self.MIN_TREND_WINDOW_SIZE:
                        window_years = self._find_optimal_trend_window(
                            train_labels, lt_test_yr, extend_forward=False
                        )
                        if window_years is not None:
                            window_values = train_labels[
                                np.isin(train_labels[:, 0], window_years)
                            ][:, 1]
                            assert len(window_years) == len(window_values)
                            trend = self._estimate_trend(
                                window_years, window_values, test_year
                            )

                    # Case 4: Estimate trend using years after
                    if (trend is None) and (
                        len(gt_test_yr) >= self.MIN_TREND_WINDOW_SIZE
                    ):
                        window_years = self._find_optimal_trend_window(
                            train_labels, gt_test_yr, extend_forward=True
                        )
                        if window_years is not None:
                            window_values = train_labels[
                                np.isin(train_labels[:, 0], window_years)
                            ][:, 1]
                            assert len(window_years) == len(window_values)
                            trend = self._estimate_trend(
                                window_years, window_values, test_year
                            )

                    # Case 5: No significant trend exists
                    if trend is None:
                        trend = sel_train_df[KEY_TARGET].mean()

            trend_predictions[i, 0] = trend

        return trend_predictions

    def predict(self, dataset: Dataset, **predict_params):
        """Run fitted model on a test dataset.

        Args:
          dataset: Dataset
          **predict_params: Additional parameters

        Returns:
          A tuple containing a np.ndarray and a dict with additional information.
        """
        predictions = self._predict_trend(dataset)

        return predictions.flatten(), {}

    def predict_items(self, X: list, **predict_params):
        """Run fitted model on a list of data items.

        Args:
          X: a list of data items, each of which is a dict
          **predict_params: Additional parameters

        Returns:
          A tuple containing a np.ndarray and a dict with additional information.
        """
        predictions = self._predict_trend(X)

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
