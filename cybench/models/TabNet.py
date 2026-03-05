# cybench/models/tabnet_model.py
from pytorch_tabnet.tab_model import TabNetRegressor
from cybench.models.model import BaseModel
from cybench.datasets.dataset import Dataset
from cybench.util.data import data_to_pandas
from cybench.util.features import unpack_time_series, design_features
from collections.abc import Iterable
from cybench.config import (
    KEY_LOC,
    KEY_YEAR,
    KEY_TARGET,
    KEY_DATES,
    SOIL_PROPERTIES,
    LOCATION_PROPERTIES,
    TIME_SERIES_INPUTS,
)


class TabNetYieldModel(BaseModel):
    def __init__(
        self,
        n_d=16,
        n_a=16,
        n_steps=5,
        gamma=1.3,
        n_independent=2,
        n_shared=2,
        **kwargs,
    ):

        # Store parameters for later use in fit()
        self.tabnet_params = dict(
            n_d=n_d,
            n_a=n_a,
            n_steps=n_steps,
            gamma=gamma,
            n_independent=n_independent,
            n_shared=n_shared,
            **kwargs,
        )

        self.model = None

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

        if LOCATION_PROPERTIES:
            location_df = data_to_pandas(
                data_items, data_cols=[KEY_LOC] + LOCATION_PROPERTIES
            )
            location_df = location_df.drop_duplicates()
            dfs_x["location"] = location_df

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

        return features

    ######### Fit the model
    def fit(
        self,
        dataset,
        max_epochs: int = 5,
        batch_size: int = 512,
        patience: int = 10,
        **kwargs,
    ):

        train_features = self._design_features(dataset.crop, dataset)
        train_labels = data_to_pandas(
            dataset, data_cols=[KEY_LOC, KEY_YEAR, KEY_TARGET]
        )
        self._feature_cols = [
            ft for ft in train_features.columns if ft not in [KEY_LOC, KEY_YEAR]
        ]
        train_data = train_features.merge(train_labels, on=[KEY_LOC, KEY_YEAR])

        X_train = train_data[self._feature_cols].values
        y_train = train_data[KEY_TARGET].values.reshape(-1, 1)

        self.model = TabNetRegressor(**self.tabnet_params)

        self.model.fit(
            X_train,
            y_train,
            max_epochs=max_epochs,
            patience=patience,
            batch_size=batch_size,
            virtual_batch_size=batch_size // 4,
            num_workers=0,
            drop_last=False,
        )

        return self, {}

    def predict(
        self,
        dataset: Dataset,
        **kwargs,
    ):
        test_features = self._design_features(dataset.crop, dataset)
        test_labels = data_to_pandas(dataset, data_cols=[KEY_LOC, KEY_YEAR, KEY_TARGET])
        # Check features are the same for training and test data
        ft_cols = [ft for ft in test_features.columns if ft not in [KEY_LOC, KEY_YEAR]]
        missing_features = [ft for ft in self._feature_cols if ft not in ft_cols]
        for ft in missing_features:
            test_features[ft] = 0.0

        test_features = test_features[[KEY_LOC, KEY_YEAR] + self._feature_cols]
        test_data = test_features.merge(test_labels, on=[KEY_LOC, KEY_YEAR])

        X_test = test_data[self._feature_cols].values
        preds = self.model.predict(X_test).squeeze()
        return preds, {}

    ###### Require API
    def predict_items(self, X: list, **kwargs):
        raise NotImplementedError("Tabnet uses batched dataset prediction")

    def save(self, model_name: str):
        self.model.save_model(model_name)

    @classmethod
    def load(cls, model_name):
        obj = cls()
        obj.model = TabNetRegressor()
        obj.model.load_model(model_name)
        return obj
