import os
import torch
import pandas as pd
import numpy as np

from cybench.datasets.dataset import Dataset
from cybench.datasets.data_factory import DataFactory
from cybench.models.naive_models import AverageYieldModel
from cybench.models.trend_models import TrendModel
from cybench.models.sklearn_models import SklearnRidge
# from cybench.models.residual_models import RidgeRes
# from cybench.models.torch.nn_models import BaselineLSTM
from cybench.evaluation.eval import evaluate_predictions
import pytest
from hydra import compose, initialize
import copy
from omegaconf import open_dict

from cybench.config import PATH_DATA_DIR
from cybench.config import (
    KEY_LOC,
    KEY_YEAR,
    KEY_TARGET,
    KEY_COMBINED_FEATURES,
    DatasetConfig
)

@pytest.fixture
def dataset_cfg():
    with initialize(version_base=None, config_path="../../cybench/conf/dataset"):
        cfg = compose(
            config_name="default",
            overrides=[
                "crop=maize",
                "country=NL",
                "framework=sklearn",
            ],
        )
    return cfg

@pytest.fixture
def evaluation_cfg():
    with initialize(version_base=None, config_path="../../cybench/conf/evaluation"):
        cfg = compose(
            config_name="default",
        )
    return cfg


def test_average_yield_model():
    model = AverageYieldModel()
    dummy_data = [
        ["US-01-001", 2000, 5.0],
        ["US-01-001", 2001, 5.5],
        ["US-01-001", 2002, 6.0],
        ["US-01-001", 2003, 5.2],
        ["US-01-002", 2000, 7.0],
        ["US-01-002", 2001, 7.5],
        ["US-01-002", 2002, 6.2],
        ["US-01-002", 2003, 5.8],
    ]
    yield_df = pd.DataFrame(dummy_data, columns=[KEY_LOC, KEY_YEAR, KEY_TARGET])
    yield_df = yield_df.set_index([KEY_LOC, KEY_YEAR])

    # test prediction for an existing item
    sel_loc = "US-01-001"
    assert sel_loc in yield_df.index.get_level_values(0)
    dataset = Dataset(None, df_y=yield_df, dfs_x={})
    model.fit(dataset)
    sel_year = 2018
    filtered_df = yield_df[yield_df.index.get_level_values(0) == sel_loc]
    expected_pred = filtered_df[KEY_TARGET].mean()
    test_data = {
        KEY_LOC: sel_loc,
        KEY_YEAR: sel_year,
    }

    test_preds, _ = model.predict_items([test_data])
    assert np.round(test_preds[0], 2) == np.round(expected_pred, 2)

    # test one more location
    sel_loc = "US-01-002"
    test_data[KEY_LOC] = sel_loc
    filtered_df = yield_df[yield_df.index.get_level_values(0) == sel_loc]
    expected_pred = filtered_df[KEY_TARGET].mean()
    test_preds, _ = model.predict_items([test_data])
    assert np.round(test_preds[0], 2) == np.round(expected_pred, 2)

    # test prediction for a non-existent item
    sel_loc = "US-01-003"
    assert sel_loc not in yield_df.index.get_level_values(0)
    dataset = Dataset(None, df_y=yield_df, dfs_x={})
    model.fit(dataset)
    expected_pred = yield_df[KEY_TARGET].mean()
    test_data[KEY_LOC] = sel_loc
    test_preds, _ = model.predict_items([test_data])
    assert np.round(test_preds[0], 2) == np.round(expected_pred, 2)


def test_trend_model():
    dummy_data = [
        ["US-01-001", 2000, 4.1],
        ["US-01-001", 2001, 4.2],
        ["US-01-001", 2002, 4.3],
        ["US-01-001", 2003, 4.4],
        ["US-01-001", 2004, 4.5],
        ["US-01-001", 2005, 4.6],
        ["US-01-001", 2006, 4.7],
        ["US-01-001", 2007, 4.8],
        ["US-01-001", 2008, 4.9],
        ["US-01-001", 2009, 5.0],
        ["US-01-002", 2000, 5.1],
        ["US-01-002", 2001, 5.2],
        ["US-01-002", 2002, 5.3],
        ["US-01-002", 2003, 5.4],
        ["US-01-002", 2004, 5.5],
        ["US-01-002", 2005, 5.6],
        ["US-01-002", 2006, 5.7],
        ["US-01-002", 2007, 5.8],
        ["US-01-002", 2008, 5.9],
        ["US-01-002", 2009, 6.0],
        ["US-01-003", 2000, 7.0],
        ["US-01-003", 2001, 8.0],
        ["US-01-003", 2003, 9.0],
        ["US-01-004", 2000, 6.0],
        ["US-01-004", 2001, 6.1],
        ["US-01-004", 2003, 6.0],
        ["US-01-004", 2004, 6.1],
        ["US-01-004", 2005, 6.0],
        ["US-01-004", 2006, 6.1],
        ["US-01-004", 2007, 6.0],
        ["US-01-004", 2008, 6.1],
        ["US-01-004", 2009, 6.0],
    ]
    yield_df = pd.DataFrame(dummy_data, columns=[KEY_LOC, KEY_YEAR, KEY_TARGET])
    for sel_loc in yield_df[KEY_LOC].unique():
        yield_loc_df = yield_df[yield_df[KEY_LOC] == sel_loc]
        all_years = sorted(yield_loc_df[KEY_YEAR].unique())

        if sel_loc in ["US-01-001", "US-01-002"]:
            test_indexes = [0, 2, len(all_years) - 1]
            for idx in test_indexes:
                test_year = all_years[idx]
                train_years = [y for y in all_years if y != test_year]
                train_yields = yield_loc_df[yield_loc_df[KEY_YEAR].isin(train_years)]
                train_yields = train_yields.set_index([KEY_LOC, KEY_YEAR])
                test_yields = yield_loc_df[yield_loc_df[KEY_YEAR] == test_year]
                train_dataset = Dataset(None, df_y=train_yields, dfs_x={})

                model = TrendModel()
                model.fit(train_dataset)
                test_data = {
                    KEY_LOC: sel_loc,
                    KEY_YEAR: test_year,
                }
                test_preds, _ = model.predict_items([test_data])
                expected_pred = test_yields[KEY_TARGET].values[0]
                assert np.round(test_preds[0], 2) == np.round(expected_pred, 2)
        else:
            test_year = all_years[-1]
            train_years = [y for y in all_years if y != test_year]
            train_yields = yield_loc_df[yield_loc_df[KEY_YEAR].isin(train_years)]
            train_yields = train_yields.set_index([KEY_LOC, KEY_YEAR])
            train_dataset = Dataset(None, df_y=train_yields, dfs_x={})

            # Expect the average due to insufficient data or no trend
            model = TrendModel()
            model.fit(train_dataset)
            test_data = {
                KEY_LOC: sel_loc,
                KEY_YEAR: test_year,
            }
            test_preds, _ = model.predict_items([test_data])
            expected_pred = train_yields[KEY_TARGET].mean()
            assert np.round(test_preds[0], 2) == np.round(expected_pred, 2)


def test_sklearn_model(dataset_cfg, evaluation_cfg):
    # Test 1: Test with raw data
    cfg = copy.deepcopy(dataset_cfg)
    with open_dict(cfg):
        cfg.crop = "wheat"
    dataset_wheat = DataFactory(cfg).build()
    all_years = list(range(2001, 2019))
    test_years = [2017, 2018]
    train_years = [yr for yr in all_years if yr not in test_years]
    train_dataset, test_dataset = dataset_wheat.split_on_years(
        (train_years, test_years)
    )

    # Model
    model = SklearnRidge()
    model.fit(train_dataset)

    test_preds, _ = model.predict(test_dataset)
    assert test_preds.shape[0] == len(test_dataset)

    # Test 2: Test with predesigned features
    data_path = os.path.join(PATH_DATA_DIR, "features", "maize", "US")
    # Training dataset
    train_csv = os.path.join(data_path, "grain_maize_US_train.csv")
    train_df = pd.read_csv(train_csv, index_col=[KEY_LOC, KEY_YEAR])
    train_yields = train_df[[KEY_TARGET]].copy()
    feature_cols = [c for c in train_df.columns if c != KEY_TARGET]
    train_features = train_df[feature_cols].copy()
    cfg = copy.deepcopy(dataset_cfg)
    with open_dict(cfg):
        cfg.crop = "maize"
    train_dataset = Dataset(
        cfg, train_yields, {KEY_COMBINED_FEATURES: train_features}
    )

    # Test dataset
    test_csv = os.path.join(data_path, "grain_maize_US_train.csv")
    test_df = pd.read_csv(test_csv, index_col=[KEY_LOC, KEY_YEAR])
    test_yields = test_df[[KEY_TARGET]].copy()
    test_features = test_df[feature_cols].copy()
    cfg = copy.deepcopy(dataset_cfg)
    with open_dict(cfg):
        cfg.crop = "maize"
    test_dataset = Dataset(cfg, test_yields, {KEY_COMBINED_FEATURES: test_features})

    # Model
    model = SklearnRidge(
        feature_cols=feature_cols,
    )
    model.fit(train_dataset)

    test_preds, _ = model.predict(test_dataset)
    assert test_preds.shape[0] == len(test_dataset)

    # TODO: Need alternative to hardcoding expected metrics.
    targets = test_dataset.targets()
    evaluation_result = evaluate_predictions(targets, test_preds, evaluation_cfg)
    expected_values = {
        "normalized_rmse": [10.0, 20.0],
        "mape": [0.10, 0.20],
    }
    for metric, expected_value in expected_values.items():
        assert (
            metric in evaluation_result
        ), f"Metric '{metric}' not found in evaluation result"
        assert (
            round(evaluation_result[metric], 2) >= expected_value[0]
            and round(evaluation_result[metric], 2) <= expected_value[1]
        ), f"Value of metric '{metric}' does not match expected value"


# def test_sklearn_res_model():
#     # wheat NL
#     dataset_wheat = Dataset.load("wheat_NL")
#     all_years = list(range(2001, 2019))
#     test_years = [2017, 2018]
#     train_years = [yr for yr in all_years if yr not in test_years]
#     train_dataset, test_dataset = dataset_wheat.split_on_years(
#         (train_years, test_years)
#     )
#     ridge = SklearnRidge()
#     ridge_res = RidgeRes()
#     ridge.fit(train_dataset)
#     ridge_res.fit(train_dataset)
#
#     targets = test_dataset.targets()
#     ridge_preds, _ = ridge.predict(test_dataset)
#     ridge_res_preds, _ = ridge_res.predict(test_dataset)
#
#     metrics_ridge = evaluate_predictions(targets, ridge_preds)
#     metrics_ridge_res = evaluate_predictions(targets, ridge_res_preds)
#     print("wheat, NL")
#     print("SklearnRidge", metrics_ridge)
#     print("RidgeRes", metrics_ridge_res)
#
#     # maize NL
#     dataset_maize = Dataset.load("maize_NL")
#     all_years = list(range(2001, 2019))
#     test_years = [2017, 2018]
#     train_years = [yr for yr in all_years if yr not in test_years]
#     train_dataset, test_dataset = dataset_maize.split_on_years(
#         (train_years, test_years)
#     )
#     ridge = SklearnRidge()
#     ridge_res = RidgeRes()
#     ridge.fit(train_dataset)
#     ridge_res.fit(train_dataset)
#
#     targets = test_dataset.targets()
#     ridge_preds, _ = ridge.predict(test_dataset)
#     ridge_res_preds, _ = ridge_res.predict(test_dataset)
#
#     metrics_ridge = evaluate_predictions(targets, ridge_preds)
#     metrics_ridge_res = evaluate_predictions(targets, ridge_res_preds)
#     print("maize, NL")
#     print("SklearnRidge", metrics_ridge)
#     print("RidgeRes", metrics_ridge_res)


# TODO: Uncomment after TorchDataset is working.
# def test_nn_model():
#     train_dataset = Dataset.load("maize_NL")
#     test_dataset = Dataset.load("maize_NL")
#     device = "cuda" if torch.cuda.is_available() else "cpu"
#
#     # Initialize model, assumes that all features are in np.ndarray format
#     model = BaselineLSTM(
#         hidden_size=64,
#         num_layers=1,
#         output_size=1,
#     )
#     scheduler_fn = torch.optim.lr_scheduler.StepLR
#
#     # Train model
#     model.fit(
#         train_dataset,
#         batch_size=16,
#         epochs=10,
#         param_space={
#             "lr": [0.0001, 0.00001],
#             "weight_decay": [0.0001, 0.00001],
#         },
#         device=device,
#         scheduler_fn=scheduler_fn,
#         **{
#             "optimize_hyperparameters": True,
#             "validation_interval": 5,
#             "loss_kwargs": {
#                 "reduction": "mean",
#             },
#             "sched_kwargs": {
#                 "step_size": 2,
#                 "gamma": 0.5,
#             },
#         },
#     )
#
#     # Test predict_items()
#     num_test_items = len(test_dataset)
#     test_data = [test_dataset[i] for i in range(min(num_test_items, 16))]
#     test_preds, _ = model.predict_items(test_data)
#     assert test_preds.shape[0] == min(num_test_items, 16)
#
#     # Check if evaluation results are within expected range
#     test_preds, _ = model.predict(test_dataset)
#     targets = test_dataset.targets()
#     evaluation_result = evaluate_predictions(targets, test_preds)
#
#     min_expected_values = {
#         "normalized_rmse": 0,
#         "mape": 0.00,
#     }
#     for metric, expected_value in min_expected_values.items():
#         assert (
#             metric in evaluation_result
#         ), f"Metric '{metric}' not found in evaluation result"
#         assert (
#             evaluation_result[metric] >= expected_value
#         ), f"Value of metric '{metric}' does not match expected value"
#         # Check metric is not NaN
#         assert not np.isnan(
#             evaluation_result[metric]
#         ), f"Value of metric '{metric}' is NaN"
