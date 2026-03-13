import os
import pandas as pd
import pytest
from hydra import compose, initialize
import copy
from omegaconf import open_dict

from cybench.datasets.dataset import Dataset
from cybench.datasets.data_factory import DataFactory
from cybench.config import (
    PATH_DATA_DIR,
    KEY_LOC,
    KEY_YEAR,
    KEY_TARGET,
    KEY_DATES,
    KEY_COMBINED_FEATURES,
    SOIL_PROPERTIES,
    LOCATION_PROPERTIES,
    METEO_INDICATORS,
    RS_FPAR,
    RS_NDVI,
    SOIL_MOISTURE_INDICATORS,
    CROP_CALENDAR_DATES,
    DatasetConfig,
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
def dataset(dataset_cfg):
    return DataFactory(dataset_cfg).build()


def test_dataset_item(dataset):
    assert isinstance(dataset[0], dict)
    expected_indices = [KEY_LOC, KEY_YEAR, KEY_DATES]
    expected_data = (
        SOIL_PROPERTIES + LOCATION_PROPERTIES + METEO_INDICATORS + [RS_FPAR, RS_NDVI]
    )
    expected_data += SOIL_MOISTURE_INDICATORS + CROP_CALENDAR_DATES + [KEY_TARGET]
    assert len(dataset[0]) == len(expected_indices + expected_data)
    assert set(dataset[0].keys()) == set(expected_indices + expected_data)


def test_split(dataset_cfg):
    data_path_county_features = os.path.join(PATH_DATA_DIR, "features", "maize", "US")
    train_csv = os.path.join(data_path_county_features, "grain_maize_US_train.csv")
    train_df = pd.read_csv(train_csv, index_col=[KEY_LOC, KEY_YEAR])
    train_yields = train_df[[KEY_TARGET]].copy()
    feature_cols = [c for c in train_df.columns if c != KEY_TARGET]
    train_features = train_df[feature_cols].copy()
    dataset_cv = Dataset(dataset_cfg, train_yields, {KEY_COMBINED_FEATURES: train_features})

    even_years = {x for x in dataset_cv.years if x % 2 == 0}
    odd_years = dataset_cv.years - even_years

    ds1, ds2 = dataset_cv.split_on_years((even_years, odd_years))
    assert ds1.years == even_years
    assert ds2.years == odd_years


def test_load(dataset_cfg):
    cfg1 = copy.deepcopy(dataset_cfg)
    with open_dict(cfg1):
        cfg1.country = "NL"
    ds1 = DataFactory(cfg1).build()

    cfg2 = copy.deepcopy(dataset_cfg)
    with open_dict(cfg2):
        cfg2.country = "ES"
    ds2 = DataFactory(cfg2).build()

    cfg3 = copy.deepcopy(dataset_cfg)
    with open_dict(cfg3):
        cfg3.country = ["NL", "ES"]
    ds3 = DataFactory(cfg3).build()

    assert len(ds3) == len(ds1) + len(ds2)


def test_memory_optimization(dataset_cfg):
    cfg = copy.deepcopy(dataset_cfg)
    with open_dict(cfg):
        cfg.country = "ES"
        cfg.use_memory_optimization = False
    dataset_no_optimization = DataFactory(cfg).build()
    df_y_no_optimization, dfs_x_no_optimization = dataset_no_optimization._df_y, dataset_no_optimization._dfs_x

    with open_dict(cfg):
        cfg.country = "ES"
        cfg.use_memory_optimization = True
    dataset_memory_optimized = DataFactory(cfg).build()
    df_y_memory_optimized, dfs_x_memory_optimized = dataset_memory_optimized._df_y, dataset_memory_optimized._dfs_x
    assert df_y_no_optimization.equals(df_y_memory_optimized)

    for key in dfs_x_no_optimization:
        assert dfs_x_no_optimization[key].equals(dfs_x_memory_optimized[key])
