import os
import pandas as pd

from cybench.datasets.dataset import Dataset
from cybench.datasets.configured import load_dfs
from cybench.config import (
    PATH_DATA_DIR,
    KEY_LOC,
    KEY_YEAR,
    KEY_TARGET,
    KEY_DATES,
    KEY_COMBINED_FEATURES,
    SOIL_PROPERTIES,
    METEO_INDICATORS,
    RS_FPAR,
    RS_NDVI,
    SOIL_MOISTURE_INDICATORS,
    CROP_CALENDAR_DATES,
)


dataset = Dataset.load("maize_NL")


def test_dataset_item():
    assert isinstance(dataset[0], dict)
    expected_indices = [KEY_LOC, KEY_YEAR, KEY_DATES]
    expected_data = SOIL_PROPERTIES + METEO_INDICATORS + [RS_FPAR, RS_NDVI]
    expected_data += SOIL_MOISTURE_INDICATORS + CROP_CALENDAR_DATES + [KEY_TARGET]
    assert len(dataset[0]) == len(expected_indices + expected_data)
    assert set(dataset[0].keys()) == set(expected_indices + expected_data)


def test_split():
    data_path_county_features = os.path.join(PATH_DATA_DIR, "features", "maize", "US")
    train_csv = os.path.join(data_path_county_features, "grain_maize_US_train.csv")
    train_df = pd.read_csv(train_csv, index_col=[KEY_LOC, KEY_YEAR])
    train_yields = train_df[[KEY_TARGET]].copy()
    feature_cols = [c for c in train_df.columns if c != KEY_TARGET]
    train_features = train_df[feature_cols].copy()
    dataset_cv = Dataset("maize", train_yields, {KEY_COMBINED_FEATURES: train_features})

    even_years = {x for x in dataset_cv.years if x % 2 == 0}
    odd_years = dataset_cv.years - even_years

    ds1, ds2 = dataset_cv.split_on_years((even_years, odd_years))
    assert ds1.years == even_years
    assert ds2.years == odd_years


def test_load():
    ds1 = Dataset.load("maize_NL")
    ds2 = Dataset.load("maize_ES")
    ds3 = Dataset.load("maize_NL_ES")
    assert len(ds3) == (len(ds1) + len(ds2))


def test_memory_optimization():
    df_y_default, dfs_x_default = load_dfs("maize", "ES", use_memory_optimization=False)
    df_y_memory_optimized, dfs_x_memory_optimized = load_dfs(
        "maize", "ES", use_memory_optimization=False
    )
    assert df_y_default.equals(df_y_memory_optimized)

    for key in dfs_x_default:
        assert dfs_x_default[key].equals(dfs_x_memory_optimized[key])
