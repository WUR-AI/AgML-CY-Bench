import pandas as pd
import numpy as np

from cybench.util.features import dekad_from_date, unpack_time_series
from cybench.util.data import data_to_pandas
from cybench.config import (
    KEY_LOC,
    KEY_YEAR,
    KEY_DATES,
    SPINUP_DAYS,
    FORECAST_LEAD_TIME,
    CROP_CALENDAR_DOYS,
    CROP_CALENDAR_DATES,
    TIME_SERIES_INPUTS,
    TIME_SERIES_PREDICTORS,
    TIME_SERIES_AGGREGATIONS,
)


def add_cutoff_days(df: pd.DataFrame, lead_time: str):
    """Add a column with cutoff days relative to end of season.

    Args:
        df (pd.DataFrame): time series data
        lead_time (str): lead_time option

    Returns:
        the same DataFrame with column added
    """
    # For lead_time, see FORECAST_LEAD_TIME in config.py.
    if "day" in lead_time:
        df["cutoff_days"] = int(lead_time.split("-")[0])
    else:
        assert "season" in lead_time
        if lead_time == "middle-of-season":
            df["cutoff_days"] = (df["season_length"] // 2).astype(int)
        elif lead_time == "quarter-of-season":
            df["cutoff_days"] = (df["season_length"] // 4).astype(int)
        else:
            raise Exception(f'Unrecognized lead time "{lead_time}"')

    return df


def compute_crop_season_window(df, min_year, max_year, lead_time=FORECAST_LEAD_TIME):
    """Compute crop season window used for forecasting.

    Args:
        df (pd.DataFrame): crop calendar data
        min_year (int): earliest year in target data
        max_year (int): latest year in target data
        lead_time (str): forecast lead time option

    Returns:
        the same DataFrame with crop season window information
    """
    df = df[[KEY_LOC] + CROP_CALENDAR_DOYS]

    df = df.astype({k: int for k in CROP_CALENDAR_DOYS})
    # Check for out-of-range values
    invalid_sos = df[(df["sos"] < 0) | (df["sos"] > 366)]
    assert invalid_sos.empty, "Error: Some 'sos' values are out of range [0, 366]!"
    df["sos"] = df["sos"].clip(1, 366)

    invalid_eos = df[(df["eos"] < 0) | (df["eos"] > 366)]
    assert invalid_eos.empty, "Error: Some 'eos' values are out of range [0, 366]!"

    df["eos"] = df["eos"].clip(1, 366)
    df = pd.concat(
        [df.assign(**{KEY_YEAR: yr}) for yr in range(min_year, max_year + 1)],
        ignore_index=True,
    )
    df["sos_date"] = pd.to_datetime(df[KEY_YEAR] * 1000 + df["sos"], format="%Y%j")
    df["eos_date"] = pd.to_datetime(df[KEY_YEAR] * 1000 + df["eos"], format="%Y%j")

    # Fix sos_date for cases where sos > eos.
    # For example, sos_date is 20011124 and eos_date is 20010615.
    # For 2001, select sos_date for the previous year because season started
    # in the previous year.
    df["sos_date"] = np.where(
        (df["sos"] > df["eos"]),
        df["sos_date"] + pd.offsets.DateOffset(years=-1),
        df["sos_date"],
    )

    df["season_length"] = (df["eos_date"] - df["sos_date"]).dt.days
    assert df[df["season_length"] > 366].empty

    df = add_cutoff_days(df, lead_time)
    df["cutoff_date"] = df["eos_date"] - pd.to_timedelta(df["cutoff_days"], unit="d")
    df["season_window_length"] = df["season_length"] + SPINUP_DAYS - df["cutoff_days"]

    # drop redundant information
    df.drop(columns=CROP_CALENDAR_DOYS + ["season_length", "cutoff_days"], inplace=True)

    return df


def ensure_same_categories_union(df_1, df_2, cat_key=KEY_LOC):
    """Ensures that cat_key has the same categories in both DataFrames, using the union of unique values."""

    # Convert cat_key to categorical in both DataFrames
    if cat_key in df_1.columns:
        df_1[cat_key] = df_1[cat_key].astype("category")
    if cat_key in df_2.columns:
        df_2[cat_key] = df_2[cat_key].astype("category")

    # Combine unique values
    if cat_key in df_1.columns and cat_key in df_2.columns:
        unique_values_df_1 = df_1[cat_key].unique()
        unique_values_df_2 = df_2[cat_key].unique()
        combined_categories = pd.Index(
            list(set(unique_values_df_1) | set(unique_values_df_2))
        )

        # Set categories in both DataFrames
        df_1[cat_key] = df_1[cat_key].cat.set_categories(combined_categories)
        df_2[cat_key] = df_2[cat_key].cat.set_categories(combined_categories)

    return df_1, df_2


def restore_category_to_string(df, cat_key=KEY_LOC, original_order=None):
    """Converts a categorical column back to string, restoring the original order if available."""
    if cat_key in df.columns and isinstance(df[cat_key].dtype, pd.CategoricalDtype):
        df[cat_key] = df[cat_key].astype(str)

        # Restore original order if available
        if original_order is not None:
            df[cat_key] = pd.Categorical(
                df[cat_key], categories=original_order, ordered=True
            )

    return df


def process_crop_seasons(
    locs,
    years,
    dates,
    crop_season_keys,
    sos_dates,
    eos_dates,
    cutoff_dates,
    season_window_lengths,
):
    """Processes crop season data efficiently using NumPy."""
    n_elements = len(locs)
    # Initialize arrays
    sos_values = np.full(n_elements, np.datetime64("NaT"), dtype="datetime64[ns]")
    eos_values = np.full(n_elements, np.datetime64("NaT"), dtype="datetime64[ns]")
    cutoff_values = np.full(n_elements, np.datetime64("NaT"), dtype="datetime64[ns]")
    season_window_values = np.zeros(n_elements, dtype=int)

    # Create crop indices based on (loc, year) mapping
    crop_indices = np.array(
        [crop_season_keys.get((loc, year), -1) for loc, year in zip(locs, years)]
    )
    valid_indices = crop_indices != -1  # Mask for valid crop seasons
    # Assign values using valid indices
    np.putmask(sos_values, valid_indices, sos_dates[crop_indices])
    np.putmask(eos_values, valid_indices, eos_dates[crop_indices])
    np.putmask(cutoff_values, valid_indices, cutoff_dates[crop_indices])
    np.putmask(season_window_values, valid_indices, season_window_lengths[crop_indices])

    del valid_indices, crop_indices

    # Adjust years if dates exceed eos_values
    beyond_eos = dates > eos_values
    next_years = years + 1

    # Compute next-year crop indices
    next_crop_indices = np.array(
        [
            crop_season_keys.get((loc, next_year), -1)
            for loc, next_year in zip(locs, next_years)
        ]
    )
    del next_years
    valid_next_years = next_crop_indices != -1  # Mask for valid next-year crops

    # Update values for the next season if beyond EOS
    np.putmask(sos_values, beyond_eos & valid_next_years, sos_dates[next_crop_indices])
    np.putmask(eos_values, beyond_eos & valid_next_years, eos_dates[next_crop_indices])
    np.putmask(
        cutoff_values, beyond_eos & valid_next_years, cutoff_dates[next_crop_indices]
    )
    np.putmask(
        season_window_values,
        beyond_eos & valid_next_years,
        season_window_lengths[next_crop_indices],
    )
    # Boolean mask for rows where the year should be updated
    update_year_mask = beyond_eos & valid_next_years
    years[update_year_mask] += 1  # Modifies the original DataFrame in-place

    del beyond_eos, valid_next_years, update_year_mask

    # Validate date ranges
    valid_date_ranges = (
        (dates - sos_values).astype("timedelta64[D]").astype(int) <= 366
    ) & ((eos_values - dates).astype("timedelta64[D]").astype(int) <= 366)

    # Check season window constraints
    season_length_valid = (
        (cutoff_values - dates).astype("timedelta64[D]").astype(int)
        <= season_window_values
    ) & ((cutoff_values - dates).astype("timedelta64[D]").astype(int) >= 0)

    keep_mask = valid_date_ranges & season_length_valid  # Initial mask

    return keep_mask, years, season_window_values


def align_to_crop_season_window_numpy(
    locs: np.ndarray,
    years: np.ndarray,
    dates: np.ndarray,
    crop_season_keys: dict,
    sos_dates: np.ndarray,
    eos_dates: np.ndarray,
    cutoff_dates: np.ndarray,
    season_window_lengths: np.ndarray,
):
    """
    Aligns time series data using NumPy arrays and full vectorization, with optimized memory.
    """
    keep_mask, years, season_window_length = process_crop_seasons(
        locs,
        years,
        dates,
        crop_season_keys,
        sos_dates,
        eos_dates,
        cutoff_dates,
        season_window_lengths,
    )

    df_minimal = pd.DataFrame(
        {
            KEY_LOC: locs,
            KEY_YEAR: years,
            "date": dates,
            "season_window_length": season_window_length,
        }
    )
    grouped = df_minimal.groupby([KEY_LOC, KEY_YEAR], observed=True).agg(
        {"date": ["min", "max"], "season_window_length": "first"}
    )
    grouped.columns = ["date_min", "date_max", "season_window_length"]
    invalid_seasons = (grouped["date_max"] - grouped["date_min"]).dt.days < grouped[
        "season_window_length"
    ]
    invalid_season_pairs = set(grouped.index[invalid_seasons])
    if invalid_season_pairs:
        grouped_indices = df_minimal.groupby([KEY_LOC, KEY_YEAR], observed=True).indices
        invalid_indices = np.concatenate(
            [grouped_indices[k] for k in invalid_season_pairs if k in grouped_indices]
        )
        keep_mask[invalid_indices] = False

    return keep_mask, years


def align_to_crop_season_window(df: pd.DataFrame, crop_season_df: pd.DataFrame):
    """Align time series data to crop season window (includes lead time and spinup).

    Args:
        df (pd.DataFrame): time series data
        crop_season_df (pd.DataFrame): crop season data
        lead_time (str): forecast lead time option

    Returns:
        the input DataFrame with data aligned to crop season and trimmed to lead time
    """
    select_cols = list(df.columns)

    # Merge with crop season data
    df = df.merge(
        crop_season_df[[KEY_LOC, KEY_YEAR] + CROP_CALENDAR_DATES],
        on=[KEY_LOC, KEY_YEAR],
    )

    # The next crop season starts right after current year's harvest.
    df[KEY_YEAR] = np.where(df["date"] > df["eos_date"], df[KEY_YEAR] + 1, df[KEY_YEAR])
    df.drop(columns=CROP_CALENDAR_DATES, inplace=True)

    # merge with crop season data again because we changed KEY_YEAR
    df = df.merge(crop_season_df, on=[KEY_LOC, KEY_YEAR])

    # Validate sos_date: date - sos_date should not be more than 366 days
    assert df[(df["date"] - df["sos_date"]).dt.days > 366].empty

    # Validate eos_date: eos_date - date should not be more than 366 days
    assert df[(df["eos_date"] - df["date"]).dt.days > 366].empty

    # Drop years with not enough data for a season
    # NOTE: We cannot filter with df.groupby(...)["date"].transform("count")
    # because ndvi and fpar don't have daily values.
    df["min_date"] = df.groupby([KEY_LOC, KEY_YEAR], observed=True)["date"].transform(
        "min"
    )
    df["max_date"] = df.groupby([KEY_LOC, KEY_YEAR], observed=True)["date"].transform(
        "max"
    )
    df = df[(df["max_date"] - df["min_date"]).dt.days >= df["season_window_length"]]
    # Keep season_window_length, i.e. season_length + SPINUP_DAYS - cutoff_days
    df = df[(df["cutoff_date"] - df["date"]).dt.days <= df["season_window_length"]]

    # Trim to lead time
    df = df[df["date"] <= df["cutoff_date"]]

    return df[select_cols]


def align_inputs_and_labels(df_y: pd.DataFrame, dfs_x: dict) -> tuple:
    """Align inputs and labels to have common indices (KEY_LOC, KEY_YEAR).
    NOTE: Input data returned may still contain more (KEY_LOC, KEY_YEAR)
    entries than label data. This is fine because the index of label data is
    used to access input data and not the other way round.

    Args:
        df_y (pd.DataFrame): target or label data
        dfs_x (dict): key is input source and value is pd.DataFrame

    Returns:
        the same DataFrame with dates aligned to crop season
    """
    # - Filter the label data based on presence within all feature data sets
    # - Filter feature data based on label data

    # Identify common locations and years
    index_y_selection = set(df_y.index.values)

    for df_x in dfs_x.values():
        if len(df_x.index.names) == 1:
            index_y_selection = {
                (loc_id, year)
                for loc_id, year in index_y_selection
                if loc_id in df_x.index
            }

        if len(df_x.index.names) == 2:
            index_y_selection = index_y_selection.intersection(set(df_x.index.values))

        if len(df_x.index.names) == 3:
            index_y_selection = index_y_selection.intersection(
                set([(loc_id, year) for loc_id, year, _ in df_x.index.values])
            )

    # Filter the labels
    df_y = df_y.loc[list(index_y_selection)]
    # check empty targets
    if df_y.empty:
        return df_y, {}

    # Filter input data by index_y_locations and index_y_years
    index_y_locations = set([loc_id for loc_id, _ in index_y_selection])
    index_y_years = set([year for _, year in index_y_selection])

    for x in dfs_x:
        df_x = dfs_x[x]
        if len(df_x.index.names) == 1:
            df_x = df_x.loc[list(index_y_locations)]

        if len(df_x.index.names) == 2:
            df_x = df_x.loc[list(index_y_selection)]

        if len(df_x.index.names) == 3:
            index_names = df_x.index.names
            df_x.reset_index(inplace=True)
            df_x = df_x[
                (df_x[KEY_YEAR] >= min(index_y_years))
                & (df_x[KEY_YEAR] <= max(index_y_years))
            ]
            df_x.set_index(index_names, inplace=True)

        dfs_x[x] = df_x

    return df_y, dfs_x


def interpolate_time_series_data(
    dfs: list, df_crop_season: pd.DataFrame, max_season_window_length: int
):
    """Add dates covering season window length and interpolate to fill in NAs.

    Args:
        dfs (list): time series DataFrames
        df_crop_season (pd.DataFrame): crop season
        max_season_window_length (int): maximum season window length

    Returns:
        pd.DataFrame with interpolated data
    """
    # combine time series data
    df_ts = pd.concat(dfs, join="outer", axis=1)

    # create a dataframe with all dates for every location and year
    df_all_dates = df_crop_season[["cutoff_date"]].copy()
    df_all_dates["date"] = df_all_dates.apply(
        lambda r: pd.date_range(
            end=r["cutoff_date"], periods=max_season_window_length, freq="D"
        ),
        axis=1,
    )
    df_all_dates = df_all_dates.explode("date").drop(columns=["cutoff_date"])
    df_all_dates.reset_index(inplace=True)
    df_all_dates.set_index([KEY_LOC, KEY_YEAR, "date"], inplace=True)

    # merge to get all dates in time series data
    df_ts = df_ts.merge(df_all_dates, how="outer", on=[KEY_LOC, KEY_YEAR, "date"])
    del df_all_dates

    # NOTE: interpolate fills data in forward direction.
    # TODO: example: cutoff-date: 2018-07-19 last ndvi: 2018-07-12;
    # it will then interpolate between 2018-07-12 and e.g. 2018-01-01 from a different location
    df_ts = df_ts.sort_index().interpolate(method="linear")
    # fill NAs in the front with 0.0
    df_ts.fillna(0.0, inplace=True)

    return df_ts


def interpolate_time_series_data_items(X: list, max_season_window_length: int):
    """Add dates covering season window length and interpolate to fill in NAs.

    Args:
        X (list): data samples
        max_season_window_length (int): maximum season window length

    Returns:
        pd.DataFrame with interpolated data
    """
    ts_inputs = []
    for x, ts_cols in TIME_SERIES_INPUTS.items():
        df = data_to_pandas(X, [KEY_LOC, KEY_YEAR, KEY_DATES] + ts_cols)
        df = unpack_time_series(df, ts_cols)
        df.set_index([KEY_LOC, KEY_YEAR, "date"], inplace=True)
        df = df.astype({k: "float" for k in ts_cols})
        ts_inputs.append(df)

    df_crop_season = data_to_pandas(X, [KEY_LOC, KEY_YEAR] + CROP_CALENDAR_DATES)
    df_crop_season.set_index([KEY_LOC, KEY_YEAR], inplace=True)
    df_ts = interpolate_time_series_data(
        ts_inputs, df_crop_season, max_season_window_length
    )

    return df_ts


def aggregate_time_series_data(df_ts: pd.DataFrame, aggregate_time_series_to: str):
    """Aggregate time series data to the specified resolution.

    Args:
        df_ts (pd.DataFrame): time series data in daily resolution
        aggregate_time_series_to (str): resolution of aggregated data

    Returns:
        pd.DataFrame with interpolated data
    """
    if aggregate_time_series_to not in ["week", "dekad"]:
        raise Exception(
            f"Unsupported time series aggregation resolution {aggregate_time_series_to}"
        )

    if "date" not in df_ts.columns:
        assert "date" in df_ts.index.names
        df_ts.reset_index(inplace=True)

    assert "date" in df_ts.columns
    if aggregate_time_series_to == "week":
        df_ts["week"] = df_ts["date"].dt.isocalendar().week
    else:
        df_ts["dekad"] = df_ts.apply(lambda r: dekad_from_date(r["date"]), axis=1)

    ts_aggrs = {k: TIME_SERIES_AGGREGATIONS[k] for k in TIME_SERIES_PREDICTORS}
    # Primarily to avoid losing the "date" column.
    ts_aggrs["date"] = "min"
    df_ts = (
        df_ts.groupby([KEY_LOC, KEY_YEAR, aggregate_time_series_to], observed=True)
        .agg(ts_aggrs)
        .reset_index()
    )
    df_ts.drop(columns=[aggregate_time_series_to], inplace=True)

    return df_ts
