import os

import pandas as pd
import argparse

from cybench.config import (
    DATASETS,
    PATH_DATA_DIR,
    PATH_OUTPUT_DIR,
)

from cybench.datasets.dataset import Dataset


def dataset_summary(
    dataset_name: str = "maize_NL", min_year: int = 2003, max_year: int = 2024
) -> dict:
    """
    Output a summary of dataset.
    Args:
        dataset_name (str): The name of the dataset to load
        min_year (int): minimum year (soil moisture data starts from 2003)
        max_year (int): maximum year (some regions in AR have data points for 2024)

    Returns:
        pd.DataFrame with a summary for given dataset
    """
    dataset = Dataset.load(dataset_name)
    all_years = sorted(dataset.years)
    # using a list causes strange issues when reading the counts into pandas
    per_year_data_sizes = ""
    ds_min_year = None
    ds_max_year = None
    n_admin_regions = 0
    n_labels = 0
    for test_year in range(min_year, max_year + 1):
        train_years = [y for y in all_years if y != test_year]
        test_years = [test_year]
        if test_year not in all_years:
            per_year_data_sizes += " 0"
        else:
            if ds_min_year is None:
                ds_min_year = test_year

            if (ds_max_year is None) or (test_year > ds_max_year):
                ds_max_year = test_year

            _, test_dataset = dataset.split_on_years((train_years, test_years))
            n_test_yr_locs = len(test_dataset)
            n_labels += n_test_yr_locs
            if n_test_yr_locs > n_admin_regions:
                n_admin_regions = n_test_yr_locs

            per_year_data_sizes += " " + str(n_test_yr_locs)

    # remove leading space
    per_year_data_sizes = per_year_data_sizes.strip()
    min_max_year = str(ds_min_year) + "-" + str(ds_max_year)
    ds_summary = {
        dataset_name: [min_max_year, n_admin_regions, n_labels, per_year_data_sizes]
    }
    summary_cols = [
        "Min Year-Max Year",
        "Admin Regions Count",
        "Labels Count",
        "Labels Count Per Year",
    ]
    summary_df = pd.DataFrame.from_dict(
        ds_summary, columns=summary_cols, orient="index"
    )
    summary_df.reset_index(inplace=True)
    summary_df.rename(columns={"index": "Dataset"}, inplace=True)

    return summary_df


def run_benchmark_summary(output_file: str = None):
    if output_file is None:
        output_file = "dataset_summary.csv"

    summary_df = pd.DataFrame()
    for crop in DATASETS:
        for cn in DATASETS[crop]:
            if os.path.exists(os.path.join(PATH_DATA_DIR, crop, cn)):
                dataset_name = crop + "_" + cn
                ds_df = dataset_summary(dataset_name=dataset_name)
                summary_df = pd.concat([summary_df, ds_df], axis=0)
    summary_df.to_csv(os.path.join(PATH_OUTPUT_DIR, output_file), index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="benchmark_summary.py", description="Output benchmark dataset summary"
    )
    parser.add_argument("-o", "--output_file")
    args = parser.parse_args()
    output_file = args.output_file
    run_benchmark_summary(output_file)
