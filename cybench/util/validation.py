from typing import Optional, List, Dict
from zoneinfo import available_timezones

import numpy as np
from omegaconf import DictConfig

from cybench.config import ValidationConfig


def get_splits(
        cfg: ValidationConfig,
        which: str,
        dataset_years: set,
        seed: int = 42,
):
    """
    Builds an iterator over test or val years based on a validation config file, see /conf/validation/...
    Yields (train_years, val_years) tuples

    Params:
        cfg: ValidationConfig
        which: either 'test' or 'val'
        dataset_years: the set of available years in the dataset to be split
        seed: random seed for all splitting requiring randomness

    Usage:
        for train, val in get_train_val_splits(cfg.validation, dataset.years):
            train_ds, val_ds = dataset.split_on_years((train, val))
    """
    # available years in the dataset
    dataset_years = sorted(dataset_years)
    split_years = cfg.test_years if which == 'test' else cfg.val_years

    #### 1. Step: Identify the set of hold-out years
    if isinstance(split_years, list):
        # test if year selection is available
        assert all([year in dataset_years for year in
                    split_years]), f"Selected test years ({split_years}) are not in dataset: {dataset_years}"
        # Explicit list of years provided
        hold_out_years = split_years

    elif isinstance(split_years, str):
        if split_years == 'loyocv':
            # Leave-one-year-out CV: all years become hold-out years
            hold_out_years = dataset_years

        elif split_years.endswith('-last'):
            # Take k last years (e.g., "3-last")
            k = int(split_years.split('-')[0])
            assert k <= len(dataset_years), f"Requested {k} last years but only {len(dataset_years)} available"
            hold_out_years = dataset_years[-k:]

        elif split_years.endswith('%-split'):
            # Random percentage split (e.g., "20%-split")
            percentage = int(split_years.split('%')[0])
            assert 0 < percentage < 100, f"Invalid percentage: {percentage}"
            n_hold_out = max(1, int(len(dataset_years) * percentage / 100))
            # Use numpy for reproducible random selection (could add seed from cfg)
            rng = np.random.RandomState(seed)
            hold_out_years = sorted(rng.choice(dataset_years, size=n_hold_out, replace=False))

        else:
            raise ValueError(f"Unknown split_years format: {split_years}")
    else:
        raise TypeError(f"split_years must be list or str, got {type(split_years)}")

    #### 2. Step: Select the training-set based on the split methode
    if cfg.name == 'single':
        # returning a single set of train- and hold-out-years
        train_years = [y for y in dataset_years if y not in hold_out_years]
        assert train_years, f"No train years left. Hold-out-years: {hold_out_years} | Available years: {dataset_years}"
        yield train_years, hold_out_years

    elif cfg.name == 'rolling':
        # returning a set of PAST train-years for each hold-out-year
        for hold_out_year in sorted(hold_out_years):
            train_years = [y for y in dataset_years if y < hold_out_year]
            assert train_years, f"No train years left. Hold-out-year: {hold_out_year} | Available years: {dataset_years}"
            yield train_years, [hold_out_year]

    elif cfg.name == 'loyocv':
        # returning a set of train-years for each hold-out-year
        for hold_out_year in sorted(hold_out_years):
            train_years = [y for y in dataset_years if y != hold_out_year]
            assert train_years, f"No train years left. Hold-out-year: {hold_out_year} | Available years: {dataset_years}"
            yield train_years, [hold_out_year]
    else:
        raise ValueError('Unknown split:', cfg.name)
