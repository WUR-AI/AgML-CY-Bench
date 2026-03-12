from typing import Tuple, List

import numpy as np
import pandas as pd
import torch.utils.data

from cybench.datasets.dataset import BaseDataset


class TorchDataset(BaseDataset, torch.utils.data.Dataset):
    def __init__(
            self,
            aligned_tensors: Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
            doy_tensor: torch.Tensor,
            column_names: Tuple[list, list, list],
            indices: pd.DataFrame,
    ):
        """
        PyTorch Dataset wrapper for compatibility with torch DataLoader objects.
        Implements splitting by year.

        :param aligned_tensors: Triplet of (target, context, time_series) tensors
        :param doy_tensor: Tensor of (samples x ts_length) listing day of the year for each time-point in a sample
        :param column_names: Triplet of column names for the three aligned tensors
        :param indices: DataFrame with at least 'adm_id' and 'year' columns
        """
        self.y, self.x_context, self.x_ts = aligned_tensors
        self.doy = doy_tensor
        self.y_columns, self.x_context_columns, self.x_ts_columns = column_names
        self.indices = indices

        # Validate that indices has required columns
        if 'year' not in self.indices.columns:
            raise ValueError("indices DataFrame must contain 'year' column")

    def __len__(self) -> int:
        """Return the total number of samples in the dataset."""
        return len(self.y)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Get a sample from the dataset.

        :param index: Index of the sample to retrieve
        :return: Tuple of (target, context, time_series) tensors for the given index
        """
        return self.y[index], self.x_context[index], self.x_ts[index], self.doy[index]

    def split_on_years(
            self, years_split: Tuple[list, list]
    ) -> Tuple['TorchDataset', 'TorchDataset']:
        """
        Create two new datasets based on the provided split in years.
        !!NOTE!!: This is a memory intensive operation, because it's making two subsets by copying the original data.
        Future implementations might want to rely on more memory-efficiency, in case that becomes a bottleneck.

        :param years_split: Tuple of two lists, e.g., ([2012, 2014], [2015, 2017])
        :return: Tuple of two TorchDataset instances
        """
        years_set1, years_set2 = years_split

        # Create boolean masks for each split
        mask1 = self.indices['year'].isin(years_set1)
        mask2 = self.indices['year'].isin(years_set2)

        # Get integer indices for each subset
        indices1 = mask1[mask1].index.tolist()
        indices2 = mask2[mask2].index.tolist()

        # Create new datasets with sliced tensors
        dataset1 = TorchDataset(
            aligned_tensors=(
                self.y[indices1],
                self.x_context[indices1],
                self.x_ts[indices1]
            ),
            doy_tensor=self.doy[indices1],
            column_names=(
                self.y_columns,
                self.x_context_columns,
                self.x_ts_columns
            ),
            indices=self.indices.iloc[indices1].reset_index(drop=True)
        )
        dataset2 = TorchDataset(
            aligned_tensors=(
                self.y[indices2],
                self.x_context[indices2],
                self.x_ts[indices2]
            ),
            doy_tensor=self.doy[indices2],
            column_names=(
                self.y_columns,
                self.x_context_columns,
                self.x_ts_columns
            ),
            indices=self.indices.iloc[indices2].reset_index(drop=True)
        )
        return dataset1, dataset2

    @property
    def years(self) -> set:
        """
        Obtain a set containing all years occurring in the dataset
        """
        return set(self.indices.year.unique())

    @property
    def location_ids(self) -> set:
        """
        Obtain a set containing all location ids occurring in the dataset
        """
        return set(self.indices.adm_id.unique())

    @property
    def targets(self) -> np.ndarray[tuple[int], np.dtype[np.number]]:
        """
        Obtain a numpy array of targets or labels
        """
        return self.y.numpy().reshape(-1)
