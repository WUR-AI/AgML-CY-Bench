import numpy as np
import pandas as pd
from copy import deepcopy
from omegaconf import OmegaConf


class Normalizer:
    """
    Normalizer that unites parameters for normalizing features depending on their value distribution.
    Parameters can be given or fitted on the experiment data.
    Parameter configurations are structured by YAML files. See: conf/dataset/normalizer
    YAML Config structure:

    features:
      year:
        type: minmax
        params: null
      bulk_density:
        type: standard
        params: null

    Supports normalization types:
      - "minmax": maps [min, max] -> [-1, 1]
      - "standard" maps N(mu, sigma) to N(0,1)
      - "logsinh"
      - "none"
    """

    def __init__(self, norm_cfg: dict):
        self.name = norm_cfg["name"]
        self.feature_cfg = norm_cfg.features

    def _fit_feature(self, series: pd.Series, ftype: str):
        """Compute needed statistics depending on normalization type."""
        if ftype == "none":
            return {}

        if ftype == "minmax":
            return {
                "min": float(series.min()),
                "max": float(series.max())
            }

        if ftype == "standard":
            return {
                "mean": float(series.mean()),
                "std": float(series.std())
            }

        if ftype == "logsinh":
            # No parameters needed — log-sinh is reversible without fitting.
            return {}

        raise ValueError(f"Unknown normalization type: {ftype}")

    def _apply_feature(self, series: pd.Series, ftype: str, params: dict):
        """Apply normalization using already-fitted parameters."""
        if ftype == "none":
            return series

        if ftype == "minmax":
            min, max = params["min"], params["max"]
            range = max - min
            if range == 0:
                return series * 0.0
            return (series - max/2 - min/2) / (range / 2)

        if ftype == "standard":
            if params["std"] == 0:
                return series * 0.0
            return (series - params["mean"]) / params["std"]

        if ftype == "logsinh":
            # x ↦ arcsinh(x)  (safe transform for skewed positive variables)
            return np.arcsinh(series)

        raise ValueError(f"Unknown normalization type: {ftype}")

    def fit_normalize(self, dfs):
        """
        Fits parameters across all DataFrames and returns
        normalized copies of the DataFrames.
        """
        # Concatenate for global statistics
        for source_name, df in dfs.items():
            for feature, cfg in self.feature_cfg.items():
                ftype = cfg["type"]
                if ftype == "logsinh": # logsinh has no parameter
                    continue

                params = cfg["params"]
                if params: # parameter already set
                    continue
                if feature not in df.columns:
                    continue

                params = self._fit_feature(df[feature], ftype)
                self.feature_cfg[feature]["params"] = params
        return self.normalize(dfs)

    def normalize(self, dfs):
        ### WARNING: this is done in-place
        """
        Normalize using already-fitted parameters.
        Returns new list of DataFrames.
        """
        for source_name, df in dfs.items():
            for feature, cfg in self.feature_cfg.items():
                if feature not in df.columns:
                    continue
                ftype = cfg["type"]
                params = cfg.get("params", {})
                df[feature] = self._apply_feature(df[feature], ftype, params)
        return dfs

    def normalize_sequence(self, series: pd.Series):
        """
        Normalize sequence.
        Args:
            series: pd.Series sequence.

        Returns: normalized sequence.
        """
        assert series.name in self.feature_cfg.keys(), f"{series.name} not in normalizer feature keys: {self.feature_cfg.keys()}"
        cfg = self.feature_cfg[series.name]
        ftype = cfg["type"]
        params = cfg.get("params", {})
        return self._apply_feature(series, ftype, params)

    def to_omegaconf(self):
        """
        Produces an OmegaConf node corresponding to normalization.yaml:

        features:
          feature_name:
            type: ...
            params: ...

        This is suitable to write back into a YAML file.
        """
        return OmegaConf.create({"features": self.feature_cfg})