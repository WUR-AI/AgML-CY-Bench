import pickle

import numpy as np
import pandas as pd
from omegaconf import OmegaConf
import hashlib
import json
import os

from cybench.datasets.dataset import Dataset
from cybench.config import DatasetConfig


def cfg_to_hash(cfg: DatasetConfig, add_str: bool = True):
    """
    Create a deterministic hash from a DatasetConfig, to use it as a keys e.g. in caching

    Args:
        cfg: The dataset configuration

    Returns:
        A hex string hash that uniquely identifies this configuration
    """
    # Convert OmegaConf to a regular dict/primitive structure
    config_dict = OmegaConf.to_container(cfg, resolve=True)

    # Convert to JSON string with sorted keys for deterministic ordering
    config_str = json.dumps(config_dict, sort_keys=True, default=str)

    # Create hash
    hash_obj = hashlib.sha256(config_str.encode('utf-8'))
    hash = hash_obj.hexdigest()
    if add_str is not None:
        hash = "_".join([cfg.name, cfg.framework, cfg.temporal.season.end_of_sequence, hash])
    return hash


def make_split_folder(run_dir: str, split_name) -> str:
    if type(split_name) == list:
        split_name = "_".join(str(x) for x in split_name)
    else:
        split_name = str(split_name)
    split_path = os.path.join(run_dir, split_name)
    os.makedirs(split_path, exist_ok=True)
    return split_path


def save_preds(
        path: str,
        dataset: Dataset,
        preds: np.ndarray[tuple[int], np.dtype[np.number]],
        seed: int
        ):
    """
    Save predictions to a csv file.
    Args:
        path:
        dataset: Dataset to get targets and indices from.
        preds: Predicted targets.
    Returns: Nada
    """
    dataset.indices.merge(
        pd.DataFrame({"targets": dataset.targets, "preds": preds}),
        left_index=True, right_index=True
    ).to_csv(os.path.join(path, f'preds_{seed}.csv'), index=False)
    return None


def save_meta_dict(path, dict, seed):
    """
    Save a dict of any metadata after training.
    Args:
        path:
        dict:

    Returns: Nichts
    """
    with open(f'meta_{seed}.pkl', 'wb') as handle: # TODO fix path
        pickle.dump(dict, handle, protocol=pickle.HIGHEST_PROTOCOL)
    return None
