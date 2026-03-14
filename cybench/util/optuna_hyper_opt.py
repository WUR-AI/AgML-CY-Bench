import logging
from pathlib import Path
from typing import Any, Union

import numpy as np
import optuna
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf

from cybench.util.config_utils import set_seed
from cybench.config import ValidationConfig, ExperimentConfig
from cybench.datasets.dataset import Dataset
from cybench.util.validation import get_splits
from cybench.util.config_utils import remove_search_keys

log = logging.getLogger(__name__)


class OptunaOptimizer:
    """
    Hyperparameter Optimizer using Optuna.

    This class handles the creation of an Optuna study, the definition of the
    objective function based on a Hydra configuration with recursive `_search_`
    keys, and the storage of the optimal configuration.

    Args:
        cfg (ExperimentConfig): Configuration containing hp_config (Optuna settings), val_cfg (for val-splitting) &
                                model_cfg (model plus `_search_` keys to be optimized)
        dataset (Dataset): The dataset instance to be used for optimization.
        path (str): Directory path where the `optimal_model.yaml` will be stored.
        study_name (str): Unique identifier for the Optuna study.
    """

    def __init__(
            self,
            cfg: ExperimentConfig,
            dataset: Dataset,
            path: str,
            study_name: str
    ):
        self.cfg = cfg
        self.hp_config = cfg.hp_search
        self.val_cfg = cfg.validation
        self.dataset = dataset
        self.path = Path(path)
        self.study_name = study_name

        # Ensure output directory exists
        self.path.mkdir(parents=True, exist_ok=True)

    def optimize(self) -> DictConfig:
        """
        Runs the Optuna optimization process.

        Returns:
            DictConfig: The best model configuration found, with `_search_` keys
                        resolved to their optimal values.
        """
        log.info(f"Starting Optuna optimization: {self.study_name}")

        # 1. Setup Optuna Storage and Sampler
        storage = self.hp_config.storage.url if self.hp_config.get("storage") else None

        if "sampler" in self.hp_config:
            sampler = instantiate(self.hp_config.sampler)
        else:
            sampler = optuna.samplers.TPESampler(seed=self.hp_config.seed)

        # 2. Create Study
        study = optuna.create_study(
            study_name=self.study_name,
            storage=storage,
            sampler=sampler,
            pruner=optuna.pruners.ThresholdPruner(upper=self.dataset.targets.var()),
            load_if_exists=self.hp_config.storage.get("load_if_exists", True),
            direction="minimize"  # Assuming loss minimization; make configurable if needed
        )

        # 3. Optimize
        study.optimize(
            self._objective,
            n_trials=self.hp_config.get("n_trials", 100),
            timeout=self.hp_config.get("timeout", None),
            n_jobs=self.hp_config.get("n_jobs", 1),
            show_progress_bar=self.hp_config.logging.get("show_progress_bar", True)
        )

        log.info(f"Optimization finished. Best trial: {study.best_trial.params}")

        # 4. Reconstruct and Save Best Config
        # We use a FixedTrial with the best params to reconstruct the full config structure
        best_trial = optuna.trial.FixedTrial(study.best_params)
        best_config = self.cfg.copy()

        # Resolve the search space with best values
        self._resolve_search_space(best_config, best_trial)
        best_model_config = best_config.model

        # Clean any remaining artifacts (though _resolve_search_space handles _search_ removal)
        best_model_config = remove_search_keys(best_model_config)

        # Save to disk
        output_file = self.path / "optimal_model.yaml"
        with open(output_file, "w") as f:
            OmegaConf.save(best_model_config, f)

        log.info(f"Optimal model config saved to: {output_file}")

        return best_model_config

    def _objective(self, trial: optuna.Trial) -> float:
        """
        The objective function for Optuna.
        1. Clones the base config.
        2. Samples parameters using the trial.
        3. Instantiates the model.
        4. Performs validation (CV or Hold-out).
        5. Returns the mean validation metric.
        """
        # Deep copy to avoid modifying the original config for other trials
        cfg_copy = self.cfg.copy()

        # Apply suggested parameters to the config
        self._resolve_search_space(cfg_copy, trial)
        trail_model_cfg = cfg_copy.model

        # Instantiate model with specific trial configuration
        trail_model_cfg = remove_search_keys(trail_model_cfg)

        val_metrics = []

        for i in range(self.hp_config.repetitions):
            set_seed(self.hp_config.seed + i)

            # Get validation splits (Using the 'val' set of the current dataset)
            splits = get_splits(
                cfg=self.val_cfg,
                which="val",
                dataset_years=self.dataset.years,
                seed=self.hp_config.seed
            )

            for train_years, val_years in splits:
                train_dataset, val_dataset = self.dataset.split_on_years((train_years, val_years))

                # Instantiate and fit
                model = instantiate(trail_model_cfg, verbose=False)
                # no val_dataset included yet, but early stopping could require one ;)
                model.fit(train_dataset)

                # Predict and Evaluate
                preds, _ = model.predict(val_dataset)
                assert preds.ndim == val_dataset.targets.ndim, f"The model output shape {preds.shape} does not match {val_dataset.shape}"
                val_metric = np.mean((val_dataset.targets - preds) ** 2)

                val_metrics.append(val_metric)
                log.info(f"Validation metric ({i}): {val_metric}")

                # Optional: Pruning based on intermediate folds
                trial.report(np.mean(val_metrics), step=len(val_metrics) * (i + 1))
                if trial.should_prune():
                    raise optuna.TrialPruned()

        return float(np.mean(val_metrics))

    def _resolve_search_space(
            self,
            cfg: Union[DictConfig, Any],
            trial: optuna.Trial,
            prefix: str = ""
    ) -> None:
        """
        Recursively traverses the config, finds `_search_` keys, samples values
        from Optuna, updates the config in-place, and removes the `_search_` key.
        """
        if isinstance(cfg, DictConfig):
            # Iterate over a copy of keys to allow modification
            for key in list(cfg.keys()):
                value = cfg[key]

                # Check if this node has a search definition
                if key == "_search_":
                    search_params = value

                    # Iterate over parameters defined in _search_
                    for param_name, param_details in search_params.items():
                        # Construct unique name for Optuna
                        full_param_name = f"{prefix}.{param_name}" if prefix else param_name

                        # Extract type and constraints
                        p_type = param_details["type"]
                        kwargs = {k: v for k, v in OmegaConf.to_container(param_details, resolve=True).items()
                                  if k != "type"}

                        suggestion = None

                        if p_type == "int":
                            suggestion = trial.suggest_int(full_param_name, **kwargs)
                        elif p_type == "float":
                            suggestion = trial.suggest_float(full_param_name, **kwargs)
                        elif p_type == "categorical":
                            # 'choices' is required for categorical
                            choices = kwargs.get("choices")
                            if not choices:
                                raise ValueError(f"Categorical param {full_param_name} missing 'choices'")
                            suggestion = trial.suggest_categorical(full_param_name, choices)
                        else:
                            raise ValueError(f"Unknown search type '{p_type}' for {full_param_name}.")

                        cfg[param_name] = suggestion

                    # Remove the _search_ key after processing
                    del cfg._search_

                elif isinstance(value, (DictConfig, dict)):
                    # Recurse
                    new_prefix = f"{prefix}.{key}" if prefix else key
                    self._resolve_search_space(value, trial, new_prefix)