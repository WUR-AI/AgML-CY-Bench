import hydra
from hydra.core.config_store import ConfigStore
from hydra.utils import instantiate
import logging

from cybench.config import ExperimentConfig
from cybench.datasets.data_factory import DataFactory
from cybench.evaluation.eval import evaluate_predictions
from cybench.util.config_utils import adjust_model_cfg_to_dataset, set_seed, remove_search_keys
from cybench.util.optuna_hyper_opt import OptunaOptimizer
from cybench.util.store_and_cache import make_split_folder, save_preds, save_meta_dict
from cybench.util.validation import get_splits

# init logger
log = logging.getLogger(__name__)

# init config store to use custom config dataclass (see config.py)
conf_store = ConfigStore.instance()
conf_store.store(name="exp_config", node=ExperimentConfig)


@hydra.main(config_path="../conf", config_name="config", version_base=None)
def main(cfg: ExperimentConfig):
    #print("=== Final Composed Config ===")
    #print(OmegaConf.to_yaml(cfg))
    set_seed(cfg.experiment.seed)

    log.info("=== Create Datasets ===")
    dataset = DataFactory(cfg.dataset).build()
    cfg.model = adjust_model_cfg_to_dataset(cfg.model, dataset)

    # split data in train- and test-set based on the validation strategy
    for train_test_split in get_splits(cfg=cfg.validation,
                                       which="test",
                                       dataset_years=dataset.years,
                                       seed=cfg.experiment.seed
                                       ):
        log.info(f"== Split Test: {train_test_split[-1]} ==")
        train_dataset, test_dataset = dataset.split_on_years(years_split=train_test_split)
        # create a folder for each split
        split_path = make_split_folder(run_dir=hydra.core.hydra_config.HydraConfig.get().runtime.output_dir, split_name=train_test_split[-1])

        # check whether hyperparameter tuning is equipped:
        if "hp_search" in cfg:
            hp_optimizer = OptunaOptimizer(
                cfg=cfg,
                dataset=train_dataset,
                path=split_path,
                study_name="_".join(split_path.split("\\")[-2:])
            )
            model_cfg = hp_optimizer.optimize()
        else:
            # _search_ keys for hyperparameter tuning have to be removed before model instantiation
            model_cfg = remove_search_keys(cfg.model)

        log.info(f"Train final model(s)")
        for i in range(cfg.experiment.n_repetitions):
            # set new seed for each repetition
            seed = cfg.experiment.seed + i
            set_seed(seed)
            # create, fit final model and predict test
            model = instantiate(model_cfg)
            fit_info = model.fit(train_dataset, val_dataset=test_dataset)
            test_preds, pred_info = model.predict(test_dataset)

            # save preds, model, ...
            save_preds(path=split_path, dataset=test_dataset, preds=test_preds, seed=seed)
            model.save(path=split_path, seed=seed)
            save_meta_dict(path=split_path, dict={"fit_info": fit_info, "test_info": pred_info}, seed=seed)

            # evaluate
            eval_metric = evaluate_predictions(y_true=test_dataset.targets, y_pred=test_preds, cfg=cfg.evaluation)
            log.info(f"Split {train_test_split[-1]} (seed {seed}) finished with metrics: {eval_metric}")


if __name__ == "__main__":
    main()