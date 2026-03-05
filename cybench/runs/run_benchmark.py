import os
from collections import defaultdict

import pandas as pd
import torch
import argparse

from cybench.config import (
    DATASETS,
    PATH_DATA_DIR,
    PATH_RESULTS_DIR,
    KEY_COUNTRY,
    KEY_LOC,
    KEY_YEAR,
    KEY_TARGET,
)

from cybench.datasets.dataset import Dataset
from cybench.evaluation.eval import (
    evaluate_predictions,
    get_default_metrics,
    prepare_targets_preds,
)
from cybench.models.naive_models import AverageYieldModel, MaxPredictorModel
from cybench.models.trend_models import TrendModel
from cybench.models.sklearn_models import SklearnRidge, SklearnRandomForest
from cybench.models.xgboost_model import XGBoostModel
from cybench.models.nn_models import (
    BaselineLSTM,
    BaselineInceptionTime,
    BaselineTransformer,
)

from cybench.models.residual_models import (
    RidgeRes,
    RandomForestRes,
    XGBoostRes,
    LSTMRes,
    InceptionTimeRes,
    TransformerRes,
)


_BASELINE_MODEL_CONSTRUCTORS = {
    "AverageYieldModel": AverageYieldModel,
    "MaxPredictorModel": MaxPredictorModel,
    "LinearTrend": TrendModel,
    "SklearnRidge": SklearnRidge,
    "RidgeRes": RidgeRes,
    "SklearnRF": SklearnRandomForest,
    "RFRes": RandomForestRes,
    "XGBoostModel": XGBoostModel,
    "XGBoostRes": XGBoostRes,
    "LSTM": BaselineLSTM,
    "LSTMRes": LSTMRes,
    "InceptionTime": BaselineInceptionTime,
    "InceptionTimeRes": InceptionTimeRes,
    "Transformer": BaselineTransformer,
    "TransformerRes": TransformerRes,
}

BASELINE_MODELS = list(_BASELINE_MODEL_CONSTRUCTORS.keys())

BASELINE_MODELS = ["AverageYieldModel", "MaxPredictorModel"]

_BASELINE_MODEL_INIT_KWARGS = defaultdict(dict)

NN_MODELS_EPOCHS = 50
_BASELINE_MODEL_FIT_KWARGS = defaultdict(dict)
_BASELINE_MODEL_FIT_KWARGS["LSTM"] = {
    "epochs": NN_MODELS_EPOCHS,
    "device": "cuda" if torch.cuda.is_available() else "cpu",
}
_BASELINE_MODEL_FIT_KWARGS["LSTMRes"] = {
    "epochs": NN_MODELS_EPOCHS,
    "device": "cuda" if torch.cuda.is_available() else "cpu",
}
_BASELINE_MODEL_FIT_KWARGS["InceptionTime"] = {
    "epochs": NN_MODELS_EPOCHS,
    "device": "cuda" if torch.cuda.is_available() else "cpu",
}
_BASELINE_MODEL_FIT_KWARGS["InceptionTimeRes"] = {
    "epochs": NN_MODELS_EPOCHS,
    "device": "cuda" if torch.cuda.is_available() else "cpu",
}

_BASELINE_MODEL_FIT_KWARGS["Transformer"] = {
    "epochs": NN_MODELS_EPOCHS,
    "device": "cuda" if torch.cuda.is_available() else "cpu",
}

_BASELINE_MODEL_FIT_KWARGS["TransformerRes"] = {
    "epochs": NN_MODELS_EPOCHS,
    "device": "cuda" if torch.cuda.is_available() else "cpu",
}
_BASELINE_MODEL_INIT_KWARGS["MaxPredictorModel"] = {
    "feature_column": "twso",
}

_BASELINE_MODEL_FIT_KWARGS["MaxPredictorModel"] = {
    "feature_column": "twso",
}


def discover_datasets_from_disk(path_data_dir: str):
    """
    Discover datasets by checking for:
      PATH_DATA_DIR/<crop>/<region>/twso_<crop>_<region>.csv
    Returns list of dataset_name strings like "maize_FR".
    """
    found = []

    # iterate crops as folders under PATH_DATA_DIR
    if not os.path.isdir(path_data_dir):
        raise FileNotFoundError(
            f"PATH_DATA_DIR does not exist or is not a directory: {path_data_dir}"
        )

    for crop in sorted(os.listdir(path_data_dir)):
        crop_dir = os.path.join(path_data_dir, crop)
        if not os.path.isdir(crop_dir):
            continue

        # iterate regions as folders under crop
        for region in sorted(os.listdir(crop_dir)):
            region_dir = os.path.join(crop_dir, region)
            if not os.path.isdir(region_dir):
                continue

            twso_file = os.path.join(region_dir, f"twso_{crop}_{region}.csv")
            if os.path.exists(twso_file):
                found.append(f"{crop}_{region}")

    return found


def run_benchmark(
    run_name: str,
    model_name: str = None,
    model_constructor: callable = None,
    model_init_kwargs: dict = None,
    model_fit_kwargs: dict = None,
    baseline_models: list = None,
    dataset_name: str = "maize_NL",
    sel_years: list = None,
    nn_models_epochs: int = None,
) -> dict:
    """
    Run CY-Bench.
    Args:
        run_name (str): The name of the run. Will be used to store log files and model results
        model_name (str): The name of the model. Will be used to store log files and model results
        model_constructor (Callable): The constructor of the model. Will be used to construct the model
        model_init_kwargs (dict): The kwargs used when constructing the model.
        model_fit_kwargs (dict): The kwargs used to fit the model.
        baseline_models (list): A list of names of baseline models to run next to the provided model.
                                If unspecified, a default list of baseline models will be used.
        dataset_name (str): The name of the dataset to load
        sel_years (list): a list of years to run leave one year out (for tests)
        nn_models_epochs (int): Number of epochs to run for nn-models (for tests)

    Returns:
        a dictionary containing the results of the benchmark
    """
    baseline_models = baseline_models or BASELINE_MODELS
    assert all([name in BASELINE_MODELS for name in baseline_models])

    model_init_kwargs = model_init_kwargs or dict()
    model_fit_kwargs = model_fit_kwargs or dict()

    # Create a directory to store model output

    path_results = os.path.join(PATH_RESULTS_DIR, run_name)
    os.makedirs(path_results, exist_ok=True)

    # Make sure model_name is not already defined
    assert (
        model_name not in BASELINE_MODELS
    ), f"Model name {model_name} already occurs in the baseline"

    model_constructors = {k: _BASELINE_MODEL_CONSTRUCTORS[k] for k in baseline_models}

    models_init_kwargs = defaultdict(dict)
    for name in baseline_models:
        models_init_kwargs[name] = _BASELINE_MODEL_INIT_KWARGS[name]

    models_fit_kwargs = defaultdict(dict)
    for name in baseline_models:
        kwargs = _BASELINE_MODEL_FIT_KWARGS[name]
        # override epochs for nn-models (mainly for testing)
        if ("epochs" in kwargs) and (nn_models_epochs is not None):
            kwargs["epochs"] = nn_models_epochs

        models_fit_kwargs[name] = kwargs

    if model_name is not None:
        assert model_constructor is not None
        model_constructors[model_name] = model_constructor
        models_init_kwargs[model_name] = model_init_kwargs
        models_fit_kwargs[model_name] = model_fit_kwargs

    dataset = Dataset.load(dataset_name)
    all_years = sorted(dataset.years)
    if sel_years is not None:
        assert all([yr in all_years for yr in sel_years])
    else:
        sel_years = all_years

    all_results = []
    for test_year in sel_years:
        train_years = [y for y in all_years if y != test_year]
        test_years = [test_year]
        train_dataset, test_dataset = dataset.split_on_years((train_years, test_years))

        # TODO: put into generic function
        models_init_kwargs["Transformer"] = {
            "seq_len": train_dataset.max_season_window_length,
        }
        models_init_kwargs["TransformerRes"] = {
            "seq_len": train_dataset.max_season_window_length,
        }

        labels = test_dataset.targets()

        model_output = {
            KEY_LOC: [loc_id for loc_id, _ in test_dataset.indices()],
            KEY_YEAR: [year for _, year in test_dataset.indices()],
            KEY_TARGET: labels,
        }

        for model_name, model_constructor in model_constructors.items():
            model = model_constructor(**models_init_kwargs[model_name])
            model.fit(train_dataset, **models_fit_kwargs[model_name])
            predictions, _ = model.predict(test_dataset)
            model_output[model_name] = predictions

        df = pd.DataFrame.from_dict(model_output)
        df[KEY_COUNTRY] = df[KEY_LOC].str[:2]
        df.set_index([KEY_COUNTRY, KEY_LOC, KEY_YEAR], inplace=True)
        all_results.append(df)

    df_all = pd.concat(all_results).sort_index()

    results_file = os.path.join(path_results, f"{dataset_name}.csv")
    print(f"write results to {results_file}")
    df_all.to_csv(results_file)
    return


def load_results(
    run_name: str,
) -> pd.DataFrame:
    """
    Load saved results for analysis or visualization.
    Args:
        run_name (str): The name of the run. Will be used to store log files and model results

    Returns:
        a pd.DataFrame containing the predictions of benchmark models
    """
    path_results = os.path.join(PATH_RESULTS_DIR, run_name)

    files = [
        f
        for f in os.listdir(path_results)
        if os.path.isfile(os.path.join(path_results, f))
    ]

    # No files, return an empty data frame
    if not files:
        return pd.DataFrame(columns=[KEY_COUNTRY, KEY_LOC, KEY_YEAR, KEY_TARGET])

    df_all = pd.DataFrame()
    for file in files:
        if not file.lower().endswith(".csv"):
            continue
        path = os.path.join(path_results, file)
        df = pd.read_csv(path)
        df_all = pd.concat([df_all, df], axis=0)

    if KEY_COUNTRY not in df_all.columns:
        df_all[KEY_COUNTRY] = df_all[KEY_LOC].str[:2]

    return df_all


def get_prediction_residuals(run_name: str, model_names: dict) -> pd.DataFrame:
    """
    Get prediction residuals (i.e., model predictions - labels).
    Args:
        run_name (str): The name of the run. Will be used to store log files and model results
        model_names (dict): A mapping of model name (key) to a shorter name (value)

    Returns:
        a pd.DataFrame containing prediction residuals
    """
    df_all = load_results(run_name)
    if df_all.empty:
        return df_all

    for model_name, model_short_name in model_names.items():
        df_all[model_short_name + "_res"] = df_all[model_name] - df_all[KEY_TARGET]

    df_all.set_index([KEY_COUNTRY, KEY_LOC, KEY_YEAR], inplace=True)

    return df_all


def compute_metrics(
    run_name: str,
    model_names: list = None,
    residual: bool = False,
) -> pd.DataFrame:
    """
    Compute evaluation metrics on saved predictions.

    Args:
        run_name (str): The name of the run. Will be used to store log files and model results.
        model_names (list): Names of models to evaluate. If None, all model columns are used.
        residual (bool): If True, compute metrics on residuals (values adjusted per location).

    Returns:
        pd.DataFrame containing evaluation metrics
    """
    df_all = load_results(run_name)
    if df_all.empty:
        return pd.DataFrame(columns=[KEY_COUNTRY, "model", KEY_YEAR])

    rows = []
    country_codes = df_all[KEY_COUNTRY].unique()

    for cn in country_codes:
        df_cn = df_all[df_all[KEY_COUNTRY] == cn]
        all_years = sorted(df_cn[KEY_YEAR].unique())

        # Precompute location means for residuals
        y_loc_mean = df_cn.groupby(KEY_LOC)[KEY_TARGET].mean() if residual else None

        for yr in all_years:
            df_yr = df_cn[df_cn[KEY_YEAR] == yr]

            if model_names is None:
                model_names = [
                    c
                    for c in df_yr.columns
                    if c not in [KEY_COUNTRY, KEY_LOC, KEY_YEAR, KEY_TARGET]
                ]

            for model_name in model_names:
                y_true, y_pred = prepare_targets_preds(
                    df_yr, model_name, y_loc_mean, residual
                )

                # Select metrics based on residual mode
                metrics_to_use = get_default_metrics(residual=residual)
                metrics = evaluate_predictions(y_true, y_pred, metrics=metrics_to_use)

                metrics_row = {
                    KEY_COUNTRY: cn,
                    "model": model_name,
                    KEY_YEAR: yr,
                    **metrics,
                }
                rows.append(metrics_row)

    df_out = pd.DataFrame(rows)
    df_out.set_index([KEY_COUNTRY, "model", KEY_YEAR], inplace=True)

    return df_out


def run_benchmark_on_all_data():
    for crop in DATASETS:
        for cn in DATASETS[crop]:
            if os.path.exists(os.path.join(PATH_DATA_DIR, crop, cn)):
                dataset_name = crop + "_" + cn
                # NOTE: using dataset name for now.
                # Load results expects dataset name and run name to be the same.
                # TODO: Update this to handle multiple runs per dataset.
                # run_name = datetime.now().strftime(f"{dataset_name}_%H_%M_%d_%m_%Y.run")
                run_name = dataset_name
                run_benchmark(run_name=run_name, dataset_name=dataset_name)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog="run_benchmark.py", description="Run cybench")
    parser.add_argument("-r", "--run-name")
    parser.add_argument(
        "-d",
        "--dataset-name",
        default=None,
        help="Dataset name (e.g. maize_FR). If omitted or 'all'/'none', run all datasets.",
    )
    parser.add_argument("-m", "--mode")
    parser.add_argument(
        "-y",
        "--years",
        nargs="+",
        type=str,
        default=None,
        help="Test year(s)",
    )
    args = parser.parse_args()

    if args.dataset_name is None or str(args.dataset_name).lower() in ("none", "all"):
        dataset_names = discover_datasets_from_disk(PATH_DATA_DIR)
        if not dataset_names:
            raise FileNotFoundError(
                f"No datasets found. Expected files like "
                f"{PATH_DATA_DIR}/<crop>/<region>/twso_<crop>_<region>.csv"
            )
    else:
        dataset_names = [args.dataset_name]

    for dataset_name in dataset_names:
        run_name = args.run_name if args.run_name is not None else dataset_name

        if args.years is None or [y.lower() for y in args.years] in (["none"], ["all"]):
            args.years = None
        else:
            args.years = [int(y) for y in args.years]
        sel_years = args.years

        if (args.mode is not None) and args.mode == "test":
            # skipping some models
            baseline_models = [
                "AverageYieldModel",
                "LinearTrend",
                "SklearnRidge",
                "RidgeRes",
                "LSTM",
                "LSTMRes",
            ]
            # override epochs for nn-models
            nn_models_epochs = 5
            results = run_benchmark(
                run_name=run_name,
                dataset_name=dataset_name,
                baseline_models=baseline_models,
                nn_models_epochs=nn_models_epochs,
            )
        else:
            run_benchmark(
                run_name=run_name, dataset_name=dataset_name, sel_years=sel_years
            )
