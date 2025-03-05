import os
import pandas as pd
import argparse

from cybench.runs.run_benchmark import (
    compute_metrics,
    get_prediction_residuals,
)
from cybench.config import (
    KEY_COUNTRY,
    KEY_LOC,
    KEY_YEAR,
    PATH_OUTPUT_DIR,
    PATH_RESULTS_DIR,
)
from cybench.evaluation.eval import get_default_metrics


def results_to_metrics():
    df_all = pd.DataFrame()
    run_names = os.listdir(PATH_RESULTS_DIR)
    default_metrics = list(get_default_metrics())
    for run_name in run_names:
        crop = run_name.split("_")[0]
        df_run = compute_metrics(run_name)
        if df_run.empty:
            continue

        metrics = [m for m in default_metrics if m in df_run.columns]
        df_run.reset_index(inplace=True)
        df_run["crop"] = crop
        # NOTE: Mean Absolute Percentage Error (MAPE) is not converted to percentage.
        # This is because we follow the default from scikit-learn.
        # TODO: Remove this when MAPE is actually a percentage.
        df_run["mape"] = df_run["mape"] * 100
        df_run = df_run[["crop", KEY_COUNTRY, KEY_YEAR, "model"] + metrics]
        df_all = pd.concat([df_all, df_run], axis=0)

    return df_all


def results_to_residuals(model_names):
    df_all = pd.DataFrame()
    run_names = os.listdir(PATH_RESULTS_DIR)
    for run_name in run_names:
        crop = run_name.split("_")[0]
        df_run = get_prediction_residuals(run_name, model_names)
        df_run["crop"] = crop
        df_run.reset_index(inplace=True)
        residual_cols = [c for c in df_run.columns if "res" in c]
        df_run = df_run[["crop", KEY_COUNTRY, KEY_LOC, KEY_YEAR] + residual_cols]
        df_all = pd.concat([df_all, df_run], axis=0)

    return df_all


# Function to format rows with the minimum value in bold
def format_row(row, metric):
    if metric == "r2":
        highlight_value = row.max()
    else:
        highlight_value = row.min()

    return " ".join(
        [
            f"**{value:.2f}**" if value == highlight_value else f"{value:.2f}"
            for value in row
        ]
    )


# Construct the Markdown table
def df_to_markdown(df, formatted_df):
    # Define column headers
    headers = ["crop", "country"] + df.columns.tolist()

    # Construct table
    table = []
    table.append("| " + " | ".join(headers) + " |")
    table.append("| " + " | ".join(["---"] * len(headers)) + " |")

    for idx, formatted_row in formatted_df.items():
        crop, country = idx
        row_values = [crop, country] + formatted_row.split()
        table.append(f"| " + " | ".join(row_values) + " |")

    return "\n".join(table)


def write_results_to_table(output_file: str):
    df_metrics = results_to_metrics()
    default_metrics = get_default_metrics()
    metrics = [m for m in default_metrics if m in df_metrics.columns]
    df_metrics = df_metrics.groupby(["crop", KEY_COUNTRY, "model"], observed=True).agg(
        {m: "mean" for m in metrics}
    )
    crops = df_metrics.index.get_level_values("crop").unique()
    metrics = df_metrics.columns.unique()
    tables = {}
    for crop in crops:
        tables[crop] = {}
        crop_df = df_metrics[df_metrics.index.get_level_values("crop").isin([crop])]
        for metric in metrics:
            tables[crop][metric] = crop_df.reset_index().pivot_table(
                index=["crop", KEY_COUNTRY], columns="model", values=metric
            )

    # Open a file to write Markdown content
    with open(os.path.join(PATH_OUTPUT_DIR, output_file), "w") as file:
        for crop, metrics in tables.items():
            for metric, values in metrics.items():
                df = tables[crop][metric]
                # Apply the formatting function to each row
                df_formatted = df.apply(lambda r: format_row(r, metric), axis=1)
                # Create the Markdown table
                markdown_table = df_to_markdown(df, df_formatted)
                file.write(f"## {crop} {metric}\n\n")
                file.write(markdown_table + "\n\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="process_results.py", description="Output markdown tables with summary of metrics"
    )
    parser.add_argument("-o", "--output_file")
    args = parser.parse_args()
    output_file = "output_tables.md"
    if (args.output_file is not None):
        output_file = args.output_file

    write_results_to_table(output_file=output_file)
