# @author: Hilmy, Dilli

import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

from cybench.config import (
    CONFIG_DIR,
    KEY_YEAR,
    KEY_COUNTRY,
)

from cybench.runs.run_benchmark import BASELINE_MODELS
from cybench.runs.process_results import (
    results_to_metrics,
    results_to_residuals,
)

PATH_GRAPHS_DIR = os.path.join(CONFIG_DIR, "output", "graphs")
os.makedirs(PATH_GRAPHS_DIR, exist_ok=True)


def plot_bars(df, metric, metric_label, title_label, file_name):
    fig = plt.figure(figsize=(12, 6))
    sns.barplot(
        x=KEY_COUNTRY,
        y=metric,
        data=df,
        hue="model",
        palette="Set2",
        width=0.5,
        native_scale=True,
    )
    plt.title(title_label)
    plt.xlabel("Country")
    plt.ylabel(metric_label)
    plt.xticks()
    plt.tight_layout()
    plt.savefig(os.path.join(PATH_GRAPHS_DIR, file_name))


def plot_graph(
    df, x_col, hue_col, x_label, metric, metric_label, title, file_name, rotation=45
):
    fig = plt.figure(figsize=(12, 6))
    sns.boxplot(
        x=x_col,
        y=metric,
        data=df,
        hue=hue_col,
        palette="Set2",
        showfliers=False,
        legend=False,
    )
    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(metric_label)
    if rotation is not None:
        plt.xticks(rotation=rotation)
    plt.grid(True)
    plt.savefig(os.path.join(PATH_GRAPHS_DIR, file_name))
    plt.close(fig)


def plot_metrics(
    df: pd.DataFrame,
    metric: str = None,
):
    sns.set_palette(palette="pastel")

    if metric == "mape_country":
        plot_graph(
            df,
            KEY_COUNTRY,
            "model",
            "Country",
            "mape",
            "Mean Absolute Percentage Error",
            "MAPE Boxplots by country",
            "mape_country.jpg",
            rotation=45,
        )
    elif metric == "nrmse_country":
        plot_graph(
            df,
            KEY_COUNTRY,
            "model",
            "Country",
            "normalized_rmse",
            "Normalized RMSE",
            "NRMSE Boxplots by country",
            "nrmse_country.jpg",
            rotation=45,
        )
    elif metric == "bars":
        plot_bars(
            df,
            "mape",
            "Mean Absolute Percentage Error",
            "MAPE by country",
            "bars_mape_country.jpg",
        )

        plot_bars(
            df,
            "normalized_rmse",
            "Mean Absolute Percentage Error",
            "MAPE by country",
            "bars_nrmse_country.jpg",
        )


def box_plots_residuals(
    data,
    crop,
    countries,
    residual_cols,
    residual_labels,
    ymin,
    ymax,
    subplots_per_row=4,
):
    # Filter data based on selected countries and crop
    data_filtered = data[(data[KEY_COUNTRY].isin(countries)) & (data["crop"] == crop)]

    # Set up the figure for subplots
    num_countries = len(countries)
    num_rows = (num_countries + subplots_per_row - 1) // subplots_per_row
    fig, axes = plt.subplots(
        num_rows, subplots_per_row, figsize=(21, 4 * num_rows), sharey="none"
    )
    if isinstance(axes, np.ndarray):
        axes = axes.flatten()

    font = {
        "color": "black",
        "size": 18,
    }

    for i in range(num_countries):
        cn = countries[i]
        data_cn = data_filtered[data_filtered[KEY_COUNTRY] == cn]
        boxplot_df = data_cn[residual_cols]
        boxplot_df = boxplot_df.rename(columns=residual_labels)
        sel_ax = axes if (num_countries == 1) else axes[i]
        sel_ax.set_title(cn, fontdict=font)
        sel_ax.set_facecolor("w")
        sel_ax.set_ylim([ymin, ymax])
        sel_ax.tick_params(axis="both", which="major", labelsize=28)
        b = sns.boxplot(
            x="variable",
            y="value",
            data=pd.melt(boxplot_df),
            hue="variable",
            ax=sel_ax,
            whis=1.5,
            palette="tab10",
        )
        b.set_xlabel(" ")
        b.set_ylabel("Prediction Residuals", fontdict=font)
        if i > 0:
            b.set(yticklabels=[])
            b.set_ylabel(None)

    # Remove any empty subplots
    if isinstance(axes, np.ndarray):
        for j in range(i + 1, len(axes)):
            fig.delaxes(axes[j])

    plt.tight_layout()
    plt.savefig(os.path.join(PATH_GRAPHS_DIR, f"boxplots_residuals_{crop}.jpg"))
    plt.close(fig)


def box_plots_metrics(data, crop, countries, metric, metric_label, subplots_per_row=4):
    # Filter data based on selected countries and crop
    data_filtered = data[(data[KEY_COUNTRY].isin(countries)) & (data["crop"] == crop)]

    # Set up the figure for subplots
    num_countries = len(countries)
    num_rows = (num_countries + subplots_per_row - 1) // subplots_per_row
    fig, axes = plt.subplots(
        num_rows, subplots_per_row, figsize=(21, 4 * num_rows), sharey=True
    )
    if isinstance(axes, np.ndarray):
        axes = axes.flatten()

    # Plot boxplots for each country
    for i, country in enumerate(countries):
        ax = axes if (num_countries == 1) else axes[i]

        sns.boxplot(
            x="model",
            y=metric,
            data=data_filtered[data_filtered[KEY_COUNTRY] == country],
            ax=ax,
            showfliers=False,
            hue="model",
            palette="tab10",
        )
        ax.set_title(f"{country} ({crop})")
        ax.set_xlabel("")
        ax.set_ylabel(metric_label if i % subplots_per_row == 0 else "")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_visible(True)
        ax.spines["bottom"].set_visible(False)
        ax.yaxis.grid(True, linestyle="--", alpha=0.7)
        ax.margins(x=0.1)
        ax.xaxis.grid(False)

    # Remove any empty subplots
    if isinstance(axes, np.ndarray):
        for j in range(i + 1, len(axes)):
            fig.delaxes(axes[j])

    plt.tight_layout()
    plt.savefig(os.path.join(PATH_GRAPHS_DIR, f"boxplots_{metric}_{crop}.jpg"))
    plt.close(fig)


def plot_yearly_metrics(data, crop, country, metric, metric_label):
    data_filtered = data[(data["crop"] == crop) & (data[KEY_COUNTRY] == country)]
    all_years = sorted(data_filtered[KEY_YEAR].unique())
    num_rows = 2
    fig, axes = plt.subplots(num_rows, 1, figsize=(12, 5 * num_rows), sharey=True)
    if isinstance(axes, np.ndarray):
        axes = axes.flatten()

    num_years = len(all_years)
    sel_years = [all_years[: int(num_years / 2)], all_years[int(num_years / 2) :]]
    for i in range(2):
        ax = axes[i]
        sns.barplot(
            x="year",
            y=metric,
            data=data_filtered[data_filtered[KEY_YEAR].isin(sel_years[i])],
            ax=ax,
            hue="model",
            palette="Set2",
            width=0.5,
            native_scale=True,
        )
        ax.set_title(f"{country} ({crop})")
        ax.set_xlabel("Year")
        ax.set_ylabel(metric_label)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_visible(True)
        ax.spines["bottom"].set_visible(False)
        ax.yaxis.grid(True, linestyle="--", alpha=0.7)
        ax.margins(x=0.1)
        ax.xaxis.grid(False)
        if i != 0:
            ax.get_legend().remove()

    # Remove any empty subplots
    if isinstance(axes, np.ndarray):
        for j in range(i + 1, len(axes)):
            fig.delaxes(axes[j])

    plt.tight_layout()
    plt.savefig(os.path.join(PATH_GRAPHS_DIR, f"yearly_{metric}_{crop}_{country}.jpg"))
    plt.close(fig)


def plot_yearly_residuals(data, crop, country, residual_cols, residual_labels):
    # Filter data based on selected countries and crop
    data_filtered = data[(data[KEY_COUNTRY] == country) & (data["crop"] == crop)]
    all_years = sorted(data_filtered[KEY_YEAR].unique())
    num_years = len(all_years)
    num_cols = int(num_years / 2) + 1
    fig, axes = plt.subplots(2, num_cols, figsize=(5 * num_cols, 12), sharey="none")
    if isinstance(axes, np.ndarray):
        axes = axes.flatten()

    font = {
        "color": "black",
        "size": 18,
    }

    ymin = None
    ymax = None
    for c in residual_cols:
        if ymin is None:
            ymin = data_filtered[c].min()
        else:
            ymin = min(ymin, data_filtered[c].min())

        if ymax is None:
            ymax = data_filtered[c].max()
        else:
            ymax = max(ymax, data_filtered[c].max())

    for i, yr in enumerate(all_years):
        data_cn = data_filtered[data_filtered[KEY_YEAR] == yr]
        boxplot_df = data_cn[residual_cols]
        boxplot_df = boxplot_df.rename(columns=residual_labels)
        sel_ax = axes if (num_years == 1) else axes[i]
        sel_ax.set_title(yr, fontdict=font)
        sel_ax.set_facecolor("w")
        sel_ax.set_ylim([ymin, ymax])
        sel_ax.tick_params(axis="both", which="major", labelsize=28)
        b = sns.boxplot(
            x="variable",
            y="value",
            data=pd.melt(boxplot_df),
            hue="variable",
            ax=sel_ax,
            whis=1.5,
            palette="tab10",
        )
        b.set_xlabel(" ")
        b.set_ylabel("Prediction Residuals", fontdict=font)
        if i > 0:
            b.set(yticklabels=[])
            b.set_ylabel(None)

    # Remove any empty subplots
    if isinstance(axes, np.ndarray):
        for j in range(i + 1, len(axes)):
            fig.delaxes(axes[j])

    plt.tight_layout()
    plt.savefig(os.path.join(PATH_GRAPHS_DIR, f"yearly_residuals_{crop}_{country}.jpg"))
    plt.close(fig)


if __name__ == "__main__":
    metric_labels = {
        "mape": "Mean Absolute Percentage Error",
        "normalized_rmse": "Normalized RMSE",
        "r2": "Coefficient of Determination",
    }
    df_metrics = results_to_metrics()
    crops = df_metrics["crop"].unique()
    metric_names = [m for m in metric_labels if m in df_metrics.columns]
    for cr in crops:
        country_codes = df_metrics[df_metrics["crop"] == cr][KEY_COUNTRY].unique()
        for met in metric_names:
            box_plots_metrics(
                df_metrics,
                cr,
                country_codes,
                met,
                metric_labels[met],
                subplots_per_row=len(country_codes),
            )

        for met in metric_names:
            for cn in country_codes:
                plot_yearly_metrics(df_metrics, cr, cn, met, metric_labels[met])

    for metric in ["bars"]:
        plot_metrics(df_metrics, metric)

    model_short_names = {
        "AverageYieldModel": "Naive",
        "LinearTrend": "Trend",
        "SklearnRidge": "Ridge",
        "SklearnRF": "RF",
        "XGBoostModel": "XGB",
        "XGBoostRes": "XGB-Res",
    }
    # for other models, add name as short name
    for model_name in BASELINE_MODELS:
        if model_name not in model_short_names:
            model_short_names[model_name] = model_name

    df_residuals = results_to_residuals(model_names=model_short_names)
    residual_labels = {}
    for model_sname in model_short_names.values():
        residual_labels[model_sname + "_res"] = model_sname

    ymin = None
    ymax = None
    residual_cols = [c for c in df_residuals if "_res" in c]
    for c in residual_cols:
        if ymin is None:
            ymin = df_residuals[c].min()
        else:
            ymin = min(ymin, df_residuals[c].min())

        if ymax is None:
            ymax = df_residuals[c].max()
        else:
            ymax = max(ymax, df_residuals[c].max())

    crops = df_residuals["crop"].unique()
    for cr in crops:
        country_codes = df_residuals[df_residuals["crop"] == cr][KEY_COUNTRY].unique()
        box_plots_residuals(
            df_residuals,
            cr,
            country_codes,
            residual_cols,
            residual_labels,
            ymin,
            ymax,
            subplots_per_row=len(country_codes),
        )

        for met in metric_names:
            for cn in country_codes:
                plot_yearly_residuals(
                    df_residuals, cr, cn, residual_cols, residual_labels
                )
