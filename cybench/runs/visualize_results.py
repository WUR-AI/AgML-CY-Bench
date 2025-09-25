import os
import argparse
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import re

from cybench.util.geo import get_shapes_from_polygons
from cybench.config import KEY_LOC, KEY_TARGET, REPO_DIR
from cybench.evaluation.eval import (
    evaluate_predictions,
    get_default_metrics,
    prepare_targets_preds,
)


def deduce_crop_country_from_filename(filepath):
    """Extract crop and country code from filename, e.g. maize_US_year_2018.csv"""
    filename = os.path.basename(filepath)
    match = re.match(r"([a-zA-Z]+)_([A-Z]{2})_", filename)
    if match:
        crop, country_code = match.groups()
    else:
        crop, country_code = "Unknown", "XX"
    return crop, country_code


def plot_results_summary(result_files, models, n_regions=5):
    # --- Load all results ---
    crop, country_code = deduce_crop_country_from_filename(result_files[0])
    all_dfs = [pd.read_csv(f) for f in result_files]
    df_all = pd.concat(all_dfs, ignore_index=True)

    results_dir = os.path.dirname(result_files[0])
    output_file = os.path.join(results_dir, f"{crop}_{country_code}.png")

    shapes = get_shapes_from_polygons(region=country_code)

    geo_df = shapes[[KEY_LOC, "geometry"]]
    geo_df = geo_df[geo_df[KEY_LOC].isin(df_all[KEY_LOC].unique())]
    merged = geo_df.merge(df_all, on=KEY_LOC, how="left")
    # Drop NaNs
    merged = merged.dropna(subset=[KEY_TARGET] + models)

    # Map bounds
    minx, miny, maxx, maxy = merged.total_bounds
    x_pad = (maxx - minx) * 0.05
    y_pad = (maxy - miny) * 0.05
    minx, maxx = minx - x_pad, maxx + x_pad
    miny, maxy = miny - y_pad, maxy + y_pad

    # World map background
    world_shapefile_path = os.path.join(
        REPO_DIR,
        "data_preparation",
        "ne_110m_admin_0_countries",
        "ne_110m_admin_0_countries.shp",
    )
    world = gpd.read_file(world_shapefile_path)

    years = sorted(merged["year"].unique())
    n_models = len(models)
    # 3 rows per year: map, scatter, residual scatter + 2 summary rows (scatter + residual)
    n_rows = len(years) * 3 + 2
    n_cols = max(1, n_models + 1)  # first column: ground truth

    fig, axes = plt.subplots(
        nrows=n_rows,
        ncols=n_cols,
        figsize=(6 * n_cols, 5 * n_rows),
        constrained_layout=True,
    )
    axes = np.atleast_2d(axes)

    # --- Select regions ---
    df_sorted = merged.sort_values(KEY_TARGET)
    n_total = len(df_sorted)
    quantiles = np.linspace(0.1, 0.9, min(n_regions, n_total))
    selected_indices = [int(q * n_total) for q in quantiles]
    selected_regions = df_sorted.iloc[selected_indices][KEY_LOC].tolist()
    region_colors = ["red", "magenta", "cyan", "orange", "purple", "green", "brown"][
        : len(selected_regions)
    ]

    # Scatter summary
    scatter_xmin = merged[KEY_TARGET].min()
    scatter_xmax = merged[KEY_TARGET].max()
    scatter_ymin = merged[models].min().min()
    scatter_ymax = merged[models].max().max()

    residuals_range = 6

    # --- Loop over years ---
    for y_idx, year in enumerate(years):
        df_year = merged[merged["year"] == year]
        row_map = y_idx * 3
        row_scatter = row_map + 1
        row_resid = row_map + 2

        # --- Maps ---
        ax = axes[row_map, 0]
        world.plot(ax=ax, color="lightgrey", edgecolor="black", linewidth=0.1)
        df_year.plot(
            column=KEY_TARGET, ax=ax, edgecolor="black", linewidth=0.2, legend=False
        )
        ax.set_xlim(minx, maxx)
        ax.set_ylim(miny, maxy)
        ax.axis("off")
        ax.set_title(f"Ground Truth {year}", fontsize=12)

        # Highlight selected regions
        for region, color in zip(selected_regions, region_colors):
            region_df = df_year[df_year[KEY_LOC] == region]
            if not region_df.empty:
                geom = region_df.geometry.values[0]
                if geom.geom_type == "Polygon":
                    ax.plot(*geom.exterior.xy, color=color, linewidth=2)
                elif geom.geom_type == "MultiPolygon":
                    for g in geom.geoms:
                        ax.plot(*g.exterior.xy, color=color, linewidth=2)

        # --- Model maps ---
        for i, model in enumerate(models):
            ax = axes[row_map, i + 1]
            world.plot(ax=ax, color="lightgrey", edgecolor="black", linewidth=0.1)
            if model in df_year.columns:
                df_year.plot(
                    column=model, ax=ax, edgecolor="black", linewidth=0.2, legend=False
                )
            ax.set_xlim(minx, maxx)
            ax.set_ylim(miny, maxy)
            ax.axis("off")
            ax.set_title(f"{model} {year}", fontsize=12)

            for region, color in zip(selected_regions, region_colors):
                region_df = df_year[df_year[KEY_LOC] == region]
                if not region_df.empty:
                    geom = region_df.geometry.values[0]
                    if geom.geom_type == "Polygon":
                        ax.plot(*geom.exterior.xy, color=color, linewidth=2)
                    elif geom.geom_type == "MultiPolygon":
                        for g in geom.geoms:
                            ax.plot(*g.exterior.xy, color=color, linewidth=2)

        for i in range(n_cols):
            ax = axes[row_scatter, i]
            if i == 0:
                ax.axis("off")
                continue

            model = models[i - 1]
            ax.hexbin(
                df_year[KEY_TARGET],
                df_year[model],
                gridsize=50,
                cmap="Blues",
                alpha=0.5,
            )

            # Highlight selected regions
            for region, color in zip(selected_regions, region_colors):
                region_df = df_year[df_year[KEY_LOC] == region]
                if not region_df.empty:
                    ax.scatter(
                        region_df[KEY_TARGET],
                        region_df[model],
                        color=color,
                        s=50,
                        label=f"{region} ({region_df[KEY_TARGET].values[0]:.1f})",
                    )

            ax.plot(
                [scatter_xmin, scatter_xmax],
                [scatter_xmin, scatter_xmax],
                "k--",
                alpha=0.7,
            )

            # Metrics
            y_true, y_pred = prepare_targets_preds(df_year, model)
            metrics = evaluate_predictions(
                y_true, y_pred, metrics=get_default_metrics()
            )
            ax.text(
                0.05,
                0.95,
                " ".join([f"{k}={v:.2f}" for k, v in metrics.items()]),
                # + "\n"
                # + " ".join([f"resid_{k}={v:.2f}" for k, v in resid_metrics.items()]),
                transform=ax.transAxes,
                verticalalignment="top",
                fontsize=12,
                bbox=dict(facecolor="white", alpha=0.5, edgecolor="none", pad=2),
            )
            ax.set_xlim(scatter_xmin, scatter_xmax)
            ax.set_ylim(scatter_ymin, scatter_ymax)
            ax.set_xlabel("Actual Yield")
            ax.set_ylabel(f"{model} Prediction")
            ax.set_title(f"{model} vs Actual")
            ax.legend(fontsize=12, loc="best")

        # --- Residual scatter plots ---

        train_years = [y for y in merged["year"].unique() if y != year]
        y_loc_mean = (
            merged[merged["year"].isin(train_years)].groupby(KEY_LOC)[KEY_TARGET].mean()
        )

        for i in range(n_cols):
            ax = axes[row_resid, i]
            if i == 0:
                ax.axis("off")
                continue

            model = models[i - 1]
            y_true_resid, y_pred_resid = prepare_targets_preds(
                df_year, model, y_loc_mean=y_loc_mean, residual=True
            )
            ax.hexbin(y_true_resid, y_pred_resid, gridsize=50, cmap="Reds", alpha=0.5)

            for region, color in zip(selected_regions, region_colors):
                region_df = df_year[df_year[KEY_LOC] == region]
                y_t_resid, y_p_resid = prepare_targets_preds(
                    region_df, model, y_loc_mean=y_loc_mean, residual=True
                )
                if len(y_t_resid) > 0:
                    ax.scatter(y_t_resid, y_p_resid, color=color, s=50)

            # Identity/reference lines
            ax.plot(
                [-residuals_range, residuals_range],
                [-residuals_range, residuals_range],
                "k--",
                alpha=0.7,
            )

            resid_metrics = evaluate_predictions(
                y_true_resid, y_pred_resid, metrics=get_default_metrics(residual=True)
            )
            ax.text(
                0.05,
                0.95,
                "\n".join([f"{k}={v:.2f}" for k, v in resid_metrics.items()]),
                transform=ax.transAxes,
                verticalalignment="top",
                fontsize=12,
                bbox=dict(facecolor="white", alpha=0.5, edgecolor="none", pad=2),
            )
            ax.set_xlim(-residuals_range, residuals_range)
            ax.set_ylim(-residuals_range, residuals_range)
            ax.set_xlabel("Residual Actual Yield")
            ax.set_ylabel(f"{model} Residual Prediction")
            ax.set_title(f"{model} Residuals")

    # --- Summary row ---
    row_summary = len(years) * 3
    row_resid_summary = row_summary + 1

    y_loc_mean = merged.groupby(KEY_LOC)[KEY_TARGET].mean()

    for i in range(n_cols):
        ax = axes[row_summary, i]
        if i == 0:
            ax.axis("off")
            continue

        model = models[i - 1]
        ax.hexbin(
            merged[KEY_TARGET],
            merged[model],
            gridsize=50,
            cmap="Blues",
            alpha=0.5,
        )

        for region, color in zip(selected_regions, region_colors):
            region_df = merged[merged[KEY_LOC] == region]
            if not region_df.empty:
                ax.scatter(
                    region_df[KEY_TARGET],
                    region_df[model],
                    color=color,
                    s=50,
                    label=f"{region} ({region_df[KEY_TARGET].values[0]:.1f})",
                )

        ax.plot(
            [scatter_xmin, scatter_xmax], [scatter_xmin, scatter_xmax], "k--", alpha=0.7
        )

        y_true, y_pred = prepare_targets_preds(merged, model)
        eval_metrics = evaluate_predictions(
            y_true, y_pred, metrics=get_default_metrics()
        )
        y_true_resid, y_pred_resid = prepare_targets_preds(
            merged, model, y_loc_mean=y_loc_mean, residual=True
        )
        resid_metrics = evaluate_predictions(
            y_true_resid, y_pred_resid, metrics=get_default_metrics(residual=True)
        )

        textstr = "\n".join([f"{k}={v:.2f}" for k, v in eval_metrics.items()])
        textstr += "\nResiduals:\n" + "\n".join(
            [f"{k}={v:.2f}" for k, v in resid_metrics.items()]
        )
        ax.text(
            0.05,
            0.95,
            textstr,
            transform=ax.transAxes,
            verticalalignment="top",
            fontsize=12,
            bbox=dict(facecolor="white", alpha=0.5, edgecolor="none", pad=2),
        )
        ax.set_xlim(scatter_xmin, scatter_xmax)
        ax.set_ylim(scatter_ymin, scatter_ymax)
        ax.set_xlabel("Actual Yield")
        ax.set_ylabel(f"{model} Prediction")
        ax.set_title(f"{model} vs Actual")
        ax.legend(fontsize=12, loc="best")

    # Residual summary
    for i in range(n_cols):
        ax = axes[row_resid_summary, i]
        if i == 0:
            ax.axis("off")
            continue

        model = models[i - 1]
        y_true_resid, y_pred_resid = prepare_targets_preds(
            merged, model, y_loc_mean=y_loc_mean, residual=True
        )
        ax.hexbin(y_true_resid, y_pred_resid, gridsize=50, cmap="Reds", alpha=0.5)
        ax.plot(
            [-residuals_range, residuals_range],
            [-residuals_range, residuals_range],
            "k--",
            alpha=0.7,
        )
        resid_metrics = evaluate_predictions(
            y_true_resid, y_pred_resid, metrics=get_default_metrics(residual=True)
        )
        ax.text(
            0.05,
            0.95,
            "\n".join([f"{k}={v:.2f}" for k, v in resid_metrics.items()]),
            transform=ax.transAxes,
            verticalalignment="top",
            fontsize=12,
            bbox=dict(facecolor="white", alpha=0.5, edgecolor="none", pad=2),
        )
        ax.set_xlim(-residuals_range, residuals_range)
        ax.set_ylim(-residuals_range, residuals_range)
        ax.set_xlabel("Residual Actual Yield")
        ax.set_ylabel(f"{model} Residual Prediction")
        ax.set_title(f"{model} Residuals Summary")

    # --- Save final PNG ---
    fig.savefig(output_file, dpi=75, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved summary figure: {output_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Yield maps + scatter summary")
    parser.add_argument(
        "-r",
        "--result",
        nargs="+",
        required=True,
        help="Result CSV files (one per year)",
    )
    parser.add_argument(
        "-m",
        "--models",
        nargs="+",
        default=["AverageYieldModel"],
        help="Model prediction columns",
    )
    parser.add_argument(
        "--n_regions",
        type=int,
        default=5,
        help="Number of highlighted regions",
    )
    args = parser.parse_args()

    plot_results_summary(args.result, args.models, n_regions=args.n_regions)
