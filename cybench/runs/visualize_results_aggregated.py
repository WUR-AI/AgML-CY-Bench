#!/usr/bin/env python3
import os
import re
import json
import argparse
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

from cybench.util.geo import get_shapes_from_polygons
from cybench.config import KEY_LOC, KEY_TARGET, REPO_DIR

# -----------------------------
# Configuration
# -----------------------------
CORE_FRACTION = 0.75
YEAR_COVERAGE_THRESHOLD = 0.50  # Skip years where < 50% of regions have valid data
BASELINE_MODEL = "AverageYieldModel"
NON_MODEL_COLS = {KEY_LOC, "year", KEY_TARGET, "country_code", "crop"}
MIN_REGIONS_THRESHOLD = 10  # Rows with fewer regions will be greyed out in the table

WORLD_SHP_PATH = os.path.join(
    REPO_DIR,
    "data_preparation",
    "ne_110m_admin_0_countries",
    "ne_110m_admin_0_countries.shp",
)


# -----------------------------
# Data Discovery & Loading
# -----------------------------
def discover_inputs(results_dir: str) -> Dict[str, List[str]]:
    """
    Scans results_dir for CSVs and groups them by crop_region key.
    """
    if not os.path.isdir(results_dir):
        raise FileNotFoundError(f"Directory not found: {results_dir}")

    groups: Dict[str, List[str]] = {}
    pat = re.compile(r"^([A-Za-z]+)_([A-Z]{2})(?:_.*)?\.csv$")

    for fn in sorted(os.listdir(results_dir)):
        if not fn.endswith(".csv"):
            continue

        match = pat.match(fn)
        if match:
            crop, region = match.groups()
            key = f"{crop}_{region}"
            groups.setdefault(key, []).append(os.path.join(results_dir, fn))

    return groups


def load_and_clean_data(
    csv_files: List[str], target_model_to_plot: str, min_years: int
) -> Tuple[Optional[pd.DataFrame], str]:
    """
    Loads CSVs and applies strict filtering:
    1. Enforce numeric types for Target and ALL Models.
    2. Skip YEARS where < 50% of the dataset's regions have valid data (for all columns).
    3. Drop remaining individual rows with missing data.
    """
    try:
        df = pd.concat([pd.read_csv(f) for f in csv_files], ignore_index=True)
    except Exception as e:
        return None, f"read_error: {e}"

    # 1. Check basic requirements
    required_fixed = [KEY_LOC, "year", KEY_TARGET]
    if not all(c in df.columns for c in required_fixed):
        return (
            None,
            f"missing_fixed_cols: {[c for c in required_fixed if c not in df.columns]}",
        )

    if target_model_to_plot not in df.columns:
        return None, f"target_model_missing: {target_model_to_plot}"

    # 2. Identify ALL Model Columns (Dynamic Scan)
    all_cols = set(df.columns)
    candidate_models = [c for c in all_cols if c not in NON_MODEL_COLS]

    # Check Columns: Target + All Models
    cols_to_check = [KEY_TARGET] + candidate_models

    # 3. Coerce to Numeric
    for c in cols_to_check:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    # ---------------------------------------------------------
    # 4. Filter Years based on Spatial Coverage
    # ---------------------------------------------------------

    # Calculate mask of valid rows (finite in ALL relevant cols)
    valid_mask = df[cols_to_check].notna().all(axis=1)

    # If a region is entirely NaN for all years, it shouldn't count towards the "total" we expect.
    valid_locations_universe = df.loc[valid_mask, KEY_LOC].unique()
    total_regions = len(valid_locations_universe)

    if total_regions > 0:
        # Count unique regions having valid data per year
        regions_per_year = df[valid_mask].groupby("year")[KEY_LOC].nunique()

        # Determine valid years
        min_required = np.ceil(YEAR_COVERAGE_THRESHOLD * total_regions)
        valid_years = regions_per_year[regions_per_year >= min_required].index

        # Filter the DataFrame to keep only valid years
        df = df[df["year"].isin(valid_years)].copy()
    else:
        # If no valid regions exist at all, return empty early
        return None, "no_valid_regions_found"

    # ---------------------------------------------------------
    # 5. Strict Row Filtering (Intersection of Validity)
    # ---------------------------------------------------------
    # Now that we only have "good" years, we still drop any
    # specific rows that might be missing data (e.g. a single region in a good year).

    n_before = len(df)
    df = df.dropna(subset=cols_to_check)
    n_after = len(df)

    if n_after == 0:
        return None, f"empty_after_strict_filter (dropped {n_before} rows)"

    if df["year"].nunique() < min_years:
        return None, f"too_few_years: {df['year'].nunique()}"

    return df, "ok"


# -----------------------------
# Math & Stats
# -----------------------------
def calc_r_r2(y_true, y_pred) -> Tuple[float, float]:
    if len(y_true) < 2:
        return np.nan, np.nan

    r = float(np.corrcoef(y_true, y_pred)[0, 1])

    ss_res = float(np.sum((y_true - y_pred) ** 2))
    ss_tot = float(np.sum((y_true - np.mean(y_true)) ** 2))
    r2 = np.nan if ss_tot == 0 else float(1.0 - ss_res / ss_tot)

    return r, r2


def get_metrics_dict(df, target_col, model_col):
    """Helper to calculate standard and residual metrics for a model."""
    y_true = df[target_col].values
    y_pred = df[model_col].values
    r, r2 = calc_r_r2(y_true, y_pred)

    # Residual stats (de-meaned by location)
    loc_means = df.groupby(KEY_LOC)[target_col].mean()
    y_true_res = df[target_col] - df[KEY_LOC].map(loc_means)
    y_pred_res = df[model_col] - df[KEY_LOC].map(loc_means)
    r_res, r2_res = calc_r_r2(y_true_res, y_pred_res)

    return {"r": r, "r2": r2, "r_res": r_res, "r2_res": r2_res}


# -----------------------------
# Reporting
# -----------------------------
def generate_markdown_table(stats_list: List[dict]) -> str:
    """Generates a Markdown table comparing Model vs Baseline, with grey text for small datasets."""

    # Header
    md = "| Dataset | N | r (Mod / Base) | R² (Mod / Base) | r_res (Mod / Base) | R²_res (Mod / Base) | r_time (Mod / Base) |\n"
    md += "| :--- | :---: | :---: | :---: | :---: | :---: | :---: |\n"

    for s in stats_list:
        d_name = s["dataset"]
        n_reg = s["n_regions"]
        mod = s["metrics_model"]
        base = s["metrics_baseline"]

        # Check threshold
        is_faint = n_reg < MIN_REGIONS_THRESHOLD

        # Format helpers
        def fmt(val):
            return f"{val:.2f}" if pd.notnull(val) else "-"

        def pair(v_mod, v_base):
            return f"{fmt(v_mod)} / {fmt(v_base)}"

        def style(text):
            # Apply grey color if below threshold
            if is_faint:
                return f'<span style="color:gray">{text}</span>'
            return text

        if base:
            # Full comparison
            row = (
                f"| {style(d_name)} "
                f"| {style(str(n_reg))} "
                f"| {style(pair(mod['r'], base['r']))} "
                f"| {style(pair(mod['r2'], base['r2']))} "
                f"| {style(pair(mod['r_res'], base['r_res']))} "
                f"| {style(pair(mod['r2_res'], base['r2_res']))} "
                f"| {style(pair(s['r_time_model'], s['r_time_base']))} |"
            )
        else:
            # Model only
            row = (
                f"| {style(d_name)} "
                f"| {style(str(n_reg))} "
                f"| {style(fmt(mod['r']))} "
                f"| {style(fmt(mod['r2']))} "
                f"| {style(fmt(mod['r_res']))} "
                f"| {style(fmt(mod['r2_res']))} "
                f"| {style(fmt(s['r_time_model']))} |"
            )
        md += row + "\n"

    return md


# -----------------------------
# Plotting
# -----------------------------
def process_dataset(
    dataset_key: str,
    df: pd.DataFrame,
    model: str,
    world: gpd.GeoDataFrame,
) -> Tuple[plt.Figure, dict]:

    # 1. Core Year Filtering
    years = sorted(df["year"].unique())
    core_years = years[1:-1]
    n_core = len(core_years)

    if n_core == 0:
        raise ValueError("Not enough years to define core years (need > 2).")

    coverage = df[df["year"].isin(core_years)].groupby(KEY_LOC)["year"].nunique()
    threshold = int(np.ceil(CORE_FRACTION * n_core))
    valid_locs = coverage[coverage >= threshold].index

    df_filtered = df[df[KEY_LOC].isin(valid_locs)].copy()

    # 2. Geometry Prep
    try:
        crop, region_code = dataset_key.split("_")[:2]
    except ValueError:
        crop, region_code = "Unknown", "XX"

    shapes = get_shapes_from_polygons(region=region_code)
    geo_df = shapes[[KEY_LOC, "geometry"]].merge(
        df_filtered.groupby(KEY_LOC)[[KEY_TARGET, model]].mean().reset_index(),
        on=KEY_LOC,
        how="inner",
    )

    if geo_df.empty:
        raise ValueError(f"No geometry matches found for regions in {dataset_key}.")

    bounds = geo_df.total_bounds
    pad_x = (bounds[2] - bounds[0]) * 0.05
    pad_y = (bounds[3] - bounds[1]) * 0.05
    bounds = (
        bounds[0] - pad_x,
        bounds[2] + pad_x,
        bounds[1] - pad_y,
        bounds[3] + pad_y,
    )

    # 3. Stats Calculation (Primary Model)
    metrics_model = get_metrics_dict(df_filtered, KEY_TARGET, model)

    # 3b. Stats Calculation (Baseline)
    has_baseline = BASELINE_MODEL in df_filtered.columns
    if has_baseline:
        metrics_base = get_metrics_dict(df_filtered, KEY_TARGET, BASELINE_MODEL)
    else:
        metrics_base = None

    # Temporal stats (Spatial Mean per Year)
    # Group for Target and Main Model
    ts = df_filtered.groupby("year")[[KEY_TARGET, model]].mean()

    # Join Baseline with a suffix to avoid collision if model == baseline
    if has_baseline:
        ts_base = df_filtered.groupby("year")[BASELINE_MODEL].mean()
        ts_base.name = f"{BASELINE_MODEL}_Base"
        ts = ts.join(ts_base)

    ts = ts.sort_index()

    r_time_model, _ = calc_r_r2(ts[KEY_TARGET], ts[model])

    # For baseline temporal correlation, we use the renamed column
    base_col_name = f"{BASELINE_MODEL}_Base"
    if has_baseline and base_col_name in ts.columns:
        r_time_base = calc_r_r2(ts[KEY_TARGET], ts[base_col_name])[0]
    else:
        r_time_base = np.nan

    n_samples = len(df_filtered)
    n_regions = df_filtered[KEY_LOC].nunique()
    n_years = df_filtered["year"].nunique()

    # Save stats dict
    stats = {
        "dataset": dataset_key,
        "n_samples": n_samples,
        "n_regions": n_regions,
        "n_years": n_years,
        "metrics_model": metrics_model,
        "metrics_baseline": metrics_base,
        "r_time_model": r_time_model,
        "r_time_base": r_time_base,
    }

    # 4. Plotting
    fig, axes = plt.subplots(1, 4, figsize=(26, 6.5), constrained_layout=True)
    fig.suptitle(f"{dataset_key} (Model: {model})", fontsize=16)

    # Map 1: GT
    ax = axes[0]
    world.plot(ax=ax, color="lightgrey", edgecolor="k", linewidth=0.1)
    geo_df.plot(column=KEY_TARGET, ax=ax, legend=True, legend_kwds={"shrink": 0.5})
    ax.set_xlim(bounds[0], bounds[1])
    ax.set_ylim(bounds[2], bounds[3])
    ax.set_title(f"Ground Truth (Mean)\n($N_{{reg}}={n_regions}$)")
    ax.axis("off")

    # Map 2: Pred
    ax = axes[1]
    world.plot(ax=ax, color="lightgrey", edgecolor="k", linewidth=0.1)
    geo_df.plot(column=model, ax=ax, legend=True, legend_kwds={"shrink": 0.5})
    ax.set_xlim(bounds[0], bounds[1])
    ax.set_ylim(bounds[2], bounds[3])
    ax.set_title(f"Prediction (Mean)\n({model})")
    ax.axis("off")

    # Scatter
    ax = axes[2]
    y_true = df_filtered[KEY_TARGET].values
    y_pred = df_filtered[model].values

    if len(y_true) > 500:
        ax.hexbin(y_true, y_pred, gridsize=50, cmap="Blues", mincnt=1)
    else:
        ax.scatter(y_true, y_pred, alpha=0.6, s=15, label="Model")

    lo, hi = min(y_true.min(), y_pred.min()), max(y_true.max(), y_pred.max())
    ax.plot([lo, hi], [lo, hi], "k--", alpha=0.5)

    ax.set_title("Scatter (All Region-Years)")
    ax.set_xlabel("Actual Yield")
    ax.set_ylabel("Predicted Yield")

    # Stats Text Box
    txt_lines = [f"$N={n_samples}$"]

    def fmt_line(label, m):
        return (
            f"**{label}**: $r={m['r']:.2f}, R^2={m['r2']:.2f}$ | "
            f"$r_{{res}}={m['r_res']:.2f}, R^2_{{res}}={m['r2_res']:.2f}$"
        )

    txt_lines.append(fmt_line("Model", metrics_model))

    if metrics_base:
        txt_lines.append(fmt_line("Base", metrics_base))
    else:
        txt_lines.append("(Baseline not found)")

    stats_text = "\n".join(txt_lines)

    ax.text(
        0.05,
        0.95,
        stats_text,
        transform=ax.transAxes,
        fontsize=10,
        verticalalignment="top",
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
    )

    # Time Series
    ax = axes[3]
    ax.plot(ts.index, ts[KEY_TARGET], "-o", linewidth=2, label="Actual")
    ax.plot(
        ts.index, ts[model], "--o", linewidth=2, label=f"Model ($r={r_time_model:.2f}$)"
    )

    if has_baseline:
        # We plot the explicitly renamed base column
        ax.plot(
            ts.index,
            ts[base_col_name],
            ":o",
            color="green",
            alpha=0.7,
            label=f"Base ($r={r_time_base:.2f}$)",
        )

    ax.set_title(f"Spatial Mean over Time ($N_{{yrs}}={n_years}$)")
    ax.legend()
    ax.grid(alpha=0.3)
    ax.set_xlabel("Year")

    return fig, stats


# -----------------------------
# Main Execution
# -----------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Generate 1x4 evaluation plots for crop datasets."
    )

    parser.add_argument(
        "--results_dir",
        required=True,
        help="Directory containing the CSV result files.",
    )
    parser.add_argument(
        "--datasets", nargs="*", default=None, help="Optional list of dataset keys."
    )
    parser.add_argument(
        "-m",
        "--model",
        default="LocationBiasCorrectedMaxPredictorModel",
        help="Model column name to PLOT.",
    )
    parser.add_argument(
        "--save_individual", action="store_true", help="Save individual PNG/JSON files."
    )
    parser.add_argument("--output_pdf", help="Custom path for combined PDF.")

    args = parser.parse_args()

    # 1. Discovery
    all_groups = discover_inputs(args.results_dir)

    if not all_groups:
        print(f"[INFO] No valid CSV files found in {args.results_dir}")
        return

    # 2. Filtering Logic
    if args.datasets:
        requested = set(args.datasets)
        datasets_to_run = {k: v for k, v in all_groups.items() if k in requested}
        if len(datasets_to_run) < len(requested):
            print(f"[WARN] Some requested datasets were not found.")
    else:
        datasets_to_run = all_groups

    if not datasets_to_run:
        print("[INFO] No datasets matched. Exiting.")
        return

    # 3. Setup Output
    if args.output_pdf:
        pdf_path = args.output_pdf
    else:
        pdf_path = os.path.join(args.results_dir, "evaluation_plots.pdf")

    print("[INFO] Loading world geometry...")
    world = gpd.read_file(WORLD_SHP_PATH)

    print(f"[INFO] Processing {len(datasets_to_run)} dataset(s). Output: {pdf_path}")
    os.makedirs(os.path.dirname(os.path.abspath(pdf_path)), exist_ok=True)

    # Accumulator for final table
    all_stats_list = []

    with PdfPages(pdf_path) as pdf:
        for key in sorted(datasets_to_run.keys()):
            files = datasets_to_run[key]
            print(f"--> {key}...", end=" ", flush=True)

            # Load
            df, msg = load_and_clean_data(files, args.model, min_years=3)
            if df is None:
                print(f"[SKIP] {msg}")
                continue

            # Process
            try:
                fig, stats = process_dataset(key, df, args.model, world)
                pdf.savefig(fig)

                # Append stats for table generation
                all_stats_list.append(stats)

                if args.save_individual:
                    fig.savefig(
                        os.path.join(args.results_dir, f"{key}_plot.png"),
                        dpi=100,
                        bbox_inches="tight",
                    )
                    with open(
                        os.path.join(args.results_dir, f"{key}_stats.json"), "w"
                    ) as f:
                        json.dump(stats, f, indent=2)

                plt.close(fig)
                print("[OK]")

            except Exception as e:
                print(f"[FAIL] {e}")
                import traceback

                traceback.print_exc()

    # 4. Generate and Print Markdown Table
    if all_stats_list:
        md_table = generate_markdown_table(all_stats_list)

        table_path = os.path.join(args.results_dir, "summary_table.md")
        with open(table_path, "w") as f:
            f.write(md_table)

        print(f"\n[DONE] Table saved to: {table_path}")
    else:
        print("\n[WARN] No stats collected. No table generated.")

    print("\n[DONE]")


if __name__ == "__main__":
    main()
