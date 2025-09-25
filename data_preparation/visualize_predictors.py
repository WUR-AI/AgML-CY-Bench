import os
import argparse
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import Normalize
import numpy as np

from cybench.util.geo import get_shapes_from_polygons
from cybench.config import KEY_LOC, CROP_YIELD_RANGES, REPO_DIR


def visualize_yield_timeseries(
    dataset, timeseries, years=None, n_regions=5, output_dir="frames"
):
    # --- Determine crop and country from dataset ---
    try:
        crop, country_code = dataset.split("_")
    except ValueError:
        raise ValueError(
            f"Dataset name should be in format '<crop>_<country>', got {dataset}"
        )

    # --- Construct yield file path ---
    yield_file = os.path.join(
        REPO_DIR,
        "cybench",
        "data",
        crop,
        country_code,
        f"yield_{crop}_{country_code}.csv",
    )
    if not os.path.exists(yield_file):
        raise FileNotFoundError(f"Yield file not found: {yield_file}")

    df_yield = pd.read_csv(yield_file)
    if years:
        df_yield = df_yield[df_yield["harvest_year"].isin([int(y) for y in years])]
    years_sorted = sorted(df_yield["harvest_year"].unique())

    # --- Load shapefile ---
    shapes = get_shapes_from_polygons(region=country_code)
    geo_df = shapes[[KEY_LOC, "geometry"]]
    adm_ids_in_data = df_yield[KEY_LOC].unique()
    geo_df = geo_df[geo_df[KEY_LOC].isin(adm_ids_in_data)]
    merged = geo_df.merge(df_yield, on=KEY_LOC, how="left")

    # --- Map bounds ---
    minx, miny, maxx, maxy = merged.total_bounds
    x_pad = (maxx - minx) * 0.05
    y_pad = (maxy - miny) * 0.05
    minx, maxx = minx - x_pad, maxx + x_pad
    miny, maxy = miny - y_pad, maxy + y_pad

    # --- World map ---
    world_shapefile_path = os.path.join(
        REPO_DIR,
        "data_preparation",
        "ne_110m_admin_0_countries",
        "ne_110m_admin_0_countries.shp",
    )
    world = gpd.read_file(world_shapefile_path)

    # --- Load timeseries ---
    ts_dfs = {}
    data_dir = os.path.join(REPO_DIR, "cybench", "data", crop, country_code)
    for ts_name in timeseries:
        ts_file = os.path.join(data_dir, f"{ts_name}_{crop}_{country_code}.csv")
        if os.path.exists(ts_file):
            ts_df = pd.read_csv(ts_file)
            ts_df["date"] = pd.to_datetime(ts_df["date"], format="%Y%m%d")
            ts_dfs[ts_name] = ts_df
        else:
            print(f"Warning: {ts_file} not found, skipping")

    # --- Select regions ---
    # Keep only regions that appear in ALL selected years
    regions_per_year = [
        set(df_yield[df_yield["harvest_year"] == y][KEY_LOC].unique())
        for y in years_sorted
    ]
    valid_regions = set.intersection(*regions_per_year) if regions_per_year else set()

    if not valid_regions:
        raise ValueError("No regions have yield data in all selected years.")

    df_sorted = df_yield[df_yield[KEY_LOC].isin(valid_regions)].sort_values("yield")
    n_total = len(df_sorted[KEY_LOC].unique())

    # Spread quantiles across unique regions
    quantiles = np.linspace(0.1, 0.9, n_regions)
    unique_regions = df_sorted[KEY_LOC].unique()
    selected_indices = [int(q * n_total) for q in quantiles]
    selected_regions = [
        unique_regions[idx] for idx in selected_indices if idx < n_total
    ]

    region_colors = ["red", "magenta", "cyan", "orange", "purple", "green", "brown"][
        : len(selected_regions)
    ]

    # --- Fixed timeseries limits ---
    ts_limits = {}
    for name, ts_df in ts_dfs.items():
        ts_limits[name] = (ts_df[name].min(), ts_df[name].max())

    os.makedirs(output_dir, exist_ok=True)
    ncols = len(years_sorted)
    nrows = 1 + len(timeseries)  # top: yield, rows: timeseries

    fig, axes = plt.subplots(
        nrows=nrows,
        ncols=ncols,
        figsize=(6 * ncols, 5 * nrows),
        constrained_layout=True,
    )
    axes = np.atleast_2d(axes)  # ensures axes[row, col] works

    # --- Top row: yield maps ---
    vmin, vmax = CROP_YIELD_RANGES.get(crop, {}).get(
        "min", df_yield["yield"].min()
    ), CROP_YIELD_RANGES.get(crop, {}).get("max", df_yield["yield"].max())
    norm = Normalize(vmin=vmin, vmax=vmax)
    cmap = cm.viridis

    for i, year in enumerate(years_sorted):
        ax = axes[0, i]
        df_year = merged[merged["harvest_year"] == year]
        world.plot(ax=ax, color="lightgrey", edgecolor="black", linewidth=0.1)
        df_year.plot(
            column="yield",
            ax=ax,
            cmap=cmap,
            edgecolor="black",
            linewidth=0.2,
            legend=False,
            vmin=norm.vmin,
            vmax=norm.vmax,
        )
        ax.set_xlim(minx, maxx)
        ax.set_ylim(miny, maxy)
        ax.axis("off")
        ax.set_title(f"Yield {year}", fontsize=12)

        # Highlight selected regions
        for region, color in zip(selected_regions, region_colors):
            region_df = df_year[df_year[KEY_LOC] == region]
            if not region_df.empty:
                geom = region_df.geometry.values[0]
                if geom.geom_type == "Polygon":
                    ax.plot(*geom.exterior.xy, color=color, linewidth=1)
                elif geom.geom_type == "MultiPolygon":
                    for g in geom.geoms:
                        ax.plot(*g.exterior.xy, color=color, linewidth=1)

    # --- Timeseries rows ---
    for r, ts_name in enumerate(timeseries):
        for i, year in enumerate(years_sorted):
            ax = axes[r + 1, i]
            ts_df = ts_dfs[ts_name]
            ts_year = ts_df[ts_df["date"].dt.year == year]

            for region, color in zip(selected_regions, region_colors):
                region_ts = ts_year[ts_year[KEY_LOC] == region]
                if not region_ts.empty:
                    months = (
                        region_ts["date"].dt.month
                        + (region_ts["date"].dt.day - 1) / 30.0
                    )
                    yield_val = df_yield[
                        (df_yield["harvest_year"] == year)
                        & (df_yield[KEY_LOC] == region)
                    ]["yield"].values
                    lbl = (
                        f"{region} ({yield_val[0]:.1f} t/ha)"
                        if len(yield_val) > 0
                        else f"{region}"
                    )
                    ax.plot(months, region_ts[ts_name], color=color, label=lbl)

            ax.set_xlim(1, 12)
            ax.set_ylim(ts_limits[ts_name])
            ax.set_xlabel("Month")
            ax.set_ylabel(ts_name)
            ax.set_title(f"{ts_name} {year}")
            ax.legend(fontsize=12)

    # --- Colorbar for top row ---
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(
        sm, ax=axes[0, :], orientation="horizontal", fraction=0.05, pad=0.05
    )
    cbar.set_label("Yield (t/ha)")

    # --- Save figure ---
    out_file = os.path.join(output_dir, f"{crop}_yield_timeseries.png")
    fig.savefig(out_file, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved figure: {out_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize yield + timeseries")
    parser.add_argument(
        "-d",
        "--dataset",
        required=True,
        help="Dataset name, e.g., 'maize_US'",
    )
    parser.add_argument(
        "--timeseries", nargs="+", default=["fpar", "ndvi"], help="Timeseries files"
    )
    parser.add_argument("--years", nargs="+", help="Years to include")
    parser.add_argument(
        "--n_regions", type=int, default=5, help="Number of regions to highlight"
    )
    parser.add_argument("-o", "--output_dir", default="./", help="Output directory")
    args = parser.parse_args()

    visualize_yield_timeseries(
        dataset=args.dataset,
        timeseries=args.timeseries,
        years=args.years,
        n_regions=args.n_regions,
        output_dir=args.output_dir,
    )
