import os
import pandas as pd
import geopandas as gpd
import argparse
import matplotlib.pyplot as plt
from PIL import Image
import rasterio
import numpy as np
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch

from cybench.config import (
    KEY_LOC,
    KEY_TARGET,
    KEY_YEAR,
    PATH_DATA_DIR,
    REPO_DIR,
)
from cybench.util.geo import get_shapes_from_polygons


def load_crop_data(crop, country_codes):
    """Load yield data and merge with geometry for a given crop."""
    crop_data = []

    for country_code in country_codes:
        geo_df = get_shapes_from_polygons(region=country_code)
        geo_df = geo_df[[KEY_LOC, "geometry"]]

        yield_file = os.path.join(
            PATH_DATA_DIR, crop, country_code, f"yield_{crop}_{country_code}.csv"
        )
        if not os.path.exists(yield_file):
            print(f"⚠️  No yield file for {crop} in {country_code}, skipping")
            continue

        df_y = pd.read_csv(yield_file)
        df_y = df_y.rename(columns={"harvest_year": KEY_YEAR})
        df_y = df_y[[KEY_LOC, KEY_YEAR, KEY_TARGET]].dropna()
        df_y = df_y[df_y[KEY_TARGET] > 0.0].reset_index(drop=True)

        merged_country_df = geo_df.merge(df_y, on=KEY_LOC, how="left")
        merged_country_df["country_code"] = country_code
        crop_data.append(merged_country_df)

    if crop_data:
        return pd.concat(crop_data, ignore_index=True)
    else:
        return None


def plot_crop(ax, crop_df, raster_path, crop_mask_threshold, coverage_color):
    """Plot crop coverage + crop mask overlay."""
    # Compute median yield
    median_yield = crop_df.groupby("adm_id")[KEY_TARGET].median().reset_index()
    crop_df = crop_df.drop_duplicates(subset="adm_id")
    crop_df = crop_df.merge(median_yield, on="adm_id", suffixes=("", "_median"))

    # Threshold
    threshold = 0.01
    crop_df["threshold_yield"] = (crop_df[f"{KEY_TARGET}_median"] > threshold).astype(
        int
    )
    masked_df = crop_df[crop_df["threshold_yield"] == 1]

    # Plot shapefile mask
    masked_df.plot(ax=ax, color=coverage_color, edgecolor="none", alpha=1.0, zorder=1)

    # Raster overlay
    with rasterio.open(raster_path) as src:
        image_array = src.read(1)
        left, bottom, right, top = src.bounds

        pil_image = Image.fromarray(image_array)
        pil_image_resized = pil_image.resize(
            (pil_image.width // 10, pil_image.height // 10), Image.Resampling.LANCZOS
        )
        downsampled_image = np.array(pil_image_resized)

        thresholded_mask = (downsampled_image > crop_mask_threshold).astype(int)
        masked_image = np.ma.masked_where(thresholded_mask == 0, thresholded_mask)

        crop_cmap = ListedColormap(["yellow"])
        ax.imshow(
            masked_image,
            cmap=crop_cmap,
            extent=(left, right, bottom, top),
            zorder=3,
            alpha=1.0,
            interpolation="none",
        )

    return ax


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        prog="visualize_coverage.py", description="Visualize coverage"
    )
    parser.add_argument(
        "--country_code",
        type=str,
        nargs="*",
        help="Specify the country code(s). If not provided, all countries are used.",
    )
    args = parser.parse_args()

    crop_mask_threshold = 0.1 * 255
    crops = ["maize", "wheat"]

    # Load world background
    world_shapefile_path = os.path.join(
        REPO_DIR,
        "data_preparation",
        "ne_110m_admin_0_countries",
        "ne_110m_admin_0_countries.shp",
    )
    world = gpd.read_file(world_shapefile_path).to_crs(epsg=4326)

    # Collect data
    all_country_data = {}
    for crop in crops:
        crop_dir = os.path.join(PATH_DATA_DIR, crop)
        if not os.path.exists(crop_dir):
            continue

        countries = [
            cc
            for cc in os.listdir(crop_dir)
            if os.path.isdir(os.path.join(crop_dir, cc))
        ]
        if args.country_code:
            countries = [cc for cc in countries if cc in args.country_code]

        crop_df = load_crop_data(crop, countries)
        if crop_df is not None:
            all_country_data[crop] = crop_df
        else:
            print(f"⚠️  No data found for {crop}")

    if not all_country_data:
        raise ValueError("❌ No crop data available at all")

    # Raster paths
    raster_paths = {
        "wheat": os.path.join(
            REPO_DIR,
            "data_preparation",
            "global_crop_AFIs_ESA_WC",
            "crop_mask_winter_spring_cereals_WC.tif",
        ),
        "maize": os.path.join(
            REPO_DIR,
            "data_preparation",
            "global_crop_AFIs_ESA_WC",
            "crop_mask_maize_WC.tif",
        ),
    }

    # Create figure with two separate subplots (maize first)
    fig, axes = plt.subplots(nrows=2, figsize=(15, 12), gridspec_kw={"hspace": 0.1})
    if not isinstance(axes, (list, np.ndarray)):
        axes = [axes]

    coverage_colors = {"maize": "palegreen", "wheat": "palegreen"}

    for ax, crop in zip(axes, ["maize", "wheat"]):
        crop_df = all_country_data.get(crop)
        if crop_df is None:
            print(f"⚠️  No data to plot for {crop}, skipping")
            ax.set_visible(False)  # optional: hide the empty subplot
            continue

        plot_crop(
            ax,
            crop_df,
            raster_paths[crop],
            crop_mask_threshold,
            coverage_colors[crop],
        )
        world.plot(
            ax=ax,
            color="lightgrey",
            edgecolor="grey",
            linewidth=0.3,
            alpha=0.4,
            zorder=0,
        )
        world.boundary.plot(ax=ax, color="grey", linewidth=0.3, zorder=5)

        # Legends
        legend_patches = [
            Patch(
                facecolor=coverage_colors[crop],
                edgecolor="none",
                label=f"CY-Bench {crop} coverage",
            ),
            Patch(facecolor="yellow", edgecolor="none", label=f"Crop mask {crop}"),
        ]
        ax.legend(
            handles=legend_patches,
            loc="lower left",
            frameon=True,
            framealpha=0.3,
            facecolor="white",
            edgecolor="gray",
        )

        # Axis formatting
        if all_country_data.get(crop) is not None:
            minx, miny, maxx, maxy = all_country_data[crop].total_bounds
            ax.set_xlim(max(-180, minx - 5), min(180, maxx + 5))
            ax.set_ylim(max(-90, miny - 5), min(90, maxy + 5))
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_aspect("equal", adjustable="box")
        ax.set_title(f"{crop.capitalize()} coverage", fontsize=16)

    # Save output
    output_dir = "output_maps"
    os.makedirs(output_dir, exist_ok=True)
    image_filename = os.path.join(output_dir, "CY-Bench-coverage.png")
    print(f"✅ Saving {image_filename}")
    plt.savefig(image_filename, dpi=400, bbox_inches="tight", pad_inches=0.1)
    plt.close()
