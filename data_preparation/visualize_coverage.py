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

if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        prog="visualize_coverage.py", description="Visualize coverage"
    )
    # Add country_code argument (optional)
    parser.add_argument(
        "--country_code",
        type=str,
        nargs="*",  # Allows for multiple values or none
        help="Specify the country code(s) to visualize. If not provided, all countries will be used.",
    )
    args = parser.parse_args()

    crop_mask_threshold = 0.1 * 255

    alpha_val = 1.0

    crops = ["wheat", "maize"]

    world_shapefile_path = os.path.join(
        REPO_DIR,
        "data_preparation",
        "ne_110m_admin_0_countries",
        "ne_110m_admin_0_countries.shp",
    )
    world = gpd.read_file(world_shapefile_path)
    world = world.to_crs(epsg=4326)

    fig, (ax1, ax2) = plt.subplots(nrows=2, figsize=(15, 15))

    # Create an empty list to store the merged data
    all_country_data = []
    for crop in crops:
        crop_data = []
        countries = [
            country_code
            for country_code in os.listdir(os.path.join(PATH_DATA_DIR, crop))
            if os.path.isdir(os.path.join(PATH_DATA_DIR, crop, country_code))
        ]

        # If country_code argument is provided, filter by it
        if args.country_code:
            countries = [cc for cc in countries if cc in args.country_code]

        for country_code in countries:
            geo_df = get_shapes_from_polygons(region=country_code)
            geo_df = geo_df[[KEY_LOC, "geometry"]]

            # targets
            yield_file = os.path.join(
                PATH_DATA_DIR, crop, country_code, f"yield_{crop}_{country_code}.csv"
            )
            df_y = pd.read_csv(yield_file, header=0)
            df_y = df_y.rename(columns={"harvest_year": KEY_YEAR})
            df_y = df_y[[KEY_LOC, KEY_YEAR, KEY_TARGET]]
            df_y = df_y.dropna(axis=0)
            df_y = df_y[df_y[KEY_TARGET] > 0.0]
            df_y = (
                df_y.reset_index()
            )  # Reset the multi-index to have the columns explicitly

            merged_country_df = geo_df.merge(df_y, on=KEY_LOC, how="left")
            merged_country_df["country_code"] = country_code
            # Append the merged country data to the list
            crop_data.append(merged_country_df)
        # Append data for this crop
        all_country_data.append(crop_data)
    # Combine all the country data into one large DataFrame for each crop
    merged_df_wheat = pd.concat(all_country_data[0], ignore_index=True)
    merged_df_maize = pd.concat(all_country_data[1], ignore_index=True)

    # Combine the two GeoDataFrames
    merged_df = pd.concat([merged_df_wheat, merged_df_maize], ignore_index=True)

    # Calculate the total bounds of the combined GeoDataFrame
    minx, miny, maxx, maxy = merged_df.total_bounds

    # Create a consistent aspect ratio for the plot by setting the axis limits
    aspect_ratio = (maxy - miny) / (maxx - minx)

    # Compute the median yield per region (adm_id) across all years
    median_yield_maize_df = (
        merged_df_maize.groupby("adm_id")["yield"].median().reset_index()
    )
    median_yield_wheat_df = (
        merged_df_wheat.groupby("adm_id")["yield"].median().reset_index()
    )

    wheat_map = os.path.join(
        REPO_DIR,
        "data_preparation",
        "global_crop_AFIs_ESA_WC",
        "crop_mask_winter_spring_cereals_WC.tif",
    )
    maize_map = os.path.join(
        REPO_DIR,
        "data_preparation",
        "global_crop_AFIs_ESA_WC",
        "crop_mask_maize_WC.tif",
    )

    # Merge the median yield back with the geometries from the original dataframe
    merged_df_maize = merged_df_maize.drop_duplicates(subset="adm_id")
    merged_df_maize = merged_df_maize.merge(
        median_yield_maize_df, on="adm_id", suffixes=("", "_median")
    )

    merged_df_wheat = merged_df_wheat.drop_duplicates(subset="adm_id")
    merged_df_wheat = merged_df_wheat.merge(
        median_yield_wheat_df, on="adm_id", suffixes=("", "_median")
    )

    threshold = 0.01
    merged_df_maize["threshold_yield"] = (
        merged_df_maize["yield_median"] > threshold
    ).astype(int)

    merged_df_wheat["threshold_yield"] = (
        merged_df_wheat["yield_median"] > threshold
    ).astype(int)

    masked_df_maize = merged_df_maize[merged_df_maize["threshold_yield"] == 1]
    masked_df_maize.plot(
        ax=ax1,
        color="palegreen",
        edgecolor="none",  # don't draw edges again
        alpha=1.0,
        zorder=1,
    )

    with rasterio.open(maize_map) as src:
        # Read the whole image as an array
        image_array = src.read(1)
        left, bottom, right, top = src.bounds

        # Resize the image (downsampling)
        pil_image = Image.fromarray(image_array)
        pil_image_resized = pil_image.resize(
            (pil_image.width // 10, pil_image.height // 10), Image.Resampling.LANCZOS
        )
        downsampled_image = np.array(pil_image_resized)
        # Plot the downsampled image with the correct extent (FIRST)
        thresholded_mask = (downsampled_image > crop_mask_threshold).astype(int)
        masked_image = np.ma.masked_where(thresholded_mask == 0, thresholded_mask)

        crop_cmap = ListedColormap(["yellow"])
        ax1.imshow(
            masked_image,
            cmap=crop_cmap,
            extent=(left, right, bottom, top),
            zorder=3,
            alpha=1.0,
            interpolation="none",
        )

    masked_df_wheat = merged_df_wheat[merged_df_wheat["threshold_yield"] == 1]
    masked_df_wheat.plot(
        ax=ax2,
        color="palegreen",
        edgecolor="none",  # don't draw edges again
        alpha=1.0,
        zorder=1,
    )

    with rasterio.open(wheat_map) as src:
        # Read the whole image as an array
        image_array = src.read(1)
        left, bottom, right, top = src.bounds

        # Resize the image (downsampling)
        pil_image = Image.fromarray(image_array)
        pil_image_resized = pil_image.resize(
            (pil_image.width // 10, pil_image.height // 10), Image.Resampling.LANCZOS
        )
        downsampled_image = np.array(pil_image_resized)

        # Plot the downsampled image with the correct extent (FIRST)
        thresholded_mask = (downsampled_image > crop_mask_threshold).astype(int)
        masked_image = np.ma.masked_where(thresholded_mask == 0, thresholded_mask)

        crop_cmap = ListedColormap(["yellow"])
        ax2.imshow(
            masked_image,
            cmap=crop_cmap,
            extent=(left, right, bottom, top),
            zorder=3,
            alpha=1.0,
            interpolation="none",
        )

    # Plot the world map as background.
    world.plot(
        ax=ax1, color="lightgrey", edgecolor="grey", linewidth=0.3, alpha=0.4, zorder=0
    )
    world.plot(
        ax=ax2, color="lightgrey", edgecolor="grey", linewidth=0.3, alpha=0.4, zorder=0
    )
    # Plot the world borders on top (edges only)
    world.boundary.plot(ax=ax1, color="grey", linewidth=0.3, zorder=5)
    world.boundary.plot(ax=ax2, color="grey", linewidth=0.3, zorder=5)

    # Create legend patches for Maize (ax1)
    maize_legend_patches = [
        Patch(facecolor="palegreen", edgecolor="none", label="CY-Bench maize coverage"),
        Patch(facecolor="yellow", edgecolor="none", label="Crop mask maize"),
    ]

    # Create legend patches for Wheat (ax2)
    wheat_legend_patches = [
        Patch(facecolor="palegreen", edgecolor="none", label="CY-Bench wheat coverage"),
        Patch(facecolor="yellow", edgecolor="none", label="Crop mask wheat"),
    ]

    # Add legends to subplots
    legend1 = ax1.legend(
        handles=maize_legend_patches,
        loc="lower left",
        frameon=True,
        framealpha=0.3,  # make frame light
        facecolor="white",
        edgecolor="gray",
    )

    legend2 = ax2.legend(
        handles=wheat_legend_patches,
        loc="lower left",
        frameon=True,
        framealpha=0.3,
        facecolor="white",
        edgecolor="gray",
    )

    ax1.set_xlim(max(-180, minx - 5), min(180, maxx + 5))
    ax1.set_ylim(max(-90, miny - 5), min(90, maxy + 5))
    ax1.set_xticks([])  # Remove x-axis ticks
    ax1.set_yticks([])  # Remove y-axis ticks
    ax1.set_aspect("equal", adjustable="box")

    ax2.set_xlim(max(-180, minx - 5), min(180, maxx + 5))
    ax2.set_ylim(max(-90, miny - 5), min(90, maxy + 5))
    ax2.set_xticks([])  # Remove x-axis ticks
    ax2.set_yticks([])  # Remove y-axis ticks
    ax2.set_aspect("equal", adjustable="box")

    # Save the figure as a single image (for all years)
    output_dir = "output_maps"  # Define the output directory for saving images
    os.makedirs(output_dir, exist_ok=True)
    image_filename = os.path.join(output_dir, "CY-Bench-coverage.png")
    plt.subplots_adjust(hspace=-0.3)
    print(f"save {image_filename}")
    plt.savefig(image_filename, dpi=400, bbox_inches="tight", pad_inches=0.1)
