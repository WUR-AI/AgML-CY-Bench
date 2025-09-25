import os
import pandas as pd
import geopandas as gpd
import argparse
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import Normalize
from PIL import Image

from cybench.config import (
    KEY_LOC,
    KEY_TARGET,
    KEY_YEAR,
    PATH_DATA_DIR,
    CROP_YIELD_RANGES,
    REPO_DIR,
)
from cybench.util.geo import get_shapes_from_polygons

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="visualize_yield.py", description="Visualize yield"
    )
    # Add country_code argument (optional)
    parser.add_argument(
        "--country_code",
        type=str,
        nargs="*",  # Allows for multiple values or none
        help="Specify the country code(s) to visualize. If not provided, all countries will be used.",
    )
    args = parser.parse_args()

    crops = ["wheat", "maize"]

    world_shapefile_path = os.path.join(
        REPO_DIR,
        "data_preparation",
        "ne_110m_admin_0_countries",
        "ne_110m_admin_0_countries.shp",
    )

    world = gpd.read_file(world_shapefile_path)

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

    # Create a folder to store individual frames (images)
    output_dir = "frames"
    os.makedirs(output_dir, exist_ok=True)

    # Define the color map (using "viridis" as an example)
    cmap = cm.viridis

    # Combine the two GeoDataFrames
    merged_df = pd.concat([merged_df_wheat, merged_df_maize], ignore_index=True)

    # Calculate the total bounds of the combined GeoDataFrame
    minx, miny, maxx, maxy = merged_df.total_bounds

    # Create a consistent aspect ratio for the plot by setting the axis limits
    aspect_ratio = (maxy - miny) / (maxx - minx)

    # Iterate over unique years in the dataset and create plots
    years = sorted(
        [year for year in merged_df_wheat[KEY_YEAR].unique() if 2003 <= year <= 2023]
    )

    image_files = []

    for year in years:
        # Filter the data for the current year for both crops
        filtered_df_maize = merged_df_maize[merged_df_maize[KEY_YEAR] == year]
        filtered_df_wheat = merged_df_wheat[merged_df_wheat[KEY_YEAR] == year]

        if not filtered_df_wheat.empty and not filtered_df_maize.empty:
            # Create the plot with two subplots (rows)
            fig, (ax1, ax2) = plt.subplots(nrows=2, figsize=(15, 15))

            # Plot the base world map for both subplots
            world.plot(ax=ax1, color="lightgrey", edgecolor="black", linewidth=0.1)
            world.plot(ax=ax2, color="lightgrey", edgecolor="black", linewidth=0.1)

            # Maize-specific normalization
            norm_maize = Normalize(
                vmin=CROP_YIELD_RANGES["maize"]["min"],
                vmax=CROP_YIELD_RANGES["maize"]["max"],
            )

            # Wheat-specific normalization
            norm_wheat = Normalize(
                vmin=CROP_YIELD_RANGES["wheat"]["min"],
                vmax=CROP_YIELD_RANGES["wheat"]["max"],
            )

            # Plot maize yield data on the second subplot (ax2)
            filtered_df_maize.plot(
                column=KEY_TARGET,  # Use the actual column name for yield
                ax=ax1,
                legend=False,
                cmap=cmap,
                edgecolor="black",
                linewidth=0.0,
                vmin=norm_maize.vmin,
                vmax=norm_maize.vmax,
            )
            # Plot wheat yield data on the first subplot (ax1)
            filtered_df_wheat.plot(
                column=KEY_TARGET,  # Use the actual column name for yield
                ax=ax2,
                legend=False,
                cmap=cmap,
                edgecolor="black",
                linewidth=0.0,
                vmin=norm_wheat.vmin,
                vmax=norm_wheat.vmax,
            )
            # Set consistent axis limits across both subplots (same bounding box)
            for ax in [ax1, ax2]:
                ax.set_xlim(minx, maxx)
                ax.set_ylim(miny, maxy)
                ax.set_xticks([])  # Removes x-axis ticks (longitude)
                ax.set_yticks([])  # Removes y-axis ticks (latitude)
                ax.set_aspect("equal", adjustable="box")

            sm_maize = plt.cm.ScalarMappable(cmap=cmap, norm=norm_maize)
            sm_maize.set_array([])  # Empty array to create the colorbar
            cax1 = ax1.inset_axes(
                [0.01, 0.07, 0.20, 0.03]
            )  # Positioning the color bar inside the plot
            cbar_maize = fig.colorbar(sm_maize, cax=cax1, orientation="horizontal")
            cbar_maize.ax.tick_params(labelsize=6, colors="black")
            cbar_maize.outline.set_linewidth(0.5)
            cbar_maize.outline.set_edgecolor("black")
            cbar_maize.set_label(f"Maize yield (tonne/ha) [{int(year)}]", fontsize=12)
            cbar_maize.ax.yaxis.set_label_position("left")  # Align label to the left
            cbar_maize.ax.xaxis.set_label_coords(0.53, 3.1)

            # Add color bars for each crop
            sm_wheat = plt.cm.ScalarMappable(cmap=cmap, norm=norm_wheat)
            sm_wheat.set_array([])  # Empty array to create the colorbar
            cax2 = ax2.inset_axes(
                [0.01, 0.07, 0.20, 0.03]
            )  # Positioning the color bar inside the plot
            cbar_wheat = fig.colorbar(sm_wheat, cax=cax2, orientation="horizontal")
            cbar_wheat.ax.tick_params(labelsize=6, colors="black")
            cbar_wheat.outline.set_linewidth(0.5)
            cbar_wheat.outline.set_edgecolor("black")
            cbar_wheat.set_label(f"Wheat yield (tonne/ha) [{int(year)}]", fontsize=12)
            cbar_wheat.ax.yaxis.set_label_position("left")
            cbar_wheat.ax.xaxis.set_label_coords(0.53, 3.1)

            # Save the plot as an image
            image_filename = os.path.join(output_dir, f"year_{int(year)}.png")
            image_files.append(image_filename)
            # plt.suptitle(f"{int(year)}", fontsize=12, y=0.79)
            plt.subplots_adjust(hspace=-0.4)  #
            plt.savefig(image_filename, dpi=400, bbox_inches="tight", pad_inches=0.1)
            plt.close(fig)

    # Create the GIF using Pillow
    gif_filename = f"yield-animation.gif"
    images = [Image.open(image_file) for image_file in image_files]

    # Save the images as a GIF
    images[0].save(
        gif_filename, save_all=True, append_images=images[1:], duration=750, loop=0
    )
