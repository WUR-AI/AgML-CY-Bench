import os
import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt

from cybench.config import REPO_DIR, PATH_POLYGONS_DIR

# -------------------------------------------------------------------
# AAGIS mapping
# -------------------------------------------------------------------
AAGIS_TO_NAME = {
    111: "NSW Far West",
    121: "NSW North West Slopes and Plains",
    122: "NSW Central West",
    123: "NSW Riverina",
    131: "NSW Tablelands (Northern Central and Southern)",
    132: "NSW Coastal",
    221: "VIC Mallee",
    222: "VIC Wimmera",
    223: "VIC Central North",
    231: "VIC Southern and Eastern Victoria",
    311: "QLD Cape York and the Gulf",
    312: "QLD West and South West",
    313: "QLD Central North",
    314: "QLD Charleville - Longreach",
    321: "QLD Eastern Darling Downs",
    322: "QLD Western Downs and Central Highlands",
    331: "QLD Southern Coastal - Curtis to Moreton",
    332: "QLD Northern Coastal - Mackay to Cairns",
    411: "SA Northern Pastoral",
    421: "SA Eyre Peninsula",
    422: "SA Murray Lands and Yorke Peninsula",
    431: "SA South East",
    511: "WA The Kimberley",
    512: "WA Pilbara and Central Pastoral",
    521: "WA Central and Southern Wheat Belt",
    522: "WA Northern and Eastern Wheat Belt",
    531: "WA South West Coastal",
    631: "TAS Tasmania",
    711: "NT Alice Springs Districts",
    712: "NT Barkly Tablelands",
    713: "NT Victoria River District - Katherine",
    714: "NT Top End Darwin",
    799: "Other territories",
}


# -------------------------------------------------------------------
# Helpers
# -------------------------------------------------------------------
def check_region_mapping(df: pd.DataFrame, mapping: dict):
    """Check consistency between mapping and CSV region names."""
    csv_regions = set(df["ABARES region"].unique())
    map_regions = set(mapping.values())

    missing_in_csv = map_regions - csv_regions
    if missing_in_csv:
        print("⚠️ Names in mapping but missing in CSV:")
        print("\n".join(missing_in_csv))
    else:
        print("✅ All names in mapping exist in CSV.")

    missing_in_map = csv_regions - map_regions
    if missing_in_map:
        print("⚠️ Names in CSV but missing in mapping:")
        print("\n".join(missing_in_map))
    else:
        print("✅ All names in CSV exist in mapping.")


def load_geometries():
    """Load and preprocess Australian regions shapefile."""
    shapefile_path = os.path.join(PATH_POLYGONS_DIR, "AU", "AU.shp")
    gdf = gpd.read_file(shapefile_path)

    # Calculate total area in hectares
    gdf = gdf.to_crs(epsg=3577)  # GDA94 / Australian Albers
    gdf["region_total_area_ha"] = gdf.geometry.area / 1e4
    gdf = gdf.to_crs(epsg=4326)  # back to lat/lon

    # Extract numeric AAGIS code
    gdf["AAGIS"] = gdf["adm_id"].str.replace("AU-", "").astype(int)
    gdf["region_name"] = gdf["AAGIS"].map(AAGIS_TO_NAME)
    return gdf


def preprocess_wheat_data(df: pd.DataFrame) -> pd.DataFrame:
    """Filter, pivot, and compute wheat production stats."""
    wheat = df[df["Variable"].isin(["Wheat produced (t)", "Wheat area sown (ha)"])]
    wheat = wheat.drop(columns=["RSE"], errors="ignore")

    wheat_pivot = wheat.pivot_table(
        index=["Year", "ABARES region"], columns="Variable", values="Value"
    ).reset_index()

    wheat_pivot = wheat_pivot.rename(
        columns={
            "Year": "harvest_year",
            "Wheat produced (t)": "production",
            "Wheat area sown (ha)": "planted_area",
        }
    )
    wheat_pivot["yield"] = wheat_pivot["production"] / wheat_pivot["planted_area"]
    return wheat_pivot


def compute_median_fraction(
    merged_gdf: gpd.GeoDataFrame, years: range
) -> gpd.GeoDataFrame:
    """Compute median planted fraction per region across given years."""
    df_filtered = merged_gdf[merged_gdf["harvest_year"].isin(years)].copy()
    records = []
    for adm_id, group in df_filtered.groupby("adm_id"):
        records.append(
            {
                "adm_id": adm_id,
                "region_name": group.iloc[0]["region_name"],
                "geometry": group.iloc[0]["geometry"],
                "planted_fraction": group["planted_fraction"].median(),
            }
        )
    return gpd.GeoDataFrame(records, geometry="geometry", crs=merged_gdf.crs)


def plot_median_fraction(
    median_gdf: gpd.GeoDataFrame, gdf: gpd.GeoDataFrame, threshold: float
):
    """Plot median planted fraction map with annotations."""
    fig, ax = plt.subplots(figsize=(12, 10))
    median_gdf.plot(
        column="planted_fraction",
        ax=ax,
        cmap="viridis",
        edgecolor="black",
        linewidth=0.5,
        legend=True,
        legend_kwds={"label": "Median planted fraction (2003–2023)"},
    )
    gdf.boundary.plot(ax=ax, color="black", linewidth=0.8)

    for _, row in median_gdf.iterrows():
        centroid = row["geometry"].centroid
        color = "green" if row["planted_fraction"] >= threshold else "red"
        ax.text(
            centroid.x,
            centroid.y,
            f"{row['planted_fraction']*100:.3f}",  # as %
            ha="center",
            va="center",
            fontsize=8,
            color=color,
            weight="bold",
        )

    ax.set_title("Median Wheat Planted Fraction in Australia (2003–2023)", fontsize=16)
    ax.axis("off")
    plt.show()


# -------------------------------------------------------------------
# Main pipeline
# -------------------------------------------------------------------
def main():
    # Load shapefile + CSV
    gdf = load_geometries()
    csv_path = os.path.join(
        REPO_DIR,
        "data_preparation",
        "crop_statistics_AU",
        "fdp-regional-historical.csv",
    )
    df = pd.read_csv(csv_path)

    # Check mapping consistency
    check_region_mapping(df, AAGIS_TO_NAME)

    # Preprocess wheat stats
    wheat_pivot = preprocess_wheat_data(df)

    # Merge geodata with wheat stats
    merged_gdf = gdf.merge(
        wheat_pivot, left_on="region_name", right_on="ABARES region", how="left"
    )
    merged_gdf["planted_fraction"] = (
        merged_gdf["planted_area"] / merged_gdf["region_total_area_ha"]
    )

    # Compute median per region
    years_to_include = range(2003, 2024)
    median_gdf = compute_median_fraction(merged_gdf, years_to_include)

    # Plot
    threshold = 0.000005
    plot_median_fraction(median_gdf, gdf, threshold)

    # Build cleaned wheat dataframe
    wheat_df = merged_gdf[
        ["harvest_year", "production", "planted_area", "adm_id", "planted_fraction"]
    ].copy()

    wheat_df["crop_name"] = "wheat"
    wheat_df["country_code"] = "AU"
    wheat_df["yield"] = wheat_df["production"] / wheat_df["planted_area"]

    wheat_df = wheat_df[
        [
            "crop_name",
            "country_code",
            "adm_id",
            "harvest_year",
            "yield",
            "planted_area",
            "production",
            "planted_fraction",
        ]
    ]

    # Clean NaNs/infs
    wheat_df = wheat_df.replace([np.inf, -np.inf], np.nan).dropna()

    # Add median stats
    median_stats = (
        wheat_df.groupby("adm_id")
        .agg(
            median_planted_area=("planted_area", "median"),
            median_planted_fraction=("planted_fraction", "median"),
        )
        .reset_index()
    )
    wheat_df = wheat_df.merge(median_stats, on="adm_id", how="left")

    # Filter invalid rows
    wheat_df = wheat_df[
        ~(
            (wheat_df["production"] == 0)
            & (wheat_df["planted_area"] == 0)
            & (wheat_df["yield"] == 0)
        )
        & (wheat_df["median_planted_fraction"] >= threshold)
    ]

    # Final cleanup
    wheat_df["harvest_year"] = wheat_df["harvest_year"].astype(int)
    wheat_df[["yield", "planted_area", "production"]] = wheat_df[
        ["yield", "planted_area", "production"]
    ].round(3)

    wheat_df = wheat_df[
        [
            "crop_name",
            "country_code",
            "adm_id",
            "harvest_year",
            "yield",
            "planted_area",
            "production",
        ]
    ]
    # Save
    out_path = os.path.join(
        REPO_DIR, "cybench", "data", "wheat", "AU", "yield_wheat_AU.csv"
    )
    wheat_df.to_csv(out_path, index=False)
    print(f"✅ Saved cleaned dataset: {out_path}")
    print("Final shape:", wheat_df.shape)


if __name__ == "__main__":
    main()
