import os
import geopandas as gpd

from cybench.config import PATH_POLYGONS_DIR


def get_shapes_from_polygons(region):
    """
    Load administrative boundaries from the polygons folder.

    Folder structure assumed:
        POLYGONS_DIR/COUNTRY/COUNTRY.shp

    :param region: 2-letter country code
    :return: GeoDataFrame with an 'adm_id' column
    """
    region_dir = os.path.join(PATH_POLYGONS_DIR, region)
    shp_path = os.path.join(region_dir, f"{region}.shp")

    if not os.path.exists(shp_path):
        raise FileNotFoundError(
            f"Shapefile for region '{region}' not found at {shp_path}"
        )

    gdf = gpd.read_file(shp_path)

    # Ensure a column 'adm_id' exists
    if "adm_id" not in gdf.columns:
        # fallback: use first column that looks like an ID
        id_cols = [c for c in gdf.columns if "id" in c.lower() or "ID" in c]
        if id_cols:
            gdf["adm_id"] = gdf[id_cols[0]]
        else:
            # fallback: create a numeric index as ID
            gdf["adm_id"] = range(len(gdf))

    # Project to EPSG 4326
    if region == "BR":
        gdf = gdf.set_crs(epsg=4326)

    gdf = gdf.to_crs(4326)

    return gdf
