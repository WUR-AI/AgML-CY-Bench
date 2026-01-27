import glob
import re
from datetime import datetime

import numpy as np
from netCDF4 import Dataset
import matplotlib.pyplot as plt
from tqdm import tqdm


def parse_date_from_filename(path):
    m = re.search(r"(\d{8})", path)
    if not m:
        return None
    return datetime.strptime(m.group(1), "%Y%m%d")


def read_precip_2d(ds, precip_var):
    v = ds.variables[precip_var]
    precip = v[:]  # may be masked, may include time dim
    units = getattr(v, "units", "")
    fill = getattr(v, "_FillValue", None)
    miss = getattr(v, "missing_value", None)

    # Handle singleton time dimension if present
    if precip.ndim == 3:
        precip = precip[0, :, :]

    # Convert masked → NaN
    if np.ma.isMaskedArray(precip):
        precip = precip.filled(np.nan)
    else:
        precip = precip.astype(np.float64)

    # Guard explicit fill values
    if fill is not None:
        precip = np.where(precip == fill, np.nan, precip)
    if miss is not None:
        precip = np.where(precip == miss, np.nan, precip)

    # Convert from kg m-2 s-1 → mm/day if needed
    if units in ["kg m-2 s-1", "kg m**-2 s**-1"]:
        precip = precip * 86400.0

    # Precip should not be negative
    precip = np.where(precip < 0, np.nan, precip)

    return precip, units, fill, miss


def main(
    file_pattern,
    output_png,
    precip_var="Precipitation_Flux",
    lat_var="lat",
    lon_var="lon",
    title="AgERA5 Annual Precipitation (2021)",
):
    files = sorted(glob.glob(file_pattern))
    if not files:
        raise RuntimeError(f"No NetCDF files found for pattern: {file_pattern}")

    precip_sum = None
    valid_count = None
    lats = None
    lons = None
    units0 = None

    # PASS 1: annual sum + find max cell
    for i, f in enumerate(tqdm(files, desc="Pass 1/2: Summing daily precipitation")):
        with Dataset(f) as ds:
            if i == 0:
                print("Variables:", list(ds.variables.keys()))
                lats = ds.variables[lat_var][:]
                lons = ds.variables[lon_var][:]

            precip, units, fill, miss = read_precip_2d(ds, precip_var)
            if i == 0:
                units0 = units
                print("Units:", units0)
                print("_FillValue:", fill, "missing_value:", miss)
                print("First file:", f)

            if precip_sum is None:
                precip_sum = np.zeros_like(precip, dtype=np.float64)
                valid_count = np.zeros_like(precip, dtype=np.int32)

            mask_valid = np.isfinite(precip)
            precip_sum += np.where(mask_valid, precip, 0.0)
            valid_count += mask_valid.astype(np.int32)

    annual = precip_sum.copy()
    annual[valid_count == 0] = np.nan

    # Find max cell
    flat_idx = np.nanargmax(annual)
    iy, ix = np.unravel_index(flat_idx, annual.shape)

    max_val = annual[iy, ix]
    max_lat = float(lats[iy])
    max_lon = float(lons[ix])

    print("\n=== Max annual precipitation cell ===")
    print(f"Index (iy, ix): ({iy}, {ix})")
    print(f"Lat/Lon: ({max_lat}, {max_lon})")
    print(f"Annual max (mm): {max_val}")

    # PASS 2: timeseries for that cell
    dates = []
    vals = []

    for f in tqdm(files, desc="Pass 2/2: Extracting timeseries for max cell"):
        dt = parse_date_from_filename(f)
        with Dataset(f) as ds:
            precip, _, _, _ = read_precip_2d(ds, precip_var)
            v = precip[iy, ix]

        dates.append(dt if dt is not None else len(dates))
        vals.append(v)

    vals = np.array(vals, dtype=np.float64)

    # Flip latitude if needed for plotting
    if lats[0] > lats[-1]:
        annual = np.flipud(annual)
        lats = lats[::-1]
        # also flip iy index for correct plotting
        iy_plot = annual.shape[0] - 1 - iy
    else:
        iy_plot = iy

    annual_ma = np.ma.masked_invalid(annual)

    # ----------- PLOTTING: 2 subplots -----------
    fig, (ax_map, ax_ts) = plt.subplots(
        2, 1, figsize=(11, 10), gridspec_kw={"height_ratios": [3, 1]}
    )

    # Top: annual map
    extent = [lons.min(), lons.max(), lats.min(), lats.max()]
    img = ax_map.imshow(
        annual_ma,
        origin="lower",
        extent=extent,
        aspect="auto",
    )
    cbar = fig.colorbar(img, ax=ax_map)
    cbar.set_label("Annual precipitation (mm)")

    ax_map.scatter(
        [max_lon],
        [max_lat],
        s=160,
        facecolors="none",
        edgecolors="red",
        linewidths=2.5,
        zorder=5,
    )

    ax_map.set_title(title + " — max cell circled")
    ax_map.set_xlabel("Longitude")
    ax_map.set_ylabel("Latitude")

    # Bottom: timeseries
    ax_ts.plot(dates, vals, lw=1.5)
    ax_ts.set_title(
        f"Daily precipitation at max cell (lat={max_lat:.3f}, lon={max_lon:.3f})"
    )
    ax_ts.set_xlabel("Date")
    ax_ts.set_ylabel("Daily precipitation (mm/day)")
    ax_ts.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_png, dpi=200)
    plt.close()

    print(f"\nSaved combined figure: {output_png}")
    print("Timeseries min/max (mm/day):", np.nanmin(vals), np.nanmax(vals))


if __name__ == "__main__":
    main(
        file_pattern="/home/michiel/WUR/AgML-CY-Bench-data/predictors/AgERA5/prec/AgERA5_Precipitation_Flux_2021*.nc",
        output_png="AgERA5_Annual_Precipitation_2021_map_plus_timeseries.png",
        precip_var="Precipitation_Flux",
        lat_var="lat",
        lon_var="lon",
        title="AgERA5 Annual Precipitation (2021)",
    )
