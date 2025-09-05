import os
import cdsapi
import zipfile
import argparse

# Mapping of parameters to ERA5 variable/statistic
AgERA5_params = {
    "Maximum_Temperature": ("2m_temperature", "24_hour_maximum"),
    "Minimum_Temperature": ("2m_temperature", "24_hour_minimum"),
    "Mean_Temperature": ("2m_temperature", "24_hour_mean"),
    "Solar_Radiation_Flux": ("solar_radiation_flux", None),
    "Precipitation_Flux": ("precipitation_flux", None),
    "Reference_Evapotranspiration": ("reference_evapotranspiration", None),
    "Vapour_Pressure_Deficit": ("vapour_pressure_deficit_at_maximum_temperature", None)
}

download_root = '/lustre/backup/SHARED/AIN/agml/predictors/AgERA5/v2.0'

def get_agera5_params(sel_param, year):
    var, stat = AgERA5_params[sel_param]
    retrieve_params = {
        "version": "2_0",
        "format": "zip",
        "variable": var,
        "year": [str(year)],
        "month": [f"{m:02d}" for m in range(1, 13)],
        "day": [f"{d:02d}" for d in range(1, 32)],
    }
    if stat is not None:
        retrieve_params["statistic"] = [stat]
    return retrieve_params

def is_zip_ok(zip_path):
    return os.path.isfile(zip_path) and zipfile.is_zipfile(zip_path)

def download_agera5(cds, year, param):
    zipfile_path = os.path.join(download_root, param, f"{param}_{year}.zip")
    if is_zip_ok(zipfile_path):
        print(f"Skipping {zipfile_path} (already exists and OK)")
        return

    os.makedirs(os.path.join(download_root, param), exist_ok=True)
    retrieve_params = get_agera5_params(param, year)
    
    try:
        print(f"Downloading {param} {year} ...")
        cds.retrieve("sis-agrometeorological-indicators", retrieve_params).download(zipfile_path)
        print(f"Download complete: {param} {year}")
    except Exception as e:
        print(f"Error downloading {param} {year}: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--year", type=int, required=True)
    parser.add_argument("--param", type=str, required=True, choices=list(AgERA5_params.keys()))
    args = parser.parse_args()

    cds = cdsapi.Client(progress=True)
    download_agera5(cds, args.year, args.param)

