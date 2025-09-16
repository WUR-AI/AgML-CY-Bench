import os
import sys
import pandas as pd

if len(sys.argv) != 3:
    print("Usage: python merge_region.py <crop> <country>")
    sys.exit(1)

crop = sys.argv[1]
country = sys.argv[2]

data_path = "/lustre/backup/SHARED/AIN/agml/python-output"
cn_dir = os.path.join(data_path, crop, country)

if not os.path.isdir(cn_dir):
    print(f"No directory found for {crop}-{country}")
    sys.exit(0)

indicators = ["tmin", "tmax", "prec", "rad", "tavg", "et0"]

print(f"Processing {crop}-{country}")

agera5_df = None
for ind in indicators:
    csv_file = os.path.join(cn_dir, ind, f"{ind}_{crop}_{country}.csv")
    if not os.path.exists(csv_file):
        print(f"  Missing: {ind}_{crop}_{country}.csv")
        continue

    ind_df = pd.read_csv(csv_file)

    if agera5_df is None:
        agera5_df = ind_df
    else:
        agera5_df = agera5_df.merge(ind_df, on=["crop_name", "adm_id", "date"])

if agera5_df is not None:
    if "prec" in agera5_df and "et0" in agera5_df:
        agera5_df["cwb"] = agera5_df["prec"] - agera5_df["et0"]
    agera5_df = agera5_df.round(3)
    out_file = os.path.join(cn_dir, f"meteo_{crop}_{country}.csv")
    agera5_df.to_csv(out_file, index=False)
    print(f"  Saved {out_file}")
else:
    print(f"No data merged for {crop}-{country}")