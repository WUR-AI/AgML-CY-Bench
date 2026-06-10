# Transform FAOSTAT website export to AgML CY-Bench compliant national-level CSV
# Input path (as provided): FAOSTAT_data.csv
# Output path: crop_yield_morocco_national.csv

import pandas as pd
from pathlib import Path
from datetime import datetime

in_path = Path("FAOSTAT_data.csv")
out_path = Path("crop_yield_morocco_national.csv")
log_path = Path("crop_yield_morocco_national_LOG.csv")

# Read
df_raw = pd.read_csv(in_path)

# Standard FAOSTAT columns expected; robust renaming if variations exist
rename_map = {
    "Area": "Area",
    "Area Code (M49)": "Area Code (M49)",
    "Element": "Element",
    "Element Code": "Element Code",
    "Item": "Item",
    "Item Code": "Item Code",
    "Year": "Year",
    "Year Code": "Year Code",
    "Unit": "Unit",
    "Value": "Value",
    "Flag": "Flag",
    "Flag Description": "Flag Description",
    "Note": "Note",
    # some exports may use slightly different headers:
    "Area Code": "Area Code (M49)",
    "Area Code (FAO)": "Area Code (M49)",
}

# Align columns
for k, v in list(rename_map.items()):
    if k in df_raw.columns:
        df_raw.rename(columns={k: v}, inplace=True)

# Filter to Morocco (guard: many exports already scoped)
if "Area" in df_raw.columns:
    df = df_raw[df_raw["Area"].str.lower() == "morocco"].copy()
else:
    df = df_raw.copy()

# Limit to requested items & years
items_requested = {
    "Wheat",
    "Barley",
    "Green corn (maize)",
    "Maize (corn)",
    "Rice",
    "Soya beans",
    "Sugar cane",
}
elements_requested = {"Area harvested", "Production", "Yield"}

if "Item" in df.columns:
    df = df[df["Item"].isin(items_requested)]
if "Element" in df.columns:
    df = df[df["Element"].isin(elements_requested)]

# Ensure Year is numeric
df["Year"] = pd.to_numeric(df["Year"], errors="coerce").astype("Int64")

# Unit normalization:
# - Harvest area expected in hectares (ha) — FAOSTAT already uses 'ha'.
# - Production expected in tonnes (t) — FAOSTAT uses 't'.
# - Yield expected in tonnes per hectare (t/ha).
#   FAOSTAT often uses 'hg/ha' (hectograms per hectare). Convert to t/ha by dividing by 10_000.
# Build a safe converted frame
df_conv = df.copy()
conversion_log = []

def convert_yield(row):
    if row["Element"] != "Yield":
        return row["Value"], row["Unit"], None
    unit = str(row["Unit"]).strip().lower()
    val = row["Value"]
    if pd.isna(val):
        return val, row["Unit"], None
    if unit in {"hg/ha", "hg / ha", "hectogram/hectare"}:
        # 10,000 hg = 1 tonne
        return float(val) / 10000.0, "t/ha", "Yield: hg/ha → t/ha (÷10000)"
    elif unit in {"t/ha", "tonnes/ha", "tonne/ha", "t per ha"}:
        return float(val), "t/ha", "Yield: already t/ha"
    else:
        return float(val), row["Unit"], f"Yield: unknown unit kept as-is ({row['Unit']})"

# Apply conversion only to Yield rows
new_vals = []
new_units = []
notes = []
for _, r in df_conv.iterrows():
    if r["Element"] == "Yield":
        v2, u2, note = convert_yield(r)
        new_vals.append(v2)
        new_units.append(u2)
        notes.append(note)
    else:
        new_vals.append(r["Value"])
        new_units.append(r["Unit"])
        notes.append(None)

df_conv.loc[df_conv["Element"] == "Yield", "Value"] = [v for v in new_vals if v is not None][:sum(df_conv["Element"]=="Yield")]
# The above slicing could be brittle; instead assign per-index for safety:
for i, r in df_conv[df_conv["Element"]=="Yield"].iterrows():
    v2, u2, note = convert_yield(r)
    df_conv.at[i, "Value"] = v2
    df_conv.at[i, "Unit"] = u2
    if note:
        conversion_log.append({
            "Item": r["Item"],
            "Year": r["Year"],
            "Original Unit": r["Unit"],
            "Element": "Yield",
            "Note": note
        })

# Pivot to wide (Element→columns)
df_wide = df_conv.pivot_table(
    index=["Item", "Year"],
    columns="Element",
    values="Value",
    aggfunc="first"
).reset_index()

# Map FAOSTAT item names to CY-Bench canonical crop_name (simple, non-destructive)
item_to_cy = {
    "Maize (corn)": "grain maize",
    "Green corn (maize)": "green maize",
    "Wheat": "wheat",
    "Barley": "barley",
    "Rice": "rice",
    "Soya beans": "soybean",
    "Sugar cane": "sugarcane",
}
df_wide["crop_name"] = df_wide["Item"].map(item_to_cy).fillna(df_wide["Item"])

# Rename element columns to CY-Bench
df_wide = df_wide.rename(columns={
    "Area harvested": "harvest_area",
    "Production": "production",
    "Yield": "yield",
})

# Build CY-Bench schema
df_out = pd.DataFrame({
    "crop_name": df_wide["crop_name"],
    "country_code": "MA",
    "adm_id": "MA-NAT",
    "season_name": "main",
    "planting_year": df_wide["Year"],
    "harvest_year": df_wide["Year"],
    "planting_date": "",   # not provided by FAOSTAT
    "harvest_date": "",    # not provided by FAOSTAT
    "yield": df_wide.get("yield"),
    "production": df_wide.get("production"),
    "planted_area": df_wide.get("harvest_area"),
    "harvest_area": df_wide.get("harvest_area"),
})

# Sort for readability
df_out = df_out.sort_values(["crop_name", "harvest_year"]).reset_index(drop=True)

# Save outputs
df_out.to_csv(out_path, index=False)

# Also save a small log of any unit conversions we performed for traceability
df_log = pd.DataFrame(conversion_log) if conversion_log else pd.DataFrame(
    [{"Note": "No yield unit conversions were necessary or detected."}]
)
df_log.to_csv(log_path, index=False)

out_path.as_posix(), log_path.as_posix(), df_out.shape
