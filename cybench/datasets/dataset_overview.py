import pandas as pd


def get_crop_country_summary(crop, yield_df, adm_id_col, year_col):
    countries_summary = {}
    countries = yield_df[adm_id_col].str[:2].unique()
    row_idx = 0
    column_names = [
        "crop_name",
        "country_code",
        "min_year",
        "max_year",
        "num_years",
        "avg num_regions",
        "data_size",
    ]
    for cn in countries:
        yield_cn_df = yield_df[yield_df[adm_id_col].str[:2] == cn]
        if len(yield_cn_df.index) <= 1:
            continue

        min_year = yield_cn_df[year_col].min()
        max_year = yield_cn_df[year_col].max()
        num_years = len(yield_cn_df[year_col].unique())
        # average number of regions
        num_regions = int(yield_cn_df.groupby([year_col], observed=True).size().mean())
        data_size = yield_cn_df[year_col].count()
        countries_summary["row" + str(row_idx)] = [
            crop,
            cn,
            min_year,
            max_year,
            num_years,
            num_regions,
            data_size,
        ]
        row_idx += 1

    return countries_summary, column_names


def print_data_summary(df, summary_file_prefix=""):
    crops = df["crop_name"].unique()
    for c in crops:
        countries_summary, column_names = get_crop_country_summary(
            c, df, "adm_id", "harvest_year"
        )
        countries_summary_df = pd.DataFrame.from_dict(
            countries_summary, columns=column_names, orient="index"
        )
        num_countries = len(countries_summary_df.index)
        print("")
        print(countries_summary_df.head(num_countries).to_string())
        countries_summary_df.to_csv(
            summary_file_prefix + "_" + c.replace(" ", "_") + "_summary.csv",
            index=False,
        )


yields_df = pd.read_csv("data/YIELD_EU.csv", header=0)
print_data_summary(yields_df, summary_file_prefix="EU")
