
# Meta --------------------------------------------------------------------
# Author:        Pablo Estrada
# Date Created:  1/27/2025
# Date Edited:   3/30/2025
# Notes:         Python file to build Medicare Advantage dataset

import pandas as pd
import numpy as np
import warnings
warnings.simplefilter('ignore')

# Call individual scripts -------------------------------------------------

print("\n1) Read in enrollment data for January of each year...")
exec(open("data-code/1_Plan_Data.py").read())

print("\n3) Read in service area data...")
exec(open("data-code/3_Service_Areas.py").read())

print("\n4) Read in market pentration data...")
exec(open("data-code/4_Penetration_Files.py").read())

print("\n5) Read in MA star rating data ...")
exec(open("data-code/rating_variables.py").read())
exec(open("data-code/5_Star_Ratings.py").read())

print("\n7) Read in final benchmark rates...")
exec(open("data-code/7_MA_Benchmark.py").read())


# Load the datasets
full_ma_data = pd.read_csv("data/output/full_ma_data.csv")
contract_service_area = pd.read_csv("data/output/contract_service_area.csv")
star_ratings = pd.read_csv("data/output/star_ratings.csv")
ma_penetration_data = pd.read_csv("data/output/ma_penetration.csv")
benchmark_final = pd.read_csv("data/output/ma_benchmarks.csv")

# Perform the first join and filter
final_data = (
    full_ma_data.merge(
        contract_service_area[["contractid", "fips", "year"]],
        on=["contractid", "fips", "year"],
        how="inner"
    )
    .query(
        "~state.isin(['VI', 'PR', 'MP', 'GU', 'AS', '']) & snp == 'No' & "
        "(planid < 800 | planid >= 900) & planid.notna() & fips.notna()"
    )
)

# Perform the subsequent joins
final_data = (
    final_data.merge(
        star_ratings.drop(columns=["contract_name", "org_type", "org_marketing"]),
        on=["contractid", "year"],
        how="left"
    )
    .merge(
        ma_penetration_data.drop(columns=["ssa"]).rename(columns={"state": "state_long", "county": "county_long"}),
        on=["fips", "year"],
        how="left"
    )
)

# Calculate Star_Rating
final_data["Star_Rating"] = np.where(
    final_data["partd"] == "No",
    final_data["partc_score"],
    np.where(
        final_data["partd"] == "Yes",
        np.where(
            final_data["partcd_score"].isna(),
            final_data["partc_score"],
            final_data["partcd_score"]
        ),
        np.nan
    )
)

# Get final state name per state
final_state = (
    final_data
    .sort_values("year")  # ensure correct ordering
    .groupby("state", as_index=False)
    .agg(state_name=("state_long", lambda x: x.dropna().iloc[-1] if not x.dropna().empty else np.nan))
)
final_data = final_data.merge(final_state, on="state", how="left")

# # Merge with plan premiums
# final_data = final_data.merge(
#     plan_premiums,
#     how="left",
#     left_on=["contractid", "planid", "state_name", "county", "year"],
#     right_on=["contractid", "planid", "state", "county", "year"]
# )

# # Merge with risk rebate data (dropping extra columns first)
# risk_rebate_cleaned = risk_rebate_final.drop(columns=["contract_name", "plan_type"], errors="ignore")

# final_data = final_data.merge(
#     risk_rebate_cleaned,
#     how="left",
#     on=["contractid", "planid", "year"]
# )

# Merge with benchmark data
final_data = final_data.merge(
    benchmark_final,
    how="left",
    on=["ssa", "year"]
)

# Calculate relevant benchmark rate based on star rating
conditions = [
    final_data["year"] < 2012,
    (final_data["year"].between(2012, 2014)) & (final_data["Star_Rating"] == 5),
    (final_data["year"].between(2012, 2014)) & (final_data["Star_Rating"] == 4.5),
    (final_data["year"].between(2012, 2014)) & (final_data["Star_Rating"] == 4),
    (final_data["year"].between(2012, 2014)) & (final_data["Star_Rating"] == 3.5),
    (final_data["year"].between(2012, 2014)) & (final_data["Star_Rating"] == 3),
    (final_data["year"].between(2012, 2014)) & (final_data["Star_Rating"] < 3),
    (final_data["year"].between(2012, 2014)) & (final_data["Star_Rating"].isna()),
    (final_data["year"] >= 2015) & (final_data["Star_Rating"] >= 4),
    (final_data["year"] >= 2015) & (final_data["Star_Rating"] < 4),
    (final_data["year"] >= 2015) & (final_data["Star_Rating"].isna())
]

choices = [
    final_data["risk_ab"],
    final_data["risk_star5"],
    final_data["risk_star45"],
    final_data["risk_star4"],
    final_data["risk_star35"],
    final_data["risk_star3"],
    final_data["risk_star25"],
    final_data["risk_star35"],
    final_data["risk_bonus5"],
    final_data["risk_bonus0"],
    final_data["risk_bonus35"]
]

final_data["ma_rate"] = np.select(conditions, choices, default=np.nan)
final_data["ma_rate"] = pd.to_numeric(final_data["ma_rate"], errors="coerce")





# Save the final dataset
final_data.to_csv("data/output/final_ma_data.csv", index=False)