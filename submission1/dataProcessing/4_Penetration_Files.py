##############################################################################
## Read in market pentration data
##############################################################################

import pandas as pd

# Define month lists
monthlists = {
    year: [f"{m:02d}" for m in range(1, 13)]
    for year in range(2010, 2016)
}

# Initialize full DataFrame
all_years = []

# Process each year and month
for year, months in monthlists.items():
    monthly_data = []

    for month in months:
        ma_path = f"data/input/monthly-ma-state-and-county-penetration/State_County_Penetration_MA_{year}_{month}.csv"

        df = pd.read_csv(
            ma_path,
            skiprows=1,
            names=[
                "state", "county", "fips_state", "fips_cnty", "fips",
                "ssa_state", "ssa_cnty", "ssa", "eligibles", "enrolled", "penetration"
            ],
            na_values="*",
        )
        df["eligibles"] = df["eligibles"].replace(",", "", regex=True).astype(float)
        df["enrolled"] = df["enrolled"].replace(",", "", regex=True).astype(float)
        df["penetration"] = df["penetration"].replace("%", "", regex=True).astype(float) / 100.0
        df["fips"] = pd.to_numeric(df["fips"], errors='coerce')  # Ensure fips is numeric

        df["month"] = month
        df["year"] = year
        monthly_data.append(df)

    
    ma_penetration = pd.concat(monthly_data, ignore_index=True)

    # Fill in missing FIPS within state-county groups
    ma_penetration["fips"] = ma_penetration.groupby(["state", "county"])["fips"].transform(lambda x: x.ffill().bfill())

    # Collapse to yearly data by averaging
    # Collapse to yearly data
    ma_penetration = (
        ma_penetration
        .groupby(["fips", "state", "county"], as_index=False)
        .agg(
            avg_eligibles=('eligibles', 'mean'),
            sd_eligibles=('eligibles', 'std'),
            min_eligibles=('eligibles', 'min'),
            max_eligibles=('eligibles', 'max'),
            first_eligibles=('eligibles', 'first'),
            last_eligibles=('eligibles', 'last'),
            avg_enrolled=('enrolled', 'mean'),
            sd_enrolled=('enrolled', 'std'),
            min_enrolled=('enrolled', 'min'),
            max_enrolled=('enrolled', 'max'),
            first_enrolled=('enrolled', 'first'),
            last_enrolled=('enrolled', 'last'),
            year=('year', 'last'),
            ssa=('ssa', 'first')
        )
    )

    all_years.append(ma_penetration)

# Combine all years into final DataFrame
ma_penetration_all = pd.concat(all_years, ignore_index=True)

# Save final dataset
ma_penetration_all.to_csv("data/output/ma_penetration.csv", index=False)