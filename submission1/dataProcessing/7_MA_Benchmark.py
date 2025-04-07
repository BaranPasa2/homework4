##############################################################################
## Read in MA Benchmark Rates (apply to each county)
##############################################################################

import pandas as pd

# Define file paths
bench_paths = {
    2007: "data/input/rateBooks/ratebook2007/countyrate2007.csv",
    2008: "data/input/rateBooks/ratebook2008/countyrate2008.csv",
    2009: "data/input/rateBooks/ratebook2009/countyrate2009.csv",
    2010: "data/input/rateBooks/ratebook2010/CountyRate2010.csv",
    2011: "data/input/rateBooks/ratebook2011/CountyRate2011.csv",
    2012: "data/input/rateBooks/ratebook2012/CountyRate2012.csv",
    2013: "data/input/rateBooks/RateBook2013/CountyRate2013.csv",
    2014: "data/input/rateBooks/ratebook2014/CountyRate2014.csv",
    2015: "data/input/rateBooks/RateBook2015/CSV/CountyRate2015.csv"
}

# Drop rows at beginning for each file
drops = {
    2007: 9, 2008: 10, 2009: 9, 2010: 9, 2011: 11,
    2012: 8, 2013: 4, 2014: 2, 2015: 3
}

benchmark_frames = []

# 2007–2011
for year in range(2007, 2012):
    path = bench_paths[year]
    skip = drops[year]
    cols = [
        "ssa", "state", "county_name", "aged_parta", "aged_partb",
        "disabled_parta", "disabled_partb", "esrd_ab", "risk_ab"
    ]
    df = pd.read_csv(path, skiprows=skip, names=cols)
    df = df[["ssa", "aged_parta", "aged_partb", "risk_ab"]]
    df = df.assign(
        risk_star5=None, risk_star45=None, risk_star4=None,
        risk_star35=None, risk_star3=None, risk_star25=None,
        risk_bonus5=None, risk_bonus35=None, risk_bonus0=None,
        year=year
    )
    benchmark_frames.append(df)

# 2012–2014
for year in range(2012, 2015):
    path = bench_paths[year]
    skip = drops[year]
    cols = [
        "ssa", "state", "county_name", "risk_star5", "risk_star45",
        "risk_star4", "risk_star35", "risk_star3", "risk_star25", "esrd_ab"
    ]
    df = pd.read_csv(path, skiprows=skip, names=cols)
    df = df[["ssa", "risk_star5", "risk_star45", "risk_star4",
             "risk_star35", "risk_star3", "risk_star25"]]
    df = df.assign(
        aged_parta=None, aged_partb=None, risk_ab=None,
        risk_bonus5=None, risk_bonus35=None, risk_bonus0=None,
        year=year
    )
    benchmark_frames.append(df)

# 2015
path_2015 = bench_paths[2015]
skip_2015 = drops[2015]
cols_2015 = [
    "ssa", "state", "county_name", "risk_bonus5",
    "risk_bonus35", "risk_bonus0", "esrd_ab"
]
df_2015 = pd.read_csv(path_2015, skiprows=skip_2015, names=cols_2015, na_values="#N/A")
df_2015 = df_2015[["ssa", "risk_bonus5", "risk_bonus35", "risk_bonus0"]]
df_2015 = df_2015.assign(
    risk_star5=None, risk_star45=None, risk_star4=None,
    risk_star35=None, risk_star3=None, risk_star25=None,
    aged_parta=None, aged_partb=None, risk_ab=None,
    year=2015
)
benchmark_frames.append(df_2015)

# Combine all
benchmark_final = pd.concat(benchmark_frames, ignore_index=True)

# Convert ssa to numeric
benchmark_final["ssa"] = pd.to_numeric(benchmark_final["ssa"], errors='coerce')

# Save dataset
benchmark_final.to_csv("data/output/ma_benchmarks.csv", index=False)