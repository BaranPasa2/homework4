##############################################################################
## Read in MA star rating data
##############################################################################

import pandas as pd

# Load variable list from pickle
import pickle
with open("data/output/rating_variables.pkl", "rb") as f:
    rating_vars = pickle.load(f)

# 2008 STAR RATING DATA
ma_path_2008a = "data/input/ratingsData/2008_Plan_Ratings/2008 Plan Ratings Spring Release (04_2008)/2008_Part_D_Report_Card_Master_Table_2008_03_14_star.csv"
ma_path_2008b = "data/input/ratingsData/2008_Plan_Ratings/2008 Plan Ratings Spring Release (04_2008)/2008_Part_D_Report_Card_Master_Table_2008_03_14_domain.csv"

# Read first file using rating variables from 2008
star_data_2008a = pd.read_csv(ma_path_2008a, skiprows=4, names=rating_vars["2008"], encoding='latin1')

# Read second file with fixed column names
domain_cols = [
    "contractid", "contract_name", "healthy", "getting_care",
    "timely_care", "chronic", "appeal", "new_contract"
]
star_data_2008b = pd.read_csv(ma_path_2008b, skiprows=2, names=domain_cols, encoding='latin1')

# Clean and keep relevant variables
star_data_2008b["new_contract"] = star_data_2008b["new_contract"].fillna(0)
star_data_2008b = star_data_2008b[["contractid", "new_contract"]]

# Merge and finalize
star_data_2008a = star_data_2008a.drop(columns=["new_contract"], errors="ignore")
star_data_2008 = pd.merge(star_data_2008a, star_data_2008b, on="contractid", how="left")
star_data_2008["year"] = 2008



# 2009 STAR RATING DATA
ma_path_2009a = "data/input/ma-star-ratings/2009/2009_Part_C_Report_Card_Master_Table_2009_11_30_stars.csv"
star_data_2009a = pd.read_csv(ma_path_2009a, skiprows=4, names=rating_vars["2009"], encoding='latin1')

# Replace star labels with numeric strings
star_recode = {
    "1 out of 5 stars": "1",
    "2 out of 5 stars": "2",
    "3 out of 5 stars": "3",
    "4 out of 5 stars": "4",
    "5 stars": "5"
}
star_data_2009a = star_data_2009a.applymap(lambda x: star_recode.get(x, x))

# Convert all columns to numeric except ID and org metadata
cols_to_convert = star_data_2009a.columns.difference(["contractid", "org_type", "contract_name", "org_marketing"])
star_data_2009a[cols_to_convert] = star_data_2009a[cols_to_convert].apply(pd.to_numeric, errors="coerce")

# Read summary data
ma_path_2009b = "data/input/ma-star-ratings/2009/2009_Part_C_Report_Card_Master_Table_2009_11_30_summary.csv"
star_data_2009b = pd.read_csv(ma_path_2009b, skiprows=2, names=[
    "contractid", "org_type", "contract_name", "org_marketing", "partc_score"
], encoding='latin1')

# Flag new contracts and convert star scores
star_data_2009b["new_contract"] = (star_data_2009b["partc_score"] == "Plan too new to be measured").astype(int)

score_recode = {
    "1 out of 5 stars": "1", "1.5 out of 5 stars": "1.5",
    "2 out of 5 stars": "2", "2.5 out of 5 stars": "2.5",
    "3 out of 5 stars": "3", "3.5 out of 5 stars": "3.5",
    "4 out of 5 stars": "4", "4.5 out of 5 stars": "4.5",
    "5 stars": "5"
}
star_data_2009b["partc_score"] = star_data_2009b["partc_score"].replace(score_recode)
star_data_2009b["partc_score"] = pd.to_numeric(star_data_2009b["partc_score"], errors="coerce")
star_data_2009b = star_data_2009b[["contractid", "new_contract", "partc_score"]]

# Merge and finalize
star_data_2009 = pd.merge(star_data_2009a, star_data_2009b, on="contractid", how="left")
star_data_2009["year"] = 2009



# 2010 STAR RATING DATA
ma_path_2010a = "data/input/ma-star-ratings/2010/2010_Part_C_Report_Card_Master_Table_2009_11_30_domain.csv"
star_data_2010a = pd.read_csv(ma_path_2010a, skiprows=4, names=rating_vars["2010"], encoding='latin1')

# Replace star labels with numeric
star_recode = {
    "1 out of 5 stars": "1",
    "2 out of 5 stars": "2",
    "3 out of 5 stars": "3",
    "4 out of 5 stars": "4",
    "5 stars": "5"
}
star_data_2010a = star_data_2010a.applymap(lambda x: star_recode.get(x, x))

# Convert rating columns to numeric (exclude key metadata columns)
cols_to_keep = ["contractid", "org_type", "contract_name", "org_marketing"]
cols_to_numeric = star_data_2010a.columns.difference(cols_to_keep)
star_data_2010a[cols_to_numeric] = star_data_2010a[cols_to_numeric].apply(pd.to_numeric, errors="coerce")

# Load summary data
ma_path_2010b = "data/input/ma-star-ratings/2010/2010_Part_C_Report_Card_Master_Table_2009_11_30_summary.csv"
star_data_2010b = pd.read_csv(ma_path_2010b, skiprows=2, names=[
    "contractid", "org_type", "contract_name", "org_marketing", "partc_score"
], encoding='latin1')

# Process partc_score and create new_contract flag
star_data_2010b["new_contract"] = (star_data_2010b["partc_score"] == "Plan too new to be measured").astype(int)

score_recode = {
    "1 out of 5 stars": "1", "1.5 out of 5 stars": "1.5",
    "2 out of 5 stars": "2", "2.5 out of 5 stars": "2.5",
    "3 out of 5 stars": "3", "3.5 out of 5 stars": "3.5",
    "4 out of 5 stars": "4", "4.5 out of 5 stars": "4.5",
    "5 stars": "5"
}
star_data_2010b["partc_score"] = star_data_2010b["partc_score"].replace(score_recode)
star_data_2010b["partc_score"] = pd.to_numeric(star_data_2010b["partc_score"], errors="coerce")
star_data_2010b = star_data_2010b[["contractid", "new_contract", "partc_score"]]

# Merge and finalize
star_data_2010 = pd.merge(star_data_2010a, star_data_2010b, on="contractid", how="left")
star_data_2010["year"] = 2010




# 2011 STAR RATING DATA
ma_path_2011a = "data/input/ma-star-ratings/2011/2011_Part_C_Report_Card_Master_Table_2011_04_20_star.csv"
star_data_2011a = pd.read_csv(ma_path_2011a, skiprows=5, names=rating_vars["2011"], encoding='latin1')

# Replace star labels with numeric
star_recode_2011 = {
    "1 stars": "1",
    "2 stars": "2",
    "3 stars": "3",
    "4 stars": "4",
    "5 stars": "5"
}
star_data_2011a = star_data_2011a.applymap(lambda x: star_recode_2011.get(x, x))

# Convert to numeric excluding identifying variables
cols_to_keep_2011 = ["contractid", "org_type", "contract_name", "org_marketing"]
cols_to_numeric_2011 = star_data_2011a.columns.difference(cols_to_keep_2011)
star_data_2011a[cols_to_numeric_2011] = star_data_2011a[cols_to_numeric_2011].apply(pd.to_numeric, errors="coerce")

# Read summary data
ma_path_2011b = "data/input/ma-star-ratings/2011/2011_Part_C_Report_Card_Master_Table_2011_04_20_summary.csv"
summary_cols_2011 = ["contractid", "org_type", "contract_name", "org_marketing",
                     "partc_lowstar", "partc_score", "partcd_score"]
star_data_2011b = pd.read_csv(ma_path_2011b, skiprows=2, names=summary_cols_2011, encoding='latin1')

# Flag new contracts and convert scores
star_data_2011b["new_contract"] = (
    (star_data_2011b["partc_score"] == "Plan too new to be measured") |
    (star_data_2011b["partcd_score"] == "Plan too new to be measured")
).astype(int)

score_recode_detailed = {
    "1 out of 5 stars": "1", "1.5 out of 5 stars": "1.5",
    "2 out of 5 stars": "2", "2.5 out of 5 stars": "2.5",
    "3 out of 5 stars": "3", "3.5 out of 5 stars": "3.5",
    "4 out of 5 stars": "4", "4.5 out of 5 stars": "4.5",
    "5 stars": "5"
}

star_data_2011b["partc_score"] = star_data_2011b["partc_score"].replace(score_recode_detailed)
star_data_2011b["partcd_score"] = star_data_2011b["partcd_score"].replace(score_recode_detailed)
star_data_2011b["partc_score"] = pd.to_numeric(star_data_2011b["partc_score"], errors="coerce")
star_data_2011b["partcd_score"] = pd.to_numeric(star_data_2011b["partcd_score"], errors="coerce")
star_data_2011b["low_score"] = (star_data_2011b["partc_lowstar"] == "Yes").astype(int)

star_data_2011b = star_data_2011b[["contractid", "new_contract", "low_score", "partc_score", "partcd_score"]]

# Merge and finalize
star_data_2011 = pd.merge(star_data_2011a, star_data_2011b, on="contractid", how="left")
star_data_2011["year"] = 2011



# 2012 STAR RATING DATA
ma_path_2012a = "data/input/ma-star-ratings/Part C 2012 Fall/2012_Part_C_Report_Card_Master_Table_2011_11_01_Star.csv"
star_data_2012a = pd.read_csv(ma_path_2012a, skiprows=5, names=rating_vars["2012"], encoding='latin1')

# Convert columns to numeric except IDs and org metadata
id_vars_2012 = ["contractid", "org_type", "contract_name", "org_marketing", "org_parent"]
cols_to_numeric_2012 = star_data_2012a.columns.difference(id_vars_2012)
star_data_2012a[cols_to_numeric_2012] = star_data_2012a[cols_to_numeric_2012].apply(pd.to_numeric, errors="coerce")

# Read summary data
ma_path_2012b = "data/input/ma-star-ratings/Part C 2012 Fall/2012_Part_C_Report_Card_Master_Table_2011_11_01_Summary.csv"
summary_cols_2012 = [
    "contractid", "org_type", "org_parent", "org_marketing",
    "partc_score", "partc_lowscore", "partc_highscore",
    "partcd_score", "partcd_lowscore", "partcd_highscore"
]
star_data_2012b = pd.read_csv(ma_path_2012b, skiprows=2, names=summary_cols_2012, encoding='latin1')

# Recode new contract and convert score columns
star_data_2012b["new_contract"] = (
    (star_data_2012b["partc_score"] == "Plan too new to be measured") |
    (star_data_2012b["partcd_score"] == "Plan too new to be measured")
).astype(int)

score_recode_2012 = {
    "1 out of 5 stars": "1", "1.5 out of 5 stars": "1.5",
    "2 out of 5 stars": "2", "2.5 out of 5 stars": "2.5",
    "3 out of 5 stars": "3", "3.5 out of 5 stars": "3.5",
    "4 out of 5 stars": "4", "4.5 out of 5 stars": "4.5",
    "5 stars": "5"
}

star_data_2012b["partc_score"] = star_data_2012b["partc_score"].replace(score_recode_2012)
star_data_2012b["partcd_score"] = star_data_2012b["partcd_score"].replace(score_recode_2012)
star_data_2012b["partc_score"] = pd.to_numeric(star_data_2012b["partc_score"], errors="coerce")
star_data_2012b["partcd_score"] = pd.to_numeric(star_data_2012b["partcd_score"], errors="coerce")
star_data_2012b["low_score"] = (star_data_2012b["partc_lowscore"] == "Yes").astype(int)

star_data_2012b = star_data_2012b[["contractid", "new_contract", "low_score", "partc_score", "partcd_score"]]

# Merge and finalize
star_data_2012 = pd.merge(star_data_2012a, star_data_2012b, on="contractid", how="left")
star_data_2012["year"] = 2012



# 2013 STAR RATING DATA
ma_path_2013a = "data/input/ma-star-ratings/Part C 2013 Fall/2013_Part_C_Report_Card_Master_Table_2012_10_17_Star.csv"
ma_path_2013b = "data/input/ma-star-ratings/Part C 2013 Fall/2013_Part_C_Report_Card_Master_Table_2012_10_17_Summary.csv"

# Read the main dataset
star_data_2013a = pd.read_csv(ma_path_2013a, skiprows=4, names=rating_vars["2013"], encoding='latin1')

# Convert all but ID/label columns to numeric
cols_to_exclude = ["contractid", "org_type", "contract_name", "org_marketing", "org_parent"]
cols_to_numeric = star_data_2013a.columns.difference(cols_to_exclude)
star_data_2013a[cols_to_numeric] = star_data_2013a[cols_to_numeric].apply(pd.to_numeric, errors="coerce")

# Read the summary dataset
summary_cols_2013 = [
    "contractid", "org_type", "org_marketing", "contract_name", "org_parent",
    "partc_score", "partc_lowscore", "partc_highscore",
    "partcd_score", "partcd_lowscore", "partcd_highscore"
]
star_data_2013b = pd.read_csv(ma_path_2013b, skiprows=2, names=summary_cols_2013, encoding='latin1')

# Flag new contracts and clean ratings
star_data_2013b["new_contract"] = (
    (star_data_2013b["partc_score"] == "Plan too new to be measured") |
    (star_data_2013b["partcd_score"] == "Plan too new to be measured")
).astype(int)

star_data_2013b["partc_score"] = pd.to_numeric(star_data_2013b["partc_score"], errors="coerce")
star_data_2013b["partcd_score"] = pd.to_numeric(star_data_2013b["partcd_score"], errors="coerce")
star_data_2013b["low_score"] = (star_data_2013b["partc_lowscore"] == "Yes").astype(int)

star_data_2013b = star_data_2013b[["contractid", "new_contract", "low_score", "partc_score", "partcd_score"]]

# Merge datasets
star_data_2013 = pd.merge(star_data_2013a, star_data_2013b, on="contractid", how="left")
star_data_2013["year"] = 2013


# 2014 STAR RATING DATA
ma_path_2014a = "data/input/ma-star-ratings/Part C 2014 Fall/2014_Part_C_Report_Card_Master_Table_2013_10_17_stars.csv"
ma_path_2014b = "data/input/ma-star-ratings/Part C 2014 Fall/2014_Part_C_Report_Card_Master_Table_2013_10_17_summary.csv"

# Read the stars data
star_data_2014a = pd.read_csv(ma_path_2014a, skiprows=3, names=rating_vars["2014"], encoding='latin1')

# Convert to numeric where appropriate
exclude_cols_2014 = ["contractid", "org_type", "contract_name", "org_marketing", "org_parent"]
numeric_cols_2014 = star_data_2014a.columns.difference(exclude_cols_2014)
star_data_2014a[numeric_cols_2014] = star_data_2014a[numeric_cols_2014].apply(pd.to_numeric, errors="coerce")

# Read the summary file using pandas read_csv and select columns 1 through 9
star_data_2014b = pd.read_csv(ma_path_2014b, skiprows=2, usecols=range(9), names=[
    "contractid", "org_type", "org_marketing", "contract_name",
    "org_parent", "snp", "sanction", "partc_score", "partcd_score"
], encoding='latin1')

# Process summary data
star_data_2014b["new_contract"] = (
    (star_data_2014b["partc_score"] == "Plan too new to be measured") |
    (star_data_2014b["partcd_score"] == "Plan too new to be measured")
).astype(int)

star_data_2014b["partc_score"] = pd.to_numeric(star_data_2014b["partc_score"], errors="coerce")
star_data_2014b["partcd_score"] = pd.to_numeric(star_data_2014b["partcd_score"], errors="coerce")

star_data_2014b = star_data_2014b[["contractid", "new_contract", "partc_score", "partcd_score"]]

# Merge and finalize
star_data_2014 = pd.merge(star_data_2014a, star_data_2014b, on="contractid", how="left")
star_data_2014["year"] = 2014



# 2015 STAR RATING DATA
ma_path_2015a = "data/input/ma-star-ratings/2015 Fall/2015_Report_Card_Master_Table_2014_10_03_stars.csv"
ma_path_2015b = "data/input/ma-star-ratings/2015 Fall/2015_Report_Card_Master_Table_2014_10_03_summary.csv"

# Read stars data
star_data_2015a = pd.read_csv(ma_path_2015a, skiprows=4, usecols=range(38), names=rating_vars["2015"], encoding='latin1')

# Convert columns to numeric (excluding identifiers)
exclude_cols_2015 = ["contractid", "org_type", "contract_name", "org_marketing", "org_parent"]
numeric_cols_2015 = star_data_2015a.columns.difference(exclude_cols_2015)
star_data_2015a[numeric_cols_2015] = star_data_2015a[numeric_cols_2015].apply(pd.to_numeric, errors="coerce")

# Read summary data
star_data_2015b = pd.read_csv(ma_path_2015b, skiprows=2, usecols=range(10), names=[
    "contractid", "org_type", "org_marketing", "contract_name", "org_parent",
    "snp", "sanction", "partc_score", "partdscore", "partcd_score"
], encoding='latin1')

# Process summary fields
star_data_2015b["new_contract"] = (
    (star_data_2015b["partc_score"] == "Plan too new to be measured") |
    (star_data_2015b["partcd_score"] == "Plan too new to be measured")
).astype(int)

star_data_2015b["partc_score"] = pd.to_numeric(star_data_2015b["partc_score"], errors="coerce")
star_data_2015b["partcd_score"] = pd.to_numeric(star_data_2015b["partcd_score"], errors="coerce")
star_data_2015b = star_data_2015b[["contractid", "new_contract", "partc_score", "partcd_score"]]

# Merge and tag year
star_data_2015 = pd.merge(star_data_2015a, star_data_2015b, on="contractid", how="left")
star_data_2015["year"] = 2015



# Combine all star ratings
star_ratings = pd.concat([
    star_data_2008, star_data_2009, star_data_2010, star_data_2011,
    star_data_2012, star_data_2013, star_data_2014, star_data_2015
], ignore_index=True)

# Ensure new_contract column exists
if "new_contract" not in star_ratings.columns:
    star_ratings["new_contract"] = pd.NA

# Save csv
star_ratings.to_csv("data/output/star_ratings.csv", index=False)