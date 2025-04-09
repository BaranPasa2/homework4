import pandas as pd
import numpy as np
import seaborn as sns
from rdd import rdd
import matplotlib.pyplot as plt

df_2010 = pd.read_csv('data/output/final_ma_data.csv')
#set the dataset to only show year 2010
df_2010 = df_2010[df_2010['year'] == 2010].copy()

# pd.options.display.max_columns = None
# pd.options.display.max_rows = None
# print(df_2010.columns.tolist())

# 5. 
rating_columns = [
    'breastcancer_screen', 'rectalcancer_screen', 'cv_cholscreen', 'diabetes_cholscreen',
    'glaucoma_test', 'monitoring', 'flu_vaccine', 'pn_vaccine', 'physical_health',
    'mental_health', 'osteo_test', 'physical_monitor', 'primaryaccess',
    'hospital_followup', 'depression_followup', 'nodelays', 'carequickly',
    'overallrating_care', 'overallrating_plan', 'calltime',
    'doctor_communicate', 'customer_service', 'osteo_manage',
    'diabetes_eye', 'diabetes_kidney', 'diabetes_bloodsugar',
    'diabetes_chol', 'antidepressant', 'bloodpressure', 'ra_manage',
    'copd_test', 'betablocker', 'bladder', 'falling', 'appeals_timely',
    'appeals_review'
]

# Create new column with the row-wise average
df_2010 = df_2010.dropna(subset=['partc_score', 'avg_enrollment'])
df_2010['raw_rating'] = df_2010[rating_columns].mean(axis=1, skipna=True)

# Round raw_rating to the nearest 0.5
df_2010['rounded_rating'] = df_2010['raw_rating'].round(1)

# Filter only ratings from 3.0 to 5.0 (rounded to nearest 0.5)
valid_ratings = [3.0, 3.5, 4.0, 4.5, 5.0]
rating_counts = df_2010[df_2010['rounded_rating'].isin(valid_ratings)]['rounded_rating'].value_counts().sort_index()

# Display as a table
rating_table = rating_counts.reset_index()
rating_table.columns = ['Star Rating', 'Number of Plans']

# Show the result
print(rating_table)


# 6.

# # Ensure necessary columns are numeric
# df_2010['partc_score'] = pd.to_numeric(df_2010['partc_score'], errors='coerce')
# df_2010['avg_enrollment'] = pd.to_numeric(df_2010['avg_enrollment'], errors='coerce')

# # Define cutoffs and bandwidth
# cutoffs = [3.0, 3.5]
# bandwidth = 0.125

# # Store results here
# results = []

# # Loop over cutoffs and run RDD
# for cutoff in cutoffs:
#     # Subset data within bandwidth
#     data_rdd = df_2010[
#         (df_2010['raw_rating'] >= cutoff - bandwidth) &
#         (df_2010['raw_rating'] <= cutoff + bandwidth)
#     ][['raw_rating', 'avg_enrollment']].dropna()

#     # Rename columns for rdd package

#     # Fit RDD model
#     model = rdd.rdd(data_rdd, 'partc_score', 'avg_enrollment', cut=cutoff)
#     result = model.fit()

#     # Extract results
#     results.append({
#         'Cutoff': cutoff,
#         'RD_Estimate': result.params['raw_rating'],
#         'Std_Error': result.bse['raw_rating'],
#         'Num_Observations': result.nobs
#     })

# # Display summary table
# summary_df = pd.DataFrame(results)
# print(summary_df)
