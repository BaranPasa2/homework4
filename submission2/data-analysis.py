import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from pyfixest.estimation import feols


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
def run_rd(df, cutoff, bandwidth):
    df = df.copy()
    df['treat'] = (df['raw_rating'] >= cutoff).astype(int)
    
    # Keep only observations within the bandwidth window
    df_band = df[(df['raw_rating'] >= cutoff - bandwidth) & (df['raw_rating'] <= cutoff + bandwidth)].copy()
    
    # Center the running variable
    df_band['running'] = df_band['raw_rating'] - cutoff

    # Run the RD regression
    model = feols("avg_enrollment ~ treat + running", data=df_band)
    
    return model

# Run RD at 3.0 and 3.5 cutoffs
model_3 = run_rd(df_2010, cutoff=3.0, bandwidth=0.125)
model_3_5 = run_rd(df_2010, cutoff=3.5, bandwidth=0.125)

# Extract results properly by calling the methods
coef_3 = model_3.coef()
se_3 = model_3.se()
pval_3 = model_3.pvalue()

coef_3_5 = model_3_5.coef()
se_3_5 = model_3_5.se()
pval_3_5 = model_3_5.pvalue()

# Summarize in a table
results = pd.DataFrame({
    'Cutoff': ['3.0 Stars', '3.5 Stars'],
    'Estimate (Treatment Effect)': [coef_3['treat'], coef_3_5['treat']],
    'Std. Error': [se_3['treat'], se_3_5['treat']],
    'p-Value': [pval_3['treat'], pval_3_5['treat']]
})

print(results)
