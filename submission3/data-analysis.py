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

# Define valid ratings
# Define function to round to nearest 0.5
def round_half(x):
    return np.round(x * 2) / 2

# Apply the rounding
df_2010['rounded_rating'] = df_2010['raw_rating'].apply(round_half)

# Define valid ratings
valid_ratings = [3.0, 3.5, 4.0, 4.5, 5.0]

# Count and reindex
rating_counts = (
    df_2010
    .loc[df_2010['rounded_rating'].isin(valid_ratings), 'rounded_rating']
    .value_counts()
    .reindex(valid_ratings, fill_value=0)
    .sort_index()
)


# Display as a table
rating_table = rating_counts.reset_index()
rating_table.columns = ['Star Rating', 'Number of Plans']
plt.figure(figsize=(10, 6))

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

#7. 
# Run RD at 3.0 and 3.5 cutoffs with different bandwidths
model_3_01 = run_rd(df_2010, cutoff=3.0, bandwidth=0.1)
model_3_012 = run_rd(df_2010, cutoff=3.0, bandwidth=0.12)
model_3_013 = run_rd(df_2010, cutoff=3.0, bandwidth=0.13)
model_3_014 = run_rd(df_2010, cutoff=3.0, bandwidth=0.14)
model_3_015 = run_rd(df_2010, cutoff=3.0, bandwidth=0.15)

model_35_01 = run_rd(df_2010, cutoff=3.5, bandwidth=0.1)
model_35_012 = run_rd(df_2010, cutoff=3.5, bandwidth=0.12)
model_35_013 = run_rd(df_2010, cutoff=3.5, bandwidth=0.13)
model_35_014 = run_rd(df_2010, cutoff=3.5, bandwidth=0.14)
model_35_015 = run_rd(df_2010, cutoff=3.5, bandwidth=0.15)


# Store all models with identifiers
models = {
    (3.0, 0.10): model_3_01,
    (3.0, 0.12): model_3_012,
    (3.0, 0.13): model_3_013,
    (3.0, 0.14): model_3_014,
    (3.0, 0.15): model_3_015,
    (3.5, 0.10): model_35_01,
    (3.5, 0.12): model_35_012,
    (3.5, 0.13): model_35_013,
    (3.5, 0.14): model_35_014,
    (3.5, 0.15): model_35_015,
}

# Collect results
rows = []
for (cutoff, bw), model in models.items():
    coef = model.coef()
    se = model.se()
    pval = model.pvalue()
    rows.append({
        'Cutoff': cutoff,
        'Bandwidth': bw,
        'Estimate': coef['treat'],
        'Std. Error': se['treat'],
        'p-Value': pval['treat']
    })

# Create DataFrame
results_df = pd.DataFrame(rows)
print(results_df)

#8.

# Plot histogram around 3.0
plt.figure(figsize=(12, 5))

# First plot: distribution around 3.0
plt.subplot(1, 2, 1)
sns.histplot(df_2010['raw_rating'], bins=50, kde=False, color='skyblue')
plt.axvline(3.0, color='red', linestyle='--', label='3.0 Cutoff')
plt.title('Distribution of Raw Ratings around 3.0')
plt.xlabel('Raw Rating')
plt.ylabel('Number of Plans')
plt.xlim(2.5, 3.5)
plt.legend()

# Second plot: distribution around 3.5
plt.subplot(1, 2, 2)
sns.histplot(df_2010['raw_rating'], bins=50, kde=False, color='lightgreen')
plt.axvline(3.5, color='red', linestyle='--', label='3.5 Cutoff')
plt.title('Distribution of Raw Ratings around 3.5')
plt.xlabel('Raw Rating')
plt.ylabel('Number of Plans')
plt.xlim(3.0, 4.0)
plt.legend()

plt.tight_layout()
plt.show()

#9.

# Step 1: Add binary indicators
df_2010['is_hmo'] = df_2010['plan_type'].str.contains('HMO', case=False, na=False).astype(int)
df_2010['has_partd'] = (df_2010['partd'] == 'Yes').astype(int)

# Step 2: Function to check balance
def covariate_balance(df, cutoff, bandwidth=0.125):
    df = df.copy()
    df_window = df[(df['raw_rating'] >= cutoff - bandwidth) & (df['raw_rating'] <= cutoff + bandwidth)].copy()
    df_window['above_cutoff'] = (df_window['raw_rating'] >= cutoff).astype(int)
    
    balance = df_window.groupby('above_cutoff')[['is_hmo', 'has_partd']].mean().T
    balance.columns = ['Below Cutoff', 'Above Cutoff']
    balance['Difference'] = balance['Above Cutoff'] - balance['Below Cutoff']
    return balance

# Step 3: Get balance tables
balance_3 = covariate_balance(df_2010, cutoff=3.0)
balance_35 = covariate_balance(df_2010, cutoff=3.5)

# Step 4: Combine and label
balance_3['Cutoff'] = '3.0'
balance_35['Cutoff'] = '3.5'
balance_combined = pd.concat([balance_3, balance_35])
balance_combined.reset_index(inplace=True)
balance_combined.rename(columns={'index': 'Characteristic'}, inplace=True)

# Display nicely using pandas
print("\nCovariate Balance Around 3.0 and 3.5 Star Cutoffs:\n")
print(balance_combined.round(3).to_string(index=False))