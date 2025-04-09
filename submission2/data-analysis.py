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
# print(data2010.columns.tolist())
# 5. 


# Load your dataset (make sure it's in the same directory or provide full path)


# Filter for 2010

# Ensure numeric values
df_2010['partcd_score'] = pd.to_numeric(df_2010['partc_score'], errors='coerce')
df_2010['Star_Rating'] = pd.to_numeric(df_2010['Star_Rating'], errors='coerce')

# Define thresholds that separate star levels
thresholds = {
    5.0: 4.75,
    4.5: 4.25,
    4.0: 3.75,
    3.5: 3.25,
    3.0: 2.75
}

# Count how many were rounded up close to cutoff
rounded_up_counts = []
for star, cutoff in thresholds.items():
    close_to_cutoff = (
        (df_2010['partcd_score'] >= cutoff) &
        (df_2010['partcd_score'] < cutoff + 0.01) &
        (df_2010['Star_Rating'] == star)
    )
    rounded_up_counts.append({
        'Rounded_Into_Star': star,
        'Num_Plans_Close_To_Cutoff': close_to_cutoff.sum()
    })

# Create and print the results table
rounded_up_table = pd.DataFrame(rounded_up_counts)
print(rounded_up_table)
print(sorted(df_2010['partc_score'].dropna().unique()))
import matplotlib.pyplot as plt

df_2010['partc_score'].hist(bins=50)
plt.xlabel("Part C Score")
plt.ylabel("Number of Plans")
plt.title("Distribution of Running Variable in 2010")
plt.show()

