import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load data
df = pd.read_csv('data/output/final_ma_data.csv')

# Quick look at the columns
print(df.columns)

# ----------------------------
# 1. Remove SNPs, 800-series, and Rx-only plans (no Part C)
# ----------------------------
df_cleaned = df[
    (df['snp'] == 0) & 
    (~df['contractid'].str.startswith('800')) & 
    (df['plan_type'].str.contains('Part C', na=False))
]

# ----------------------------
# 2. Box and whisker plot of plan counts per county per year
# ----------------------------
# Count plans by county and year
plan_counts = df_cleaned.groupby(['year', 'county'])['planid'].nunique().reset_index(name='plan_count')

# Plot
plt.figure(figsize=(12, 6))
sns.boxplot(x='year', y='plan_count', data=plan_counts)
plt.title('Distribution of Plan Counts by County Over Time')
plt.ylabel('Number of Plans')
plt.xlabel('Year')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# ----------------------------
# 3. Bar graphs of star ratings in 2010, 2012, 2015
# ----------------------------
for year in [2010, 2012, 2015]:
    plt.figure(figsize=(8, 5))
    sns.countplot(data=df_cleaned[df_cleaned['year'] == year], x='star_rating', order=sorted(df_cleaned['star_rating'].dropna().unique()))
    plt.title(f'Distribution of Star Ratings in {year}')
    plt.xlabel('Star Rating')
    plt.ylabel('Number of Plans')
    plt.tight_layout()
    plt.show()

# ----------------------------
# 4. Plot average benchmark payment over time
# ----------------------------
benchmark_avg = df_cleaned.groupby('year')['benchmark'].mean().reset_index()

plt.figure(figsize=(10, 5))
sns.lineplot(data=benchmark_avg, x='year', y='benchmark', marker='o')
plt.title('Average Benchmark Payment Over Time')
plt.xlabel('Year')
plt.ylabel('Average Benchmark Payment ($)')
plt.grid(True)
plt.tight_layout()
plt.show()

# Print change in benchmark payment
benchmark_change = benchmark_avg.iloc[-1]['benchmark'] - benchmark_avg.iloc[0]['benchmark']
print(f"Average benchmark payment increased by ${benchmark_change:.2f} from 2010 to 2015.")

# ----------------------------
# 5. Plot average MA share over time and compare to benchmark
# ----------------------------
# Compute average share
share_avg = df_cleaned.groupby('year')['ma_share'].mean().reset_index()

# Plot MA share over time
plt.figure(figsize=(10, 5))
sns.lineplot(data=share_avg, x='year', y='ma_share', marker='o', color='green')
plt.title('Average Medicare Advantage Share Over Time')
plt.xlabel('Year')
plt.ylabel('Average MA Share (%)')
plt.grid(True)
plt.tight_layout()
plt.show()

# Merge with benchmark for correlation
merged = pd.merge(benchmark_avg, share_avg, on='year')
correlation = merged['benchmark'].corr(merged['ma_share'])
print(f"Correlation between benchmark payments and MA share: {correlation:.3f}")
