# Import necessary libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr
import numpy as np

# Load the data
df = pd.read_csv('data/output/final_ma_data.csv')

# First, let's print some information about the data to better understand it
print("Original dataset shape:", df.shape)

# Check data types to understand column formats
print("\nData types:")
print(df.dtypes.head(10))  # Show first 10 data types

# Print first few rows for some key columns to understand the data structure
print("\nSample data for key columns:")
cols_to_check = ['contractid', 'planid', 'plan_type', 'snp', 'partd']
print(df[cols_to_check].head())

# Print unique values for key filtering columns
print("\nUnique values in key columns:")
for col in ['plan_type', 'snp', 'partd']:
    if col in df.columns:
        print(f"\n{col} unique values:")
        print(df[col].value_counts().head(10))  # Show top 10 values

# Now, let's create a step-by-step filtering approach
# Create a copy of the dataframe to work with
filtered_df = df.copy()

# Step 1: Check if we have the columns needed for filtering
print("\nFiltering steps:")
columns_present = []
for col in ['snp', 'planid', 'plan_type', 'partd']:
    columns_present.append(col in df.columns)
    print(f"Column '{col}' present: {col in df.columns}")

# Step 2: Filter out SNPs if the column exists
if 'snp' in df.columns:
    # Check the data type to decide how to filter
    if pd.api.types.is_numeric_dtype(df['snp']):
        filtered_df = filtered_df[filtered_df['snp'] == 0]
        print(f"Filtered SNPs (numeric): {df.shape[0] - filtered_df.shape[0]} rows removed")
    else:
        # If it's not numeric, try to filter based on string values
        filtered_df = filtered_df[~filtered_df['snp'].astype(str).str.contains('1|Yes|Y|True', case=False, na=False)]
        print(f"Filtered SNPs (string): {df.shape[0] - filtered_df.shape[0]} rows removed")
else:
    print("Warning: 'snp' column not found. Skipping SNP filtering.")

# Step 3: Filter out 800-series plans if planid column exists
if 'planid' in df.columns:
    # Check if planid is numeric
    if pd.api.types.is_numeric_dtype(df['planid']):
        filtered_df = filtered_df[~((filtered_df['planid'] >= 800) & (filtered_df['planid'] <= 899))]
        print(f"Filtered 800-series plans (numeric): {df.shape[0] - filtered_df.shape[0]} rows removed")
    else:
        # If it's not numeric, convert to string and check
        filtered_df = filtered_df[~filtered_df['planid'].astype(str).str.match(r'8\d{2}')]
        print(f"Filtered 800-series plans (string): {df.shape[0] - filtered_df.shape[0]} rows removed")
else:
    print("Warning: 'planid' column not found. Skipping 800-series plan filtering.")

# Step 4: Filter out prescription drug only plans
# This is more complex as we need to check multiple conditions
if all(col in df.columns for col in ['plan_type', 'partd']):
    # Assuming that Part D only plans are marked in plan_type or partd
    pdo_mask = (filtered_df['plan_type'].astype(str).str.contains('Prescription Drug|PDP', case=False, na=False))
    if pd.api.types.is_numeric_dtype(df['partd']):
        pdo_mask = pdo_mask | (filtered_df['partd'] == 1 & ~filtered_df['plan_type'].astype(str).str.contains('MA', case=False, na=False))
    
    filtered_df = filtered_df[~pdo_mask]
    print(f"Filtered prescription drug only plans: {df.shape[0] - filtered_df.shape[0]} rows removed")
else:
    print("Warning: Columns needed for prescription drug plan filtering not found. Using alternative approach.")
    # Alternative approach: keep only MA, HMO, PPO, etc. plans
    if 'plan_type' in df.columns:
        ma_mask = filtered_df['plan_type'].astype(str).str.contains('MA|HMO|PPO|Private Fee|Cost Plan|Medical', case=False, na=False)
        filtered_df = filtered_df[ma_mask]
        print(f"Filtered non-MA plans (alternative approach): {df.shape[0] - filtered_df.shape[0]} rows removed")

print("\nFinal filtered dataset shape:", filtered_df.shape)

# Group by county and year to count plans
if all(col in filtered_df.columns for col in ['county', 'year']):
    county_year_plan_counts = filtered_df.groupby(['county', 'year']).size().reset_index(name='plan_count')
    county_year_plan_counts['plan_count'] = np.log(county_year_plan_counts['plan_count'])
    # Set the style for a cleaner look
    sns.set_style("whitegrid")
    plt.figure(figsize=(12, 8))
    
    # Create a more polished box plot
    ax = sns.boxplot(
        x='year', 
        y='plan_count', 
        data=county_year_plan_counts,
        palette="Blues",
        width=0.6,
        fliersize=3,
        linewidth=1.5
    )
    
    # Add a more descriptive title and labels with better formatting
    plt.title('Distribution of Medicare Advantage Plans by County (2010-2015)', 
              fontsize=16, fontweight='bold', pad=20)
    plt.xlabel('Year', fontsize=14, labelpad=10)
    plt.ylabel('Number of Plans Per County', fontsize=14, labelpad=10)
    
    # Improve the tick formatting
    plt.xticks(rotation=0, fontsize=12)
    plt.yticks(fontsize=12)
    
    # Add a light grid only on the y-axis for better readability
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Add a more concise summary of statistics as a table below
    years = sorted(county_year_plan_counts['year'].unique())
    stats_data = []
    
    for year in years:
        year_data = county_year_plan_counts[county_year_plan_counts['year'] == year]['plan_count']
        stats_data.append([
            year,
            f"{year_data.median():.0f}",
            f"{year_data.mean():.1f}",
            f"{year_data.min():.0f}",
            f"{year_data.max():.0f}"
        ])
    
    # Create a nicely formatted table for the statistics
    table = plt.table(
        cellText=stats_data,
        colLabels=['Year', 'Median', 'Mean', 'Min', 'Max'],
        loc='bottom',
        bbox=[0.15, -0.38, 0.7, 0.2],  # [left, bottom, width, height]
        cellLoc='center'
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 1.5)  # Adjust table size for better readability
    
    # Adjust layout to make room for the table
    plt.subplots_adjust(bottom=0.25)
    
    # Save the figure with higher quality
 #   plt.savefig('ma_plan_distribution_by_county.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Additional analysis: Calculate average plan count and changes over time
    yearly_avg = county_year_plan_counts.groupby('year')['plan_count'].agg(['mean', 'median', 'min', 'max']).reset_index()
    print("\nYearly statistics of plan counts per county:")
    print(yearly_avg)
    
    # Calculate percentage of counties with fewer than 5 plans - a potential measure of competition
    low_choice_counties = county_year_plan_counts[county_year_plan_counts['plan_count'] < 5].groupby('year').size()
    total_counties = county_year_plan_counts.groupby('year')['county'].nunique()
    percent_low_choice = (low_choice_counties / total_counties) * 100
    
    print("\nPercentage of counties with fewer than 5 plans by year:")
    print(percent_low_choice)
    
    # Also analyze how many counties have a high number of plans (e.g. more than 15)
    high_choice_counties = county_year_plan_counts[county_year_plan_counts['plan_count'] > 15].groupby('year').size()
    percent_high_choice = (high_choice_counties / total_counties) * 100
    
    print("\nPercentage of counties with more than 15 plans by year:")
    print(percent_high_choice)
else:
    print("Error: 'county' or 'year' columns not found. Cannot create the required analysis.")


#------------------------------------------


# Load the data
df = pd.read_csv('data/output/final_ma_data.csv')

# 1. Star rating distribution for 2010, 2012, and 2015
years_to_analyze = [2010, 2012, 2015]
star_rating_data = []

for year in years_to_analyze:
    year_data = df[df['year'] == year]
    if 'Star_Rating' in year_data.columns:
        # Remove missing values and ensure numeric
        valid_ratings = pd.to_numeric(year_data['Star_Rating'], errors='coerce')
        valid_ratings = valid_ratings.dropna()
        
        # Count occurrences of each rating
        rating_counts = valid_ratings.value_counts().sort_index()
        
        # Save data for comparison
        for rating, count in rating_counts.items():
            star_rating_data.append({
                'Year': year,
                'Rating': rating,
                'Count': count
            })

# Convert to DataFrame for easier plotting
star_df = pd.DataFrame(star_rating_data)

# Figure 1: Star Ratings Distribution
plt.figure(figsize=(12, 8))
if not star_df.empty:
    sns.barplot(x='Rating', y='Count', hue='Year', data=star_df)
    plt.title('Distribution of Star Ratings (2010, 2012, 2015)', fontsize=16)
    plt.xlabel('Star Rating', fontsize=14)
    plt.ylabel('Number of Plans', fontsize=14)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Add percentages for each year
    for year in years_to_analyze:
        year_subset = star_df[star_df['Year'] == year]
        if not year_subset.empty:
            year_total = year_subset['Count'].sum()
            year_pcts = year_subset.copy()
            year_pcts['Percentage'] = year_pcts['Count'] / year_total * 100
            print(f"\nStar Rating Distribution for {year}:")
            for _, row in year_pcts.iterrows():
                print(f"  {row['Rating']} stars: {row['Count']} plans ({row['Percentage']:.1f}%)")
else:
    plt.text(0.5, 0.5, 'No valid star rating data found', 
             horizontalalignment='center', verticalalignment='center')

plt.tight_layout()
plt.savefig('star_rating_distribution.png', dpi=300)
plt.show()

# Figure 2: Average benchmark payment over time
plt.figure(figsize=(12, 8))
if 'ma_rate' in df.columns:
    # Ensure ma_rate is numeric and handle NaN values
    df['ma_rate'] = pd.to_numeric(df['ma_rate'], errors='coerce')
    
    # Group by year and calculate average benchmark (ignoring NaN values)
    benchmark_by_year = df.groupby('year')['ma_rate'].mean().reset_index()
    
    # Plot benchmark trends
    plt.plot(benchmark_by_year['year'], benchmark_by_year['ma_rate'], 
             marker='o', linestyle='-', linewidth=2)
    plt.title('Average Benchmark Payment (2010-2015)', fontsize=16)
    plt.xlabel('Year', fontsize=14)
    plt.ylabel('Average Benchmark Payment ($)', fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # Calculate the percentage increase from 2010 to 2015 if both values exist
    if 2010 in benchmark_by_year['year'].values and 2015 in benchmark_by_year['year'].values:
        start_value = benchmark_by_year[benchmark_by_year['year'] == 2010]['ma_rate'].values[0]
        end_value = benchmark_by_year[benchmark_by_year['year'] == 2015]['ma_rate'].values[0]
        percent_increase = ((end_value - start_value) / start_value) * 100
        
        # Add text showing percentage increase
        plt.text(2012, benchmark_by_year['ma_rate'].max() * 0.9, 
                f'Increase from 2010 to 2015: ${end_value - start_value:.2f} ({percent_increase:.1f}%)',
                bbox=dict(facecolor='white', alpha=0.8))
        
        print("\nAverage Benchmark Payment by Year:")
        for _, row in benchmark_by_year.iterrows():
            print(f"  {row['year']}: ${row['ma_rate']:.2f}")
        print(f"\nBenchmark payment increase from 2010 to 2015: ${end_value - start_value:.2f} ({percent_increase:.1f}%)")
else:
    plt.text(0.5, 0.5, 'No benchmark payment data found', 
             horizontalalignment='center', verticalalignment='center')

plt.tight_layout()
plt.savefig('benchmark_payment_trend.png', dpi=300)
plt.show()

