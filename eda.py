# ============================================================================
# EXPLORATORY DATA ANALYSIS (EDA) — Used Car Price Evaluator
# ============================================================================
# 
# WHAT IS EDA?
# EDA is the first step in any ML project. Before building a model, we need to
# UNDERSTAND our data — what it looks like, what's missing, what patterns exist.
# Think of it like studying the exam paper before answering questions.
#
# WHAT THIS SCRIPT DOES:
# 1. Loads the dataset and shows basic statistics
# 2. Checks for missing values (gaps in our data)
# 3. Creates visualizations to understand price patterns
# 4. Saves all plots to a 'plots/' folder
# ============================================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

# ── Step 1: Load the Dataset ────────────────────────────────────────────────
# pandas.read_csv() reads a CSV file into a DataFrame (a table-like structure)
# Think of a DataFrame as an Excel spreadsheet in Python

df = pd.read_csv('train.csv')

# Drop the unnamed index column (it's just row numbers from the original file)
if 'Unnamed: 0' in df.columns:
    df = df.drop('Unnamed: 0', axis=1)
    # axis=1 means we're dropping a COLUMN (axis=0 would drop a ROW)

print("=" * 60)
print("DATASET OVERVIEW")
print("=" * 60)

# .shape returns (rows, columns) — tells us the SIZE of our data
print(f"\n📊 Dataset Size: {df.shape[0]} rows × {df.shape[1]} columns")
print(f"   → Each row is ONE car listing")
print(f"   → Each column is ONE feature (attribute) of that car\n")

# .dtypes tells us the DATA TYPE of each column
# - object  = text/string (like car names, fuel type)
# - float64 = decimal numbers (like price)
# - int64   = whole numbers (like year)
print("📋 Column Data Types:")
print(df.dtypes)

# ── Step 2: Check for Missing Values ────────────────────────────────────────
# Missing values are a HUGE problem in ML. If data is missing, our model can't 
# learn from it. We need to know WHERE and HOW MUCH data is missing.

print("\n" + "=" * 60)
print("MISSING VALUES ANALYSIS")
print("=" * 60)

missing = df.isnull().sum()  # Count NaN (Not a Number) values per column
missing_pct = (missing / len(df)) * 100  # Convert to percentage

# Only show columns that actually HAVE missing values
missing_df = pd.DataFrame({
    'Missing Count': missing,
    'Missing %': missing_pct.round(2)
})
missing_df = missing_df[missing_df['Missing Count'] > 0]
print(f"\n⚠️  Columns with missing data:")
print(missing_df)
print(f"\n💡 WHY THIS MATTERS: We'll need to handle these missing values before")
print(f"   training our model. Common strategies: fill with median, drop rows, etc.")

# ── Step 3: Basic Statistics ────────────────────────────────────────────────
# .describe() gives us statistical summary: mean, min, max, percentiles
# This helps us understand the RANGE and DISTRIBUTION of our numerical data

print("\n" + "=" * 60)
print("PRICE STATISTICS (in Lakhs)")
print("=" * 60)
print(f"\n💰 Average Price:  ₹{df['Price'].mean():.2f} Lakhs")
print(f"💰 Median Price:   ₹{df['Price'].median():.2f} Lakhs")
print(f"💰 Cheapest Car:   ₹{df['Price'].min():.2f} Lakhs")
print(f"💰 Most Expensive: ₹{df['Price'].max():.2f} Lakhs")
print(f"\n💡 WHY MEDIAN vs MEAN?")
print(f"   Mean = {df['Price'].mean():.2f}, Median = {df['Price'].median():.2f}")
print(f"   When mean > median, data is RIGHT-SKEWED (few expensive cars pull mean up)")
print(f"   Median is often MORE RELIABLE for skewed data.")

# ── Step 4: Categorical Feature Distribution ────────────────────────────────
# Categorical = text-based features (Fuel_Type, Transmission, etc.)
# value_counts() tells us how many of each category exist

print("\n" + "=" * 60)
print("CATEGORICAL FEATURE BREAKDOWN")
print("=" * 60)

for col in ['Fuel_Type', 'Transmission', 'Owner_Type']:
    print(f"\n🔹 {col}:")
    counts = df[col].value_counts()
    for val, count in counts.items():
        pct = (count / len(df)) * 100
        print(f"   {val}: {count} ({pct:.1f}%)")

# ── Step 5: Extract Brand (Feature Engineering Preview) ─────────────────────
# The 'Name' column contains the full car name like "Hyundai Creta 1.6 CRDi SX"
# We extract just the BRAND (first word) — this is called FEATURE ENGINEERING

df['Brand'] = df['Name'].apply(lambda x: x.split(' ')[0])
# lambda = a mini function | split(' ') = split by space | [0] = take first word

print("\n" + "=" * 60)
print("TOP 15 CAR BRANDS (by listing count)")
print("=" * 60)
brand_counts = df['Brand'].value_counts().head(15)
for brand, count in brand_counts.items():
    bar = '█' * (count // 20)
    print(f"   {brand:15s} {count:4d} {bar}")

# ── Step 6: Create Visualizations ───────────────────────────────────────────
# Visualizations help us SEE patterns that numbers alone can't show.
# We save all plots to a 'plots/' folder.

os.makedirs('plots', exist_ok=True)  # Create folder if it doesn't exist

# Set a clean visual style
sns.set_style("darkgrid")
plt.rcParams['figure.figsize'] = (12, 6)
plt.rcParams['font.size'] = 12

# ── Plot 1: Price Distribution ──
# A histogram shows how prices are DISTRIBUTED
# If most bars are on the left = most cars are cheap (right-skewed)
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

axes[0].hist(df['Price'], bins=50, color='#3498db', edgecolor='white', alpha=0.8)
axes[0].set_title('Price Distribution (All Cars)', fontweight='bold')
axes[0].set_xlabel('Price (Lakhs)')
axes[0].set_ylabel('Number of Cars')
axes[0].axvline(df['Price'].median(), color='red', linestyle='--', label=f"Median: ₹{df['Price'].median():.1f}L")
axes[0].legend()

# Log-transformed price (reduces skewness — makes patterns clearer)
axes[1].hist(np.log1p(df['Price']), bins=50, color='#e74c3c', edgecolor='white', alpha=0.8)
axes[1].set_title('Log-Transformed Price Distribution', fontweight='bold')
axes[1].set_xlabel('log(Price)')
axes[1].set_ylabel('Number of Cars')

plt.tight_layout()
plt.savefig('plots/01_price_distribution.png', dpi=150, bbox_inches='tight')
plt.close()
print("\n✅ Saved: plots/01_price_distribution.png")

# ── Plot 2: Price vs Year ──
# Newer cars should cost more — let's verify this pattern
fig, ax = plt.subplots(figsize=(12, 6))
ax.scatter(df['Year'], df['Price'], alpha=0.3, s=15, color='#2ecc71')
ax.set_title('Price vs Manufacturing Year', fontweight='bold')
ax.set_xlabel('Year')
ax.set_ylabel('Price (Lakhs)')
plt.tight_layout()
plt.savefig('plots/02_price_vs_year.png', dpi=150, bbox_inches='tight')
plt.close()
print("✅ Saved: plots/02_price_vs_year.png")

# ── Plot 3: Price by Fuel Type ──
# Box plots show the MIN, Q1, MEDIAN, Q3, MAX price for each fuel type
fig, ax = plt.subplots(figsize=(10, 6))
df.boxplot(column='Price', by='Fuel_Type', ax=ax)
ax.set_title('Price Distribution by Fuel Type', fontweight='bold')
ax.set_xlabel('Fuel Type')
ax.set_ylabel('Price (Lakhs)')
plt.suptitle('')  # Remove auto-generated title
plt.tight_layout()
plt.savefig('plots/03_price_by_fuel_type.png', dpi=150, bbox_inches='tight')
plt.close()
print("✅ Saved: plots/03_price_by_fuel_type.png")

# ── Plot 4: Price by Transmission ──
fig, ax = plt.subplots(figsize=(8, 6))
df.boxplot(column='Price', by='Transmission', ax=ax)
ax.set_title('Price Distribution by Transmission', fontweight='bold')
ax.set_xlabel('Transmission Type')
ax.set_ylabel('Price (Lakhs)')
plt.suptitle('')
plt.tight_layout()
plt.savefig('plots/04_price_by_transmission.png', dpi=150, bbox_inches='tight')
plt.close()
print("✅ Saved: plots/04_price_by_transmission.png")

# ── Plot 5: Top 10 Brands - Average Price ──
fig, ax = plt.subplots(figsize=(12, 6))
top_brands = df['Brand'].value_counts().head(10).index
brand_avg_price = df[df['Brand'].isin(top_brands)].groupby('Brand')['Price'].mean().sort_values(ascending=True)
brand_avg_price.plot(kind='barh', ax=ax, color='#9b59b6', edgecolor='white')
ax.set_title('Average Price by Top 10 Brands', fontweight='bold')
ax.set_xlabel('Average Price (Lakhs)')
ax.set_ylabel('Brand')
plt.tight_layout()
plt.savefig('plots/05_avg_price_by_brand.png', dpi=150, bbox_inches='tight')
plt.close()
print("✅ Saved: plots/05_avg_price_by_brand.png")

# ── Plot 6: Correlation Heatmap ──
# Correlation measures how STRONGLY two numerical features are related
# +1 = perfect positive correlation (A goes up → B goes up)
# -1 = perfect negative correlation (A goes up → B goes down)
#  0 = no correlation
# We need to clean numeric columns first for the heatmap

df_numeric = df[['Year', 'Kilometers_Driven', 'Seats', 'Price']].copy()

# Clean Mileage, Engine, Power by stripping text units
def extract_number(series):
    """Extract the numeric part from strings like '19.67 kmpl' or '1582 CC'"""
    return pd.to_numeric(series.astype(str).str.extract(r'([\d.]+)')[0], errors='coerce')

df_numeric['Mileage'] = extract_number(df['Mileage'])
df_numeric['Engine'] = extract_number(df['Engine'])
df_numeric['Power'] = extract_number(df['Power'])

fig, ax = plt.subplots(figsize=(10, 8))
corr_matrix = df_numeric.corr()
sns.heatmap(corr_matrix, annot=True, cmap='RdYlBu_r', center=0, fmt='.2f', ax=ax,
            square=True, linewidths=0.5)
ax.set_title('Feature Correlation Heatmap', fontweight='bold', fontsize=14)
plt.tight_layout()
plt.savefig('plots/06_correlation_heatmap.png', dpi=150, bbox_inches='tight')
plt.close()
print("✅ Saved: plots/06_correlation_heatmap.png")

# ── Summary ─────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("EDA COMPLETE! KEY TAKEAWAYS")
print("=" * 60)
print("""
📌 1. Price is right-skewed — most cars are under ₹10 Lakhs, few are very expensive
📌 2. Newer cars (higher Year) tend to cost more
📌 3. Automatic cars cost more than Manual on average
📌 4. Diesel vs Petrol pricing varies by brand
📌 5. Some columns have missing data that we'll need to handle
📌 6. Features like Year, Engine, Power have good correlation with Price

🔜 NEXT STEP: Run train_model.py to preprocess data and train the ML model!
""")
