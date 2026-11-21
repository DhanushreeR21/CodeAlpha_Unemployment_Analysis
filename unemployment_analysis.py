import kagglehub

# Download latest version
path = kagglehub.dataset_download("gokulrajkmv/unemployment-in-india")

print("Path to dataset files:", path)
import os

print("\nFiles available in dataset:")

for file in os.listdir(path):
    print(file)
import pandas as pd
import os

# Dataset file path
file_path = os.path.join(path, "Unemployment in India.csv")

# Load dataset
df = pd.read_csv(file_path)

# Display first five rows
print("\nFirst 5 rows:")
print(df.head())

# Display dataset information
print("\nDataset Information:")
print(df.info())

# Display dataset shape
print("\nDataset Shape:")
print(df.shape)

# Display column names
print("\nColumn Names:")
print(df.columns.tolist())  
# Clean column names
df.columns = df.columns.str.strip()

print("\nCleaned Column Names:")
print(df.columns.tolist())

# Check missing values
print("\nMissing Values:")
print(df.isnull().sum())

# Check duplicate records
print("\nDuplicate Records:")
print(df.duplicated().sum())

# Remove missing values
df = df.dropna()

# Remove duplicate records
df = df.drop_duplicates()

print("\nDataset Shape After Cleaning:")
print(df.shape)

# Check dataset information again
print("\nUpdated Dataset Information:")
print(df.info())  
# Exploratory Data Analysis (EDA)

print("\n========== UNEMPLOYMENT ANALYSIS ==========")

# Basic statistical summary
print("\nStatistical Summary:")
print(df.describe())

# Average unemployment rate
average_unemployment = df["Estimated Unemployment Rate (%)"].mean()
print("\nAverage Unemployment Rate:", round(average_unemployment, 2), "%")

# Highest unemployment rate
highest_unemployment = df["Estimated Unemployment Rate (%)"].max()
print("Highest Unemployment Rate:", round(highest_unemployment, 2), "%")

# Lowest unemployment rate
lowest_unemployment = df["Estimated Unemployment Rate (%)"].min()
print("Lowest Unemployment Rate:", round(lowest_unemployment, 2), "%")

# State-wise average unemployment rate
state_unemployment = df.groupby("Region")["Estimated Unemployment Rate (%)"].mean()

print("\nState-wise Average Unemployment Rate:")
print(state_unemployment.sort_values(ascending=False))

# Area-wise unemployment rate
area_unemployment = df.groupby("Area")["Estimated Unemployment Rate (%)"].mean()

print("\nArea-wise Average Unemployment Rate:")
print(area_unemployment)
import matplotlib.pyplot as plt
import seaborn as sns

# Create output folder for graphs
import os

output_folder = os.path.dirname(os.path.abspath(__file__))

# 1. State-wise unemployment rate
plt.figure(figsize=(12, 7))

state_data = df.groupby("Region")[
    "Estimated Unemployment Rate (%)"
].mean().sort_values(ascending=False)

sns.barplot(x=state_data.values, y=state_data.index)

plt.title("Average Unemployment Rate by State")
plt.xlabel("Unemployment Rate (%)")
plt.ylabel("State")

plt.tight_layout()
plt.savefig(os.path.join(output_folder, "state_unemployment.png"), dpi=300)
plt.show()


# 2. Unemployment rate distribution
plt.figure(figsize=(10, 6))

sns.histplot(
    df["Estimated Unemployment Rate (%)"],
    bins=20,
    kde=True
)

plt.title("Distribution of Unemployment Rate")
plt.xlabel("Unemployment Rate (%)")
plt.ylabel("Frequency")

plt.tight_layout()
plt.savefig(os.path.join(output_folder, "unemployment_distribution.png"), dpi=300)
plt.show()


# 3. Rural vs Urban comparison
plt.figure(figsize=(8, 6))

sns.barplot(
    data=df,
    x="Area",
    y="Estimated Unemployment Rate (%)"
)

plt.title("Rural vs Urban Unemployment Rate")
plt.xlabel("Area")
plt.ylabel("Unemployment Rate (%)")

plt.tight_layout()
plt.savefig(os.path.join(output_folder, "area_comparison.png"), dpi=300)
plt.show()

print("\nAll graphs saved successfully!")
# Convert Date column into datetime format
df["Date"] = pd.to_datetime(
    df["Date"].str.strip(),
    dayfirst=True,
    errors="coerce"
)

# Remove invalid dates
date_df = df.dropna(subset=["Date"]).copy()

# Calculate monthly average unemployment
monthly_data = date_df.groupby(
    date_df["Date"].dt.to_period("M")
)["Estimated Unemployment Rate (%)"].mean()

# Plot monthly unemployment trend
plt.figure(figsize=(12, 6))

plt.plot(
    monthly_data.index.astype(str),
    monthly_data.values,
    marker="o"
)

plt.title("Monthly Unemployment Rate Trend in India")
plt.xlabel("Month")
plt.ylabel("Unemployment Rate (%)")
plt.xticks(rotation=45)
plt.grid(True)

plt.tight_layout()
plt.savefig(
    os.path.join(output_folder, "monthly_unemployment_trend.png"),
    dpi=300
)
plt.show()

print("\nMonthly unemployment trend graph saved successfully!")