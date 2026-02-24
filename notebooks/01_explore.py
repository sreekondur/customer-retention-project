import pandas as pd

# Load the data
df = pd.read_csv("data/WA_Fn-UseC_-Telco-Customer-Churn.csv")

# How big is the dataset?
print("Shape:", df.shape)
print("That means", df.shape[0], "customers and", df.shape[1], "columns")
print()

# What columns do we have?
print("Columns:")
print(df.columns.tolist())
print()

# First 5 rows
print("First 5 rows:")
print(df.head())
print()

# How many customers churned?
print("Churn breakdown:")
print(df["Churn"].value_counts())
print()

# What percentage churned?
churn_rate = df["Churn"].value_counts(normalize=True) * 100
print("Churn rate %:")
print(churn_rate.round(2))