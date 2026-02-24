import pandas as pd

#Load the data
df = pd.read_csv("data/WA_Fn-UseC_-Telco-Customer-Churn.csv")

#------Checking for MISSING VALUES-----
print("Missing values in each colimn:")
print(df.isnull().sum())
print()

# ---------- Fix TOTALCHARGES ------
# TotalCharges should be a number but pandas reads it as text
# Let's convert it and see what happens
df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")

# Now check again for missing values
print("Missing values after fixing TotalCharges:")
print(df.isnull().sum()[df.isnull().sum() > 0])
print()

# ---- FIX CHURN COLUMN ---
# Convert Yes/No to 1/0 so the model can understand it
df["Churn"] = df["Churn"].map({"Yes": 1, "No": 0})
print(df["Churn"].value_counts())
print()

# ------ SAVE CLEAN DATA ----
df.to_csv("data/clean_data.csv", index=False)
print("Clean data saved to data/clean_data.csv!")