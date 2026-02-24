import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load predictions file (has churn probability from our model)
df = pd.read_csv("data/predictions.csv")

sns.set_theme(style="whitegrid")

# ---- CALCULATE CLV ----
# Expected months remaining based on churn probability
# If churn probability is 80%, we expect them to stay 20% of max tenure
# Max tenure in dataset is 72 months

MAX_TENURE = 72

df["Expected_Months_Remaining"] = (1 - df["Churn_Probability"]) * MAX_TENURE

# CLV = Monthly Charges x Expected Months Remaining
df["CLV"] = df["MonthlyCharges"] * df["Expected_Months_Remaining"]

print("CLV Summary:")
print(df["CLV"].describe().round(2))
print()

# ---- CATEGORIZE CLV ----
df["CLV_Tier"] = pd.qcut(df["CLV"], q=3, labels=["Low", "Medium", "High"])

print("CLV Tiers:")
print(df["CLV_Tier"].value_counts())
print()

# ---- CHART 1: CLV Distribution ----
plt.figure(figsize=(8, 4))
sns.histplot(data=df, x="CLV", bins=40, color="#2E75B6")
plt.title("Customer Lifetime Value Distribution")
plt.xlabel("CLV ($)")
plt.ylabel("Number of Customers")
plt.savefig("data/chart9_clv_distribution.png")
plt.show()
print("Chart 1 done!")

# ---- CHART 2: CLV by Segment ----
plt.figure(figsize=(8, 4))
sns.boxplot(data=df, x="Segment", y="CLV",
            order=["Champion", "Loyal", "At Risk", "Sleeping"],
            hue="Segment", legend=False, palette="Blues")
plt.title("CLV by Customer Segment")
plt.ylabel("CLV ($)")
plt.savefig("data/chart10_clv_by_segment.png")
plt.show()
print("Chart 2 done!")

# ---- CHART 3: Churn Risk vs CLV ----
plt.figure(figsize=(8, 5))
scatter = plt.scatter(df["Churn_Probability"], df["CLV"],
                      c=df["CLV"], cmap="Blues", alpha=0.5, s=10)
plt.colorbar(scatter, label="CLV ($)")
plt.title("Churn Risk vs Customer Lifetime Value")
plt.xlabel("Churn Probability (0 = Safe, 1 = Will Churn)")
plt.ylabel("CLV ($)")
plt.savefig("data/chart11_churn_vs_clv.png")
plt.show()
print("Chart 3 done!")

# ---- BUSINESS PRIORITY TABLE ----
print()
print("=" * 50)
print("BUSINESS ACTION SUMMARY")
print("=" * 50)

high_risk_high_value = df[(df["Churn_Probability"] > 0.5) & (df["CLV"] > df["CLV"].median())]
high_risk_low_value = df[(df["Churn_Probability"] > 0.5) & (df["CLV"] <= df["CLV"].median())]
low_risk_high_value = df[(df["Churn_Probability"] <= 0.5) & (df["CLV"] > df["CLV"].median())]

print(f"🔴 High Risk + High Value: {len(high_risk_high_value)} customers → URGENT: offer retention deals")
print(f"🟡 High Risk + Low Value:  {len(high_risk_low_value)} customers → LOW PRIORITY: let them go")
print(f"🟢 Low Risk + High Value:  {len(low_risk_high_value)} customers → REWARD: loyalty perks")
print()

# ---- SAVE ----
df.to_csv("data/clv_data.csv", index=False)
print("CLV data saved!")