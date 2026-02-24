import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load clean data
df = pd.read_csv("data/clean_data.csv")

sns.set_theme(style="whitegrid")

# ---- BUILD RFM FEATURES ----
service_cols = ["PhoneService", "InternetService", "OnlineSecurity",
                "OnlineBackup", "DeviceProtection", "TechSupport",
                "StreamingTV", "StreamingMovies"]

# Count how many services each customer has
df["ServiceCount"] = df[service_cols].apply(
    lambda row: sum(1 for val in row if val not in ["No", "No internet service", "No phone service"]), axis=1
)

# ---- SCORE EACH CUSTOMER 1, 2, OR 3 ----
df["R_Score"] = pd.qcut(df["tenure"], q=3, labels=[1, 2, 3])
df["F_Score"] = pd.qcut(df["ServiceCount"].rank(method="first"), q=3, labels=[1, 2, 3])
df["M_Score"] = pd.qcut(df["MonthlyCharges"], q=3, labels=[1, 2, 3])

# Convert to numbers
df["R_Score"] = df["R_Score"].astype(int)
df["F_Score"] = df["F_Score"].astype(int)
df["M_Score"] = df["M_Score"].astype(int)

# ---- COMBINE INTO ONE RFM SCORE ----
df["RFM_Score"] = df["R_Score"] + df["F_Score"] + df["M_Score"]

print("RFM Score range:", df["RFM_Score"].min(), "to", df["RFM_Score"].max())
print()

# ---- ASSIGN SEGMENTS ----
def assign_segment(score):
    if score >= 8:
        return "Champion"
    elif score >= 6:
        return "Loyal"
    elif score >= 4:
        return "At Risk"
    else:
        return "Sleeping"

df["Segment"] = df["RFM_Score"].apply(assign_segment)

print("Customer Segments:")
print(df["Segment"].value_counts())
print()

# ---- CHART: Segments ----
plt.figure(figsize=(8, 4))
ax = sns.countplot(data=df, x="Segment", hue="Segment", legend=False,
                   order=["Champion", "Loyal", "At Risk", "Sleeping"],
                   palette="Blues_r")
plt.title("Customer Segments")
total = len(df)
for p in ax.patches:
    count = int(p.get_height())
    percentage = f'{100 * count / total:.1f}%'
    ax.annotate(percentage,
                (p.get_x() + p.get_width() / 2, p.get_height() / 2),
                ha='center', va='center', fontsize=12, fontweight='bold', color='black')
plt.savefig("data/chart5_segments.png")
plt.show()
print("Chart saved!")

# ---- CHART: Churn rate per segment ----
churn_by_segment = df.groupby("Segment")["Churn"].mean() * 100
print("Churn rate by segment %:")
print(churn_by_segment.round(2))

plt.figure(figsize=(8, 4))
ax = churn_by_segment.reindex(["Champion", "Loyal", "At Risk", "Sleeping"]).plot(
    kind="bar", color=["#1F4E79", "#2E75B6", "#5BA3D9", "#A8C8E8"]
)
plt.title("Churn Rate by Customer Segment (%)")
plt.ylabel("Churn Rate %")
plt.xticks(rotation=0)
for p in ax.patches:
    ax.annotate(f'{p.get_height():.1f}%',
                (p.get_x() + p.get_width() / 2, p.get_height() / 2),
                ha='center', va='center', fontsize=12, fontweight='bold', color='black')
plt.savefig("data/chart6_churn_by_segment.png")
plt.show()
print("Done!")