import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load clean data
df = pd.read_csv("data/clean_data.csv")

# Make all charts look nice
sns.set_theme(style="whitegrid")


# ----- CHART 1: Churn Count ----
plt.figure(figsize=(6, 4))
ax= sns.countplot(data=df, x="Churn", palette="Blues")
plt.title("How many customers churned?")
plt.xticks([0, 1], ["Stayed", "Left"])
for p in ax.patches:
    ax.annotate(f'{int(p.get_height())}', 
                (p.get_x() + p.get_width() / 2, p.get_height()), 
                ha='center', va='bottom', fontsize=12, fontweight='bold')
plt.savefig("data/chart1_churn_count.png")
plt.show()
print("Chart 1 done!")

# ---- CHART 2: Tenure vs Churn -----
plt.figure(figsize=(8, 4))
sns.histplot(data=df, x="tenure", hue="Churn", bins=30, palette="Blues")
plt.title("How long had customers been with the company before churning?")
plt.xlabel("Months as a customer (tenure)")
plt.savefig("data/chart2_tenure.png")
plt.show()
print("Chart 2 done!")

# ----- CHART 3: Contract Type vs Churn -----
plt.figure(figsize=(8, 4))
ax= sns.countplot(data=df, x="Contract", hue="Churn", palette="Blues")
plt.title("Does contract type affect churn?")
for p in ax.patches:
    ax.annotate(f'{int(p.get_height())}', 
                (p.get_x() + p.get_width() / 2, p.get_height()), 
                ha='center', va='bottom', fontsize=10, fontweight='bold')
plt.savefig("data/chart3_contract.png")
plt.show()
print("Chart 3 done!")

# ----- CHART 4: Monthly Charges vs Churn -----
plt.figure(figsize=(8, 4))
sns.boxplot(data=df, x="Churn", y="MonthlyCharges", palette="Blues")
plt.title("Do churned customers pay more per month?")
plt.xticks([0, 1], ["Stayed", "Left"])
plt.savefig("data/chart4_monthly_charges.png")
plt.show()
print("Chart 4 done!")

print()
print("All charts saved to the data folder!")