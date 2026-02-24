import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score

# Load RFM data
df = pd.read_csv("data/rfm_data.csv")

sns.set_theme(style="whitegrid")

# ---- STEP 1: PREPARE THE DATA ----
df = df.drop(columns=["customerID", "Segment"])

# Convert all text columns to numbers
le = LabelEncoder()
text_columns = df.select_dtypes(include=["object"]).columns
for col in text_columns:
    df[col] = le.fit_transform(df[col])

print("Data prepared!")
print("Shape:", df.shape)
print()

# ---- STEP 2: SPLIT INTO FEATURES AND TARGET ----
X = df.drop(columns=["Churn"])
y = df["Churn"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print("Training set size:", len(X_train))
print("Test set size:", len(X_test))
print()

# ---- STEP 3: TRAIN THE MODEL ----
print("Training model...")
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)
print("Model trained!")
print()

# ---- STEP 4: EVALUATE THE MODEL ----
y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]

print("Classification Report:")
print(classification_report(y_test, y_pred, target_names=["Stayed", "Churned"]))

auc = roc_auc_score(y_test, y_prob)
print(f"AUC-ROC Score: {auc:.4f}")
print()

# ---- CHART: Confusion Matrix ----
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6, 4))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=["Stayed", "Churned"],
            yticklabels=["Stayed", "Churned"])
plt.title("Confusion Matrix")
plt.ylabel("Actual")
plt.xlabel("Predicted")
plt.savefig("data/chart7_confusion_matrix.png")
plt.show()
print("Confusion matrix saved!")

# ---- CHART: Feature Importance ----
importance = pd.Series(model.feature_importances_, index=X.columns)
importance = importance.sort_values(ascending=False).head(10)

plt.figure(figsize=(8, 5))
importance.plot(kind="bar", color="#2E75B6")
plt.title("Top 10 Features That Predict Churn")
plt.ylabel("Importance Score")
plt.xticks(rotation=45, ha="right")
plt.tight_layout()
plt.savefig("data/chart8_feature_importance.png")
plt.show()
print("Feature importance saved!")

# ---- SAVE PREDICTIONS ----
results = pd.read_csv("data/rfm_data.csv")
results = results.iloc[X_test.index]
results["Churn_Probability"] = y_prob
results.to_csv("data/predictions.csv", index=False)
print("Predictions saved!")