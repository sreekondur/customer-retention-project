import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

# ---- PAGE CONFIG ----
st.set_page_config(
    page_title="Customer Retention Dashboard",
    page_icon="📊",
    layout="wide"
)

# ---- LOAD DATA ----
@st.cache_data
def load_data():
    df = pd.read_csv("data/rfm_data.csv")
    return df

@st.cache_resource
def train_model():
    df = pd.read_csv("data/rfm_data.csv")
    df = df.drop(columns=["customerID", "Segment"])
    le = LabelEncoder()
    text_columns = df.select_dtypes(include=["object"]).columns
    for col in text_columns:
        df[col] = le.fit_transform(df[col])
    X = df.drop(columns=["Churn"])
    y = df["Churn"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    return model, X.columns.tolist()

df = load_data()
model, feature_cols = train_model()

# ---- SIDEBAR ----
st.sidebar.title("📊 Navigation")
page = st.sidebar.radio("Go to", ["Overview", "Customer Lookup", "Business Summary"])

# ============================
# PAGE 1: OVERVIEW
# ============================
if page == "Overview":
    st.title("Customer Retention Dashboard")
    st.markdown("Analyze churn, segments, and customer value for a telecom company.")

    # Key metrics
    total = len(df)
    churned = df["Churn"].sum()
    churn_rate = churned / total * 100
    avg_monthly = df["MonthlyCharges"].mean()

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Customers", f"{total:,}")
    col2.metric("Churned Customers", f"{int(churned):,}")
    col3.metric("Churn Rate", f"{churn_rate:.1f}%")
    col4.metric("Avg Monthly Charges", f"${avg_monthly:.2f}")

    st.markdown("---")

    # Charts side by side
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Churn Breakdown")
        fig, ax = plt.subplots(figsize=(5, 3))
        sns.countplot(data=df, x="Churn", hue="Churn", legend=False,
                      palette="Blues", ax=ax)
        ax.set_xticks([0, 1])
        ax.set_xticklabels(["Stayed", "Churned"])
        st.pyplot(fig)

    with col2:
        st.subheader("Customer Segments")
        fig, ax = plt.subplots(figsize=(5, 3))
        segment_counts = df["Segment"].value_counts()
        segment_counts.plot(kind="bar", color="#2E75B6", ax=ax)
        ax.set_xticklabels(segment_counts.index, rotation=0)
        st.pyplot(fig)

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Churn by Contract Type")
        fig, ax = plt.subplots(figsize=(5, 3))
        sns.countplot(data=df, x="Contract", hue="Churn", palette="Blues", ax=ax)
        st.pyplot(fig)

    with col2:
        st.subheader("Monthly Charges vs Churn")
        fig, ax = plt.subplots(figsize=(5, 3))
        sns.boxplot(data=df, x="Churn", y="MonthlyCharges",
                    hue="Churn", legend=False, palette="Blues", ax=ax)
        ax.set_xticks([0, 1])
        ax.set_xticklabels(["Stayed", "Churned"])
        st.pyplot(fig)

# ============================
# PAGE 2: CUSTOMER LOOKUP
# ============================
elif page == "Customer Lookup":
    st.title("🔍 Customer Churn Predictor")
    st.markdown("Enter a customer's details to predict their churn risk and lifetime value.")

    col1, col2, col3 = st.columns(3)

    with col1:
        tenure = st.slider("Tenure (months)", 0, 72, 12)
        monthly_charges = st.slider("Monthly Charges ($)", 18, 120, 65)
        contract = st.selectbox("Contract Type", ["Month-to-month", "One year", "Two year"])

    with col2:
        internet_service = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
        online_security = st.selectbox("Online Security", ["Yes", "No", "No internet service"])
        tech_support = st.selectbox("Tech Support", ["Yes", "No", "No internet service"])

    with col3:
        payment_method = st.selectbox("Payment Method", [
            "Electronic check", "Mailed check",
            "Bank transfer (automatic)", "Credit card (automatic)"
        ])
        senior_citizen = st.selectbox("Senior Citizen", ["No", "Yes"])
        partner = st.selectbox("Has Partner", ["Yes", "No"])

    if st.button("🔮 Predict", use_container_width=True):
        # Build input
        input_data = pd.read_csv("data/rfm_data.csv").drop(
            columns=["customerID", "Segment", "Churn"]
        ).iloc[0:1].copy()

        # Override with user inputs
        input_data["tenure"] = tenure
        input_data["MonthlyCharges"] = monthly_charges
        input_data["TotalCharges"] = monthly_charges * tenure
        input_data["Contract"] = contract
        input_data["InternetService"] = internet_service
        input_data["OnlineSecurity"] = online_security
        input_data["TechSupport"] = tech_support
        input_data["PaymentMethod"] = payment_method
        input_data["SeniorCitizen"] = 1 if senior_citizen == "Yes" else 0
        input_data["Partner"] = partner

        # Encode
        le = LabelEncoder()
        df_temp = pd.read_csv("data/rfm_data.csv").drop(columns=["customerID", "Segment"])
        text_cols = df_temp.select_dtypes(include=["object"]).columns
        for col in text_cols:
            le.fit(df_temp[col])
            if col in input_data.columns:
                try:
                    input_data[col] = le.transform(input_data[col])
                except:
                    input_data[col] = 0

        input_data = input_data[feature_cols]
        churn_prob = model.predict_proba(input_data)[0][1]
        clv = monthly_charges * (1 - churn_prob) * 72

        # Display results
        st.markdown("---")
        col1, col2, col3 = st.columns(3)

        col1.metric("Churn Probability", f"{churn_prob * 100:.1f}%",
                    delta="High Risk" if churn_prob > 0.5 else "Low Risk",
                    delta_color="inverse")
        col2.metric("Estimated CLV", f"${clv:,.0f}")

        if tenure < 12:
            segment = "Sleeping"
        elif tenure < 36:
            segment = "At Risk"
        elif tenure < 60:
            segment = "Loyal"
        else:
            segment = "Champion"
        col3.metric("Segment", segment)

        # Recommendation
        st.markdown("### 💡 Business Recommendation")
        if churn_prob > 0.5 and clv > 2000:
            st.error("🔴 URGENT: High value customer at risk! Offer a discount or upgrade to annual contract immediately.")
        elif churn_prob > 0.5 and clv <= 2000:
            st.warning("🟡 LOW PRIORITY: Customer likely to churn but low value. Use automated campaign only.")
        else:
            st.success("🟢 SAFE: Customer is stable. Consider loyalty rewards to keep them happy.")

# ============================
# PAGE 3: BUSINESS SUMMARY
# ============================
elif page == "Business Summary":
    st.title("📋 Business Action Summary")
    st.markdown("Where should the business focus its retention budget?")

    clv_data = pd.read_csv("data/clv_data.csv")
    median_clv = clv_data["CLV"].median()

    high_risk_high_value = clv_data[
        (clv_data["Churn_Probability"] > 0.5) & (clv_data["CLV"] > median_clv)
    ]
    high_risk_low_value = clv_data[
        (clv_data["Churn_Probability"] > 0.5) & (clv_data["CLV"] <= median_clv)
    ]
    low_risk_high_value = clv_data[
        (clv_data["Churn_Probability"] <= 0.5) & (clv_data["CLV"] > median_clv)
    ]

    col1, col2, col3 = st.columns(3)

    col1.error(f"🔴 Urgent Action\n\n**{len(high_risk_high_value)} customers**\n\nHigh risk + High value\n\nOffer retention deals now")
    col2.warning(f"🟡 Low Priority\n\n**{len(high_risk_low_value)} customers**\n\nHigh risk + Low value\n\nAutomated campaigns only")
    col3.success(f"🟢 Reward Loyalty\n\n**{len(low_risk_high_value)} customers**\n\nLow risk + High value\n\nGive loyalty perks")

    st.markdown("---")
    st.subheader("High Value At-Risk Customers")
    st.markdown("These are the customers that need immediate attention:")
    st.dataframe(
        high_risk_high_value[["customerID", "tenure", "MonthlyCharges", "CLV", "Churn_Probability", "Segment"]]
        .sort_values("CLV", ascending=False)
        .head(20)
        .reset_index(drop=True)
    )