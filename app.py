import joblib
import pandas as pd
import streamlit as st

# Load saved model, feature columns, and decision threshold
model = joblib.load("model/churn_model.joblib")
feature_columns = joblib.load("model/feature_columns.joblib")
threshold = joblib.load("model/decision_threshold.joblib")

st.set_page_config(
    page_title="Customer Churn Risk Predictor",
    page_icon="📉",
    layout="centered"
)

st.title("📉 Customer Churn Risk Predictor")

st.write(
    "Enter customer information below to estimate the probability of churn."
)

# Customer inputs
tenure = st.number_input(
    "Tenure (months)",
    min_value=0,
    max_value=72,
    value=12
)

monthly_charges = st.number_input(
    "Monthly Charges",
    min_value=0.0,
    value=70.0
)

total_charges = st.number_input(
    "Total Charges",
    min_value=0.0,
    value=500.0
)

contract = st.selectbox(
    "Contract Type",
    ["Month-to-month", "One year", "Two year"]
)

internet_service = st.selectbox(
    "Internet Service",
    ["DSL", "Fiber optic", "No"]
)

payment_method = st.selectbox(
    "Payment Method",
    [
        "Electronic check",
        "Mailed check",
        "Bank transfer (automatic)",
        "Credit card (automatic)"
    ]
)

# Create input dataframe
input_data = pd.DataFrame({
    "tenure": [tenure],
    "MonthlyCharges": [monthly_charges],
    "TotalCharges": [total_charges],
    "Contract": [contract],
    "InternetService": [internet_service],
    "PaymentMethod": [payment_method]
})

# One-hot encode input
input_encoded = pd.get_dummies(input_data)

# Match training feature columns
input_encoded = input_encoded.reindex(
    columns=feature_columns,
    fill_value=0
)

if st.button("Predict Churn Risk"):

    churn_probability = model.predict_proba(input_encoded)[:, 1][0]

    st.metric(
        "Predicted Churn Probability",
        f"{churn_probability:.1%}"
    )

    if churn_probability >= threshold:
        st.error("High Churn Risk")
    else:
        st.success("Low Churn Risk")