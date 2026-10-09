import joblib
import pandas as pd
import streamlit as st

# --------------------------------------------------
# Page configuration
# --------------------------------------------------
st.set_page_config(
    page_title="Customer Churn Risk Predictor",
    page_icon="📉",
    layout="centered"
)

# --------------------------------------------------
# Load trained model and configuration
# --------------------------------------------------
model = joblib.load("model/churn_model.joblib")
feature_columns = joblib.load("model/feature_columns.joblib")
threshold = float(joblib.load("model/decision_threshold.joblib"))

# --------------------------------------------------
# Application introduction
# --------------------------------------------------
st.title("📉 Customer Churn Risk Predictor")

st.write(
    "This interactive app uses machine learning to estimate "
    "how likely a customer is to leave a telecommunications company. "
    "It demonstrates how businesses can identify customers "
    "who may need proactive retention support."
)

st.divider()

# --------------------------------------------------
# Customer information
# --------------------------------------------------
st.subheader("Customer Information")

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

# --------------------------------------------------
# Prepare model input
# --------------------------------------------------
input_data = pd.DataFrame({
    "tenure": [tenure],
    "MonthlyCharges": [monthly_charges],
    "TotalCharges": [total_charges],
    "Contract": [contract],
    "InternetService": [internet_service],
    "PaymentMethod": [payment_method]
})

# One-hot encode categorical variables
input_encoded = pd.get_dummies(input_data)

# Align input columns with training features
input_encoded = input_encoded.reindex(
    columns=feature_columns,
    fill_value=0
)

st.divider()

# --------------------------------------------------
# Churn prediction
# --------------------------------------------------
if st.button("Predict Churn Risk", type="primary"):

    churn_probability = float(
        model.predict_proba(input_encoded)[:, 1][0]
    )

    st.subheader("Prediction Results")

    st.metric(
        "Predicted Churn Probability",
        f"{churn_probability:.1%}"
    )

    if churn_probability >= threshold:
        st.error("High Churn Risk")

        st.write(
            "The predicted churn probability is at or above "
            "the model's decision threshold. This customer "
            "is therefore classified as higher risk."
        )

    else:
        st.success("Low Churn Risk")

        st.write(
            "The predicted churn probability is below "
            "the model's decision threshold. This customer "
            "is therefore classified as lower risk."
        )

    st.caption(
        f"Decision threshold: {threshold:.0%}. "
        "This is a model-based estimate, not a certainty "
        "that the customer will leave."
    )

st.divider()

# --------------------------------------------------
# About the model
# --------------------------------------------------
with st.expander("About This Model"):

    st.markdown("### Business Problem")
    st.write(
        "Customer churn occurs when customers stop using a "
        "company's services. Identifying customers at risk "
        "can help businesses prioritize retention efforts."
    )

    st.markdown("### Machine Learning Approach")
    st.write(
        "This project explores customer churn using a "
        "telecommunications dataset. Multiple classification "
        "models were evaluated, including Logistic Regression "
        "and Random Forest."
    )

    st.write(
        "Balanced Logistic Regression was selected to place "
        "greater emphasis on identifying customers who churn, "
        "rather than focusing only on overall accuracy."
    )

    st.markdown("### Model Evaluation")
    st.write(
        "Model performance was evaluated using classification "
        "metrics including precision, recall, F1-score, "
        "and ROC-AUC."
    )

    st.write(
        "The decision threshold was adjusted to support "
        "higher recall, reflecting the business importance "
        "of identifying potentially at-risk customers."
    )

    st.markdown("### Limitations")
    st.write(
        "Predictions are based on historical patterns in "
        "the dataset. They do not establish causation and "
        "should not be interpreted as guaranteed outcomes."
    )

    st.info(
        "This is a machine learning portfolio demonstration, "
        "not a production-ready customer retention system."
    )

# --------------------------------------------------
# Footer
# --------------------------------------------------
st.caption(
    "Customer Churn Prediction | Machine Learning Portfolio Project"
)
