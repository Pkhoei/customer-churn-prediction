
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
@st.cache_resource
def load_artifacts():
    model = joblib.load("model/churn_model.joblib")
    columns = list(joblib.load("model/feature_columns.joblib"))
    threshold = float(
        joblib.load("model/decision_threshold.joblib")
    )
    return model, columns, threshold


model, feature_columns, threshold = load_artifacts()

# --------------------------------------------------
# Introduction
# --------------------------------------------------
st.title("📉 Customer Churn Risk Predictor")

st.write(
    "This interactive app uses machine learning to estimate "
    "how likely a customer is to leave a telecommunications "
    "company. It demonstrates how businesses can identify "
    "customers who may need proactive retention support."
)

st.divider()

# --------------------------------------------------
# Customer information
# --------------------------------------------------
st.subheader("Customer Information")

st.write(
    "Enter the customer's account, service, and billing "
    "information to estimate churn risk."
)

with st.form("churn_prediction_form"):

    st.markdown("#### Customer Profile")

    gender = st.selectbox(
        "Gender", ["Female", "Male"]
    )

    senior_citizen = st.selectbox(
        "Senior Citizen", ["No", "Yes"]
    )

    partner = st.selectbox(
        "Partner", ["No", "Yes"]
    )

    dependents = st.selectbox(
        "Dependents", ["No", "Yes"]
    )

    tenure = st.number_input(
        "Tenure (months)",
        min_value=1,
        max_value=72,
        value=12
    )

    st.markdown("#### Services")

    phone_service = st.selectbox(
        "Phone Service", ["Yes", "No"]
    )

    if phone_service == "No":
        multiple_lines = "No phone service"
        st.caption("Multiple Lines: No phone service")
    else:
        multiple_lines = st.selectbox(
            "Multiple Lines", ["No", "Yes"]
        )

    internet_service = st.selectbox(
        "Internet Service",
        ["DSL", "Fiber optic", "No"]
    )

    if internet_service == "No":
        online_security = "No internet service"
        online_backup = "No internet service"
        device_protection = "No internet service"
        tech_support = "No internet service"
        streaming_tv = "No internet service"
        streaming_movies = "No internet service"

        st.caption(
            "Internet add-on services are unavailable "
            "when Internet Service is set to No."
        )
    else:
        online_security = st.selectbox(
            "Online Security", ["No", "Yes"]
        )

        online_backup = st.selectbox(
            "Online Backup", ["No", "Yes"]
        )

        device_protection = st.selectbox(
            "Device Protection", ["No", "Yes"]
        )

        tech_support = st.selectbox(
            "Tech Support", ["No", "Yes"]
        )

        streaming_tv = st.selectbox(
            "Streaming TV", ["No", "Yes"]
        )

        streaming_movies = st.selectbox(
            "Streaming Movies", ["No", "Yes"]
        )

    st.markdown("#### Contract and Billing")

    contract = st.selectbox(
        "Contract Type",
        ["Month-to-month", "One year", "Two year"]
    )

    paperless_billing = st.selectbox(
        "Paperless Billing", ["Yes", "No"]
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

    monthly_charges = st.number_input(
        "Monthly Charges",
        min_value=0.0,
        value=70.0,
        step=1.0
    )

    total_charges = st.number_input(
        "Total Charges",
        min_value=0.0,
        value=500.0,
        step=1.0
    )

    submitted = st.form_submit_button(
        "Predict Churn Risk",
        type="primary",
        use_container_width=True
    )

# --------------------------------------------------
# Prepare input and predict
# --------------------------------------------------
if submitted:

    numeric_features = {
        "SeniorCitizen": int(senior_citizen == "Yes"),
        "tenure": tenure,
        "MonthlyCharges": monthly_charges,
        "TotalCharges": total_charges
    }

    categorical_features = {
        "gender": gender,
        "Partner": partner,
        "Dependents": dependents,
        "PhoneService": phone_service,
        "MultipleLines": multiple_lines,
        "InternetService": internet_service,
        "OnlineSecurity": online_security,
        "OnlineBackup": online_backup,
        "DeviceProtection": device_protection,
        "TechSupport": tech_support,
        "StreamingTV": streaming_tv,
        "StreamingMovies": streaming_movies,
        "Contract": contract,
        "PaperlessBilling": paperless_billing,
        "PaymentMethod": payment_method
    }

    # Start with all expected training columns.
    encoded_row = {
        column: 0 for column in feature_columns
    }

    # Populate numerical features.
    for name, value in numeric_features.items():
        encoded_row[name] = value

    # Populate categorical dummy columns.
    # Columns absent from the training feature list
    # represent the drop_first baseline categories.
    for name, selected_value in categorical_features.items():
        dummy_column = f"{name}_{selected_value}"

        if dummy_column in encoded_row:
            encoded_row[dummy_column] = 1

    input_encoded = pd.DataFrame(
        [encoded_row],
        columns=feature_columns
    )

    # Verify feature alignment.
    expected_count = getattr(
        model, "n_features_in_", len(feature_columns)
    )

    if input_encoded.shape[1] != expected_count:
        st.error(
            "Feature mismatch: the input does not match "
            "the trained model."
        )
        st.stop()

    churn_probability = float(
        model.predict_proba(input_encoded)[0, 1]
    )

    st.divider()
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

# --------------------------------------------------
# About the model
# --------------------------------------------------
st.divider()

with st.expander("About This Model"):

    st.markdown("### Business Problem")

    st.write(
        "Customer churn occurs when customers stop using "
        "a company's services. Identifying customers at "
        "risk can help businesses prioritize retention efforts."
    )

    st.markdown("### Machine Learning Approach")

    st.write(
        "This project uses a telecommunications dataset "
        "and evaluates multiple classification models, "
        "including Logistic Regression and Random Forest."
    )

    st.write(
        "A tuned Balanced Logistic Regression model was "
        "selected to emphasize the identification of "
        "customers who may churn."
    )

    st.markdown("### Model Evaluation")

    st.write(
        "The models were evaluated using precision, recall, "
        "F1-score, and ROC-AUC. The decision threshold was "
        "adjusted to prioritize churn recall."
    )

    st.markdown("### Limitations")

    st.write(
        "Predictions reflect patterns learned from historical "
        "data. They do not establish causation or guarantee "
        "future customer behavior."
    )

    st.write(
        "Predicted probabilities should not automatically "
        "be interpreted as calibrated real-world probabilities."
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
