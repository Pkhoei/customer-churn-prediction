import joblib
import pandas as pd
import streamlit as st
from supabase import create_client
import logging

# --------------------------------------------------
# Page configuration
# --------------------------------------------------
st.set_page_config(
    page_title="Customer Churn Risk Predictor",
    page_icon="📉",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# --------------------------------------------------
# Styling
# --------------------------------------------------
st.markdown(
    """
    <style>
    .block-container {
        max-width: 1080px;
        padding-top: 1.5rem;
        padding-bottom: 2.5rem;
    }

    .app-subtitle {
        color: #64748b;
        font-size: 1rem;
        line-height: 1.6;
    }

    .section-intro {
        color: #64748b;
        margin-bottom: 0.5rem;
    }

    .risk-bar {
        position: relative;
        height: 18px;
        border-radius: 10px;
        background: #e5e7eb;
        margin-top: 20px;
        margin-bottom: 14px;
    }

    .risk-fill {
        height: 100%;
        border-radius: 10px;
    }

    .threshold-marker {
        position: absolute;
        top: -7px;
        height: 32px;
        width: 3px;
        background: #334155;
        transform: translateX(-50%);
        border-radius: 2px;
    }

    .validation-note {
        color: #64748b;
        font-size: 0.9rem;
        margin-top: 0.5rem;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# --------------------------------------------------
# Load trained model
# --------------------------------------------------
@st.cache_resource
def load_artifacts():
    model = joblib.load("model/churn_model.joblib")
    columns = list(
        joblib.load("model/feature_columns.joblib")
    )
    threshold = float(
        joblib.load("model/decision_threshold.joblib")
    )
    return model, columns, threshold


model, feature_columns, threshold = load_artifacts()

# --------------------------------------------------
# Anonymous prediction event tracking
# --------------------------------------------------
logger = logging.getLogger(__name__)

@st.cache_resource
def get_supabase_client():
    url = st.secrets["SUPABASE_URL"]
    key = st.secrets["SUPABASE_KEY"]
    return create_client(url, key)


def record_prediction_event(risk_label):
    try:
        get_supabase_client().table("prediction_events").insert(
            {"risk_label": risk_label}
        ).execute()
    except Exception:
        # Logging must not prevent users from getting predictions.
        logger.warning("Prediction event could not be recorded")


# --------------------------------------------------
# Header
# --------------------------------------------------
st.title("📉 Customer Churn Risk Predictor")

st.markdown(
    "An interactive machine learning application for "
    "estimating customer churn risk in telecommunications. "
    "Explore customer profiles, review risk predictions, "
    "and understand how predictive analytics can support "
    "customer retention decisions."
)

st.divider()

# --------------------------------------------------
# Customer information
# --------------------------------------------------
st.header("Customer Information")

st.write(
    "Complete the customer profile, service configuration, "
    "and billing details to generate a churn-risk estimate."
)

with st.form("churn_prediction_form"):

    st.subheader("01 · Customer Profile")

    profile_col1, profile_col2 = st.columns(2)

    with profile_col1:
        gender = st.selectbox(
            "Gender", ["Female", "Male"]
        )

        partner = st.selectbox(
            "Partner", ["No", "Yes"]
        )

        tenure = st.number_input(
            "Tenure (months)",
            min_value=1,
            max_value=72,
            value=12
        )

    with profile_col2:
        senior_citizen = st.selectbox(
            "Senior Citizen", ["No", "Yes"]
        )

        dependents = st.selectbox(
            "Dependents", ["No", "Yes"]
        )

    st.divider()
    st.subheader("02 · Services")

    service_col1, service_col2 = st.columns(2)

    with service_col1:
        phone_service = st.selectbox(
            "Phone Service", ["Yes", "No"]
        )

        if phone_service == "No":
            multiple_lines = "No phone service"
            st.caption(
                "Multiple Lines: No phone service"
            )
        else:
            multiple_lines = st.selectbox(
                "Multiple Lines", ["No", "Yes"]
            )

    with service_col2:
        internet_service = st.selectbox(
            "Internet Service",
            ["DSL", "Fiber optic", "No"]
        )

        if internet_service == "No":
            st.caption(
                "Internet add-on services are unavailable."
            )

    if internet_service == "No":
        online_security = "No internet service"
        online_backup = "No internet service"
        device_protection = "No internet service"
        tech_support = "No internet service"
        streaming_tv = "No internet service"
        streaming_movies = "No internet service"

    else:
        addon_col1, addon_col2 = st.columns(2)

        with addon_col1:
            online_security = st.selectbox(
                "Online Security", ["No", "Yes"]
            )

            online_backup = st.selectbox(
                "Online Backup", ["No", "Yes"]
            )

            device_protection = st.selectbox(
                "Device Protection", ["No", "Yes"]
            )

        with addon_col2:
            tech_support = st.selectbox(
                "Tech Support", ["No", "Yes"]
            )

            streaming_tv = st.selectbox(
                "Streaming TV", ["No", "Yes"]
            )

            streaming_movies = st.selectbox(
                "Streaming Movies", ["No", "Yes"]
            )

    st.divider()
    st.subheader("03 · Contract and Billing")

    billing_col1, billing_col2 = st.columns(2)

    with billing_col1:
        contract = st.selectbox(
            "Contract Type",
            ["Month-to-month", "One year", "Two year"]
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

    with billing_col2:
        paperless_billing = st.selectbox(
            "Paperless Billing", ["Yes", "No"]
        )

        total_charges = st.number_input(
            "Total Charges",
            min_value=0.0,
            value=500.0,
            step=1.0
        )

    submitted = st.form_submit_button(
        "Analyze Customer Churn Risk",
        type="primary",
        use_container_width=True
    )

# --------------------------------------------------
# Validation and prediction
# --------------------------------------------------
if submitted:

    validation_errors = []
    validation_warnings = []

    if tenure < 1:
        validation_errors.append(
            "Tenure must be at least one month."
        )

    if monthly_charges <= 0:
        validation_errors.append(
            "Monthly Charges must be greater than zero."
        )

    if total_charges <= 0:
        validation_errors.append(
            "Total Charges must be greater than zero."
        )

    if monthly_charges > 0 and total_charges > 0:

        estimated_total = tenure * monthly_charges

        if estimated_total > 0:
            difference_ratio = abs(
                total_charges - estimated_total
            ) / estimated_total

            if difference_ratio > 0.50:
                validation_warnings.append(
                    "Total Charges differ substantially "
                    "from Tenure × Monthly Charges. "
                    "Please confirm the values. "
                    "Historical pricing, discounts, or "
                    "service changes may explain this."
                )

    if validation_errors:
        st.divider()
        st.subheader("Input Validation")

        for message in validation_errors:
            st.error(message)

        st.warning(
            "Please correct the invalid inputs "
            "before generating a prediction."
        )

        st.stop()

    if validation_warnings:
        st.divider()
        st.subheader("Input Validation")

        for message in validation_warnings:
            st.warning(message)

    else:
        st.caption(
            "✓ Input validation completed. "
            "No issues detected by the current checks."
        )

    # --------------------------------------------------
    # Model input preparation
    # --------------------------------------------------
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

    encoded_row = {
        column: 0 for column in feature_columns
    }

    for name, value in numeric_features.items():
        encoded_row[name] = value

    for name, selected_value in categorical_features.items():
        dummy_column = f"{name}_{selected_value}"

        if dummy_column in encoded_row:
            encoded_row[dummy_column] = 1

    input_encoded = pd.DataFrame(
        [encoded_row],
        columns=feature_columns
    )

    # --------------------------------------------------
    # Feature verification
    # --------------------------------------------------
    expected_count = getattr(
        model,
        "n_features_in_",
        len(feature_columns)
    )

    if input_encoded.shape[1] != expected_count:
        st.error(
            "Feature mismatch: input does not match "
            "the trained model."
        )
        st.stop()

    if hasattr(model, "feature_names_in_"):
        if list(model.feature_names_in_) != list(
            input_encoded.columns
        ):
            st.error(
                "Feature order mismatch: input columns "
                "do not match the trained model."
            )
            st.stop()

    # --------------------------------------------------
    # Prediction
    # --------------------------------------------------
    churn_probability = float(
        model.predict_proba(input_encoded)[0, 1]
    )

    is_high_risk = churn_probability >= threshold

    risk_label = (
        "High Churn Risk"
        if is_high_risk
        else "Low Churn Risk"
    )

    risk_color = (
        "#DC6B61"
        if is_high_risk
        else "#26977B"
    )

    record_prediction_event(risk_label)

    probability_percent = churn_probability * 100
    threshold_percent = threshold * 100

    # --------------------------------------------------
    # Prediction results - native Streamlit
    # --------------------------------------------------
    st.divider()
    st.header("Prediction Results")

    with st.container(border=True):

        result_col1, result_col2 = st.columns(2)

        with result_col1:
            st.metric(
                "Predicted Churn Probability",
                f"{churn_probability:.1%}"
            )

        with result_col2:
            st.metric(
                "Decision Threshold",
                f"{threshold:.0%}"
            )

        if is_high_risk:
            st.error(risk_label)
        else:
            st.success(risk_label)

        # Single-line HTML avoids Markdown code rendering.
        bar_html = (
            '<div class="risk-bar">'
            f'<div class="risk-fill" style="width:{probability_percent:.2f}%;'
            f'background:{risk_color};"></div>'
            f'<div class="threshold-marker" '
            f'style="left:{threshold_percent:.2f}%;"></div>'
            '</div>'
        )

        st.markdown(
            bar_html,
            unsafe_allow_html=True
        )

        st.caption(
            "Risk score (colored bar) · "
            f"Decision threshold {threshold:.0%} "
            "(vertical marker)"
        )

        if is_high_risk:
            st.write(
                "The customer's predicted churn score "
                "is at or above the decision threshold. "
                "This profile is classified as higher risk."
            )
        else:
            st.write(
                "The customer's predicted churn score "
                "is below the decision threshold. "
                "This profile is classified as lower risk."
            )

    st.caption(
        "The prediction is a model-based estimate, "
        "not a guaranteed outcome or a verified "
        "real-world probability."
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

st.caption(
    "Customer Churn Prediction | Machine Learning Portfolio Project"
)
