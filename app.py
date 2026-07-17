import streamlit as st
import pandas as pd
import numpy as np
import joblib

st.set_page_config(
    page_title="Customer Churn Predictor",
    page_icon="📡",
    layout="centered"
)

@st.cache_resource
def load_artifacts():
    model  = joblib.load("customer_churn_model.pkl")     # your Random Forest pkl
    scaler = joblib.load("scaler.pkl")
    cols   = joblib.load("col_w.pkl")
    return model, scaler, cols

model, scaler, cols = load_artifacts()

def preprocess(tenure, senior_citizen, partner, dependents, paperless_billing,
               multiple_lines, internet_service, online_security, online_backup,
               device_protection, tech_support, streaming_tv, streaming_movies,
               contract, payment_method):

    data = {col: 0 for col in cols}

    # Binary fields
    data['tenure']           = tenure
    data['SeniorCitizen']    = 1 if senior_citizen == 'Yes' else 0
    data['Partner']          = 1 if partner         == 'Yes' else 0
    data['Dependents']       = 1 if dependents      == 'Yes' else 0
    data['PaperlessBilling'] = 1 if paperless_billing == 'Yes' else 0

    # One-hot fields
    for key, val in [
        ('MultipleLines',     multiple_lines),
        ('InternetService',   internet_service),
        ('OnlineSecurity',    online_security),
        ('OnlineBackup',      online_backup),
        ('DeviceProtection',  device_protection),
        ('TechSupport',       tech_support),
        ('StreamingTV',       streaming_tv),
        ('StreamingMovies',   streaming_movies),
        ('Contract',          contract),
        ('PaymentMethod',     payment_method),
    ]:
        col_name = f"{key}_{val}"
        if col_name in data:
            data[col_name] = 1

    # TenureGroup
    if tenure <= 12:
        data['TenureGroup_0-1 Year']   = 1
    elif tenure <= 48:
        data['TenureGroup_2-4 Years']  = 1
    else:
        data['TenureGroup_4-6 Years']  = 1

    df_input = pd.DataFrame([data])
    df_input['tenure'] = scaler.transform([[tenure]])[0][0]

    return df_input

st.title("📡 Customer Churn Prediction")
st.caption("Predict whether a customer is likely to churn based on their profile.")
st.divider()

# Section 1 — Customer Profile
st.subheader("Customer Profile")
col1, col2, col3 = st.columns(3)

with col1:
    tenure         = st.slider("Tenure (months)", 0, 72, 12)
    senior_citizen = st.selectbox("Senior Citizen", ["No", "Yes"])

with col2:
    partner    = st.selectbox("Partner", ["No", "Yes"])
    dependents = st.selectbox("Dependents", ["No", "Yes"])

with col3:
    paperless_billing = st.selectbox("Paperless Billing", ["No", "Yes"])
    contract = st.selectbox("Contract", [
        "Month-to-month", "One year", "Two year"
    ])

st.divider()

# Section 2 — Services
st.subheader("Services Subscribed")
col4, col5 = st.columns(2)

with col4:
    multiple_lines   = st.selectbox("Multiple Lines",    ["No", "Yes"])
    internet_service = st.selectbox("Internet Service",  ["DSL", "Fiber optic", "No"])
    online_security  = st.selectbox("Online Security",   ["No", "Yes", "No internet service"])
    online_backup    = st.selectbox("Online Backup",     ["No", "Yes", "No internet service"])
    device_protection= st.selectbox("Device Protection", ["No", "Yes", "No internet service"])

with col5:
    tech_support     = st.selectbox("Tech Support",      ["No", "Yes", "No internet service"])
    streaming_tv     = st.selectbox("Streaming TV",      ["No", "Yes", "No internet service"])
    streaming_movies = st.selectbox("Streaming Movies",  ["No", "Yes", "No internet service"])
    payment_method   = st.selectbox("Payment Method", [
        "Electronic check", "Mailed check",
        "Bank transfer (automatic)", "Credit card (automatic)"
    ])

st.divider()

if st.button("Predict Churn", use_container_width=True, type="primary"):
    input_df = preprocess(
        tenure, senior_citizen, partner, dependents, paperless_billing,
        multiple_lines, internet_service, online_security, online_backup,
        device_protection, tech_support, streaming_tv, streaming_movies,
        contract, payment_method
    )

    prediction = model.predict(input_df)[0]
    probability = model.predict_proba(input_df)[0][1]

    st.divider()

    if prediction == 1:
        st.error(f"⚠️ High Churn Risk — {probability * 100:.1f}% probability")
        st.markdown("This customer is likely to churn. Consider a retention offer.")
    else:
        st.success(f"✅ Low Churn Risk — {probability * 100:.1f}% probability")
        st.markdown("This customer is likely to stay.")

    # Probability bar
    st.markdown("**Churn Probability**")
    st.progress(float(probability))

st.divider()
st.caption("Educational project only. Not intended for production business decisions without validation.")