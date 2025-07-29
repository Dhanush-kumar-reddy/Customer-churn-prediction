# =================================================================
# CORRECTED CODE FOR app.py (with Probability Display)
# =================================================================

import streamlit as st
import pandas as pd
import joblib
import numpy as np

# --- 1. Load Saved Objects ---
try:
    model = joblib.load('final_churn_model_tuned.joblib')
    train_cols = joblib.load('training_columns.joblib')
except FileNotFoundError:
    st.error("Model or column file not found. Please run your notebook to create them first.")
    st.stop()

# --- 2. Set up the Streamlit page ---
st.set_page_config(page_title="Customer Churn Predictor", page_icon="🚀", layout="wide")
st.title('Customer Churn Prediction App 🔮')
st.write("This app uses a machine learning model to predict whether a customer is likely to churn. Please provide the customer's details below.")


# --- 3. Create the User Input Form ---
col1, col2 = st.columns(2)
with col1:
    st.header("Customer Details")
    gender = st.selectbox('Gender', ['Male', 'Female'])
    partner = st.selectbox('Has a Partner?', ['Yes', 'No'])
    dependents = st.selectbox('Has Dependents?', ['Yes', 'No'])
    phone_service = st.selectbox('Phone Service?', ['Yes', 'No'])
    paperless_billing = st.selectbox('Paperless Billing?', ['Yes', 'No'])
with col2:
    st.header("Contract & Financials")
    tenure = st.slider('Tenure (in months)', 1, 72, 12)
    monthly_charges = st.slider('Monthly Charges ($)', 18.0, 120.0, 70.0)
    contract = st.selectbox('Contract Type', ['Month-to-month', 'One year', 'Two year'])
    payment_method = st.selectbox('Payment Method', ['Electronic check', 'Mailed check', 'Bank transfer (automatic)', 'Credit card (automatic)'])
with st.expander("Subscribed Services (click to expand)"):
    sub_col1, sub_col2, sub_col3 = st.columns(3)
    with sub_col1:
        multiple_lines = st.selectbox('Multiple Lines', ['No', 'Yes', 'No phone service'])
        online_security = st.selectbox('Online Security', ['No', 'Yes', 'No internet service'])
        online_backup = st.selectbox('Online Backup', ['No', 'Yes', 'No internet service'])
    with sub_col2:
        device_protection = st.selectbox('Device Protection', ['No', 'Yes', 'No internet service'])
        tech_support = st.selectbox('Tech Support', ['No', 'Yes', 'No internet service'])
        streaming_tv = st.selectbox('Streaming TV', ['No', 'Yes', 'No internet service'])
    with sub_col3:
        streaming_movies = st.selectbox('Streaming Movies', ['No', 'Yes', 'No internet service'])


# --- 4. Process Inputs and Make Prediction ---
if st.button('Predict Churn', key='predict'):
    input_data = {
        'gender': gender, 'Partner': partner, 'Dependents': dependents, 'tenure': tenure,
        'PhoneService': phone_service, 'MultipleLines': multiple_lines, 'OnlineSecurity': online_security,
        'OnlineBackup': online_backup, 'DeviceProtection': device_protection, 'TechSupport': tech_support,
        'StreamingTV': streaming_tv, 'StreamingMovies': streaming_movies, 'Contract': contract,
        'PaperlessBilling': paperless_billing, 'PaymentMethod': payment_method,
        'MonthlyCharges': monthly_charges,
    }
    
    input_data['TotalCharges'] = input_data['tenure'] * input_data['MonthlyCharges']
    input_data['tenure_monthly_ratio'] = input_data['tenure'] / (input_data['MonthlyCharges'] + 1e-6)

    input_df = pd.DataFrame([input_data])
    input_df_encoded = pd.get_dummies(input_df)
    final_df = input_df_encoded.reindex(columns=train_cols, fill_value=0)
    
    prediction_proba = model.predict_proba(final_df)[0]
    prediction = model.predict(final_df)[0]
    
    st.header('Prediction Result')
    # **NEW:** Display the probabilities
    st.write(f"Raw Probabilities: STAY: **{prediction_proba[0]:.2%}**, CHURN: **{prediction_proba[1]:.2%}**")

    if prediction == 1:
        st.error(f'Prediction: This customer is likely to CHURN.', icon="🚨")
    else:
        st.success(f'Prediction: This customer is likely to STAY.', icon="✅")