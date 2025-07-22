# =================================================================
# COMPLETE CODE FOR STEP 6: STREAMLIT WEB APPLICATION
# =================================================================

import streamlit as st
import pandas as pd
import joblib
import numpy as np

# --- 1. Load the Saved Model ---
# Load the pipeline object you saved from the notebook
# This pipeline includes the scaler and the final logistic regression model
try:
    model = joblib.load('final_churn_model_tuned.joblib')
except FileNotFoundError:
    st.error("Model file not found. Please run your notebook to train and save the model first.")
    st.stop()

# --- 2. Set up the Streamlit page ---
st.set_page_config(page_title="Customer Churn Predictor", page_icon="🚀", layout="wide")
st.title('Customer Churn Prediction App 🔮')
st.write("This app uses a machine learning model to predict whether a customer is likely to churn. Please provide the customer's details below.")


# --- 3. Create the User Input Form ---
# Use columns for a cleaner layout
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

# More services in an expandable section
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
    # Create a dictionary with the user's inputs
    input_data = {
        'gender': gender,
        'Partner': partner,
        'Dependents': dependents,
        'tenure': tenure,
        'PhoneService': phone_service,
        'MultipleLines': multiple_lines,
        'OnlineSecurity': online_security,
        'OnlineBackup': online_backup,
        'DeviceProtection': device_protection,
        'TechSupport': tech_support,
        'StreamingTV': streaming_tv,
        'StreamingMovies': streaming_movies,
        'Contract': contract,
        'PaperlessBilling': paperless_billing,
        'PaymentMethod': payment_method,
        'MonthlyCharges': monthly_charges,
    }

    # IMPORTANT: The model was trained on TotalCharges and our new feature.
    # We must create these features for the prediction input.
    input_data['TotalCharges'] = input_data['tenure'] * input_data['MonthlyCharges']
    input_data['tenure_monthly_ratio'] = input_data['tenure'] / (input_data['MonthlyCharges'] + 1e-6)

    # Convert the dictionary to a pandas DataFrame
    input_df = pd.DataFrame([input_data])
    
    # Make the prediction
    prediction_proba = model.predict_proba(input_df)[0]
    prediction = model.predict(input_df)[0]
    
    # Display the result
    st.header('Prediction Result')
    if prediction == 1:
        st.error(f'Prediction: This customer is likely to CHURN.', icon="🚨")
        st.write(f"Confidence (Probability of Churn): **{prediction_proba[1]:.2%}**")
    else:
        st.success(f'Prediction: This customer is likely to STAY.', icon="✅")
        st.write(f"Confidence (Probability of Staying): **{prediction_proba[0]:.2%}**")