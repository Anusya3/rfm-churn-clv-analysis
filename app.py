import streamlit as st
import pickle
import numpy as np

# Load model and scaler
with open('churn_model.pkl', 'rb') as f:
    model = pickle.load(f)

with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

# App title
st.title("🛒 Customer Churn Prediction App")
st.write("Enter customer details below to predict if they will churn.")

# Input form
st.header("Customer Details")

frequency = st.number_input(
    "How many times has the customer purchased?",
    min_value=1,
    max_value=100,
    value=5
)

monetary = st.number_input(
    "How much has the customer spent in total? (£)",
    min_value=0,
    max_value=100000,
    value=500
)

# Predict button
if st.button("🔍 Predict Churn"):
    input_data = np.array([[frequency, monetary]])
    input_scaled = scaler.transform(input_data)
    prediction = model.predict(input_scaled)[0]
    probability = model.predict_proba(input_scaled)[0][1] * 100

    st.header("Prediction Result")

    if prediction == 1:
        st.error(f"⚠️ This customer is LIKELY TO CHURN")
        st.write(f"Churn Probability: **{probability:.1f}%**")
        st.write("**Action:** Send a retention offer immediately!")
    else:
        st.success(f"✅ This customer is LIKELY TO STAY")
        st.write(f"Churn Probability: **{probability:.1f}%**")
        st.write("**Action:** Consider upselling to this customer.")