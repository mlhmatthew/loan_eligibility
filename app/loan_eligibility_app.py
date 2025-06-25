import sys
import os
import pandas as pd

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import streamlit as st
from loan_eligibility.data_loader import load_data
from loan_eligibility.preprocess import preprocess
from loan_eligibility.model import train_model, evaluate_model
from loan_eligibility.predict import predict_applicant

st.title("Loan Eligibility Predictor")

# Load and prepare data
df = load_data("credit.csv")  # Replace with actual path if needed
df = preprocess(df)
model, X_test, y_test, feature_order = train_model(df)

acc, cm = evaluate_model(model, X_test, y_test)

# Show model metrics
st.subheader("Model Evaluation")
if acc is not None:
    st.write(f"Accuracy: {acc:.2f}")
    st.write("Confusion Matrix:")
    st.write(cm)
else:
    st.info("No training data found. Model is running in prediction-only mode.")


# Input form
st.subheader("Predict New Applicant")
input_data = {
    "ApplicantIncome": st.number_input("Applicant Income", value=5000),
    "CoapplicantIncome": st.number_input("Coapplicant Income", value=0),
    "LoanAmount": st.number_input("Loan Amount", value=100),
    "Loan_Amount_Term": st.number_input("Loan Term", value=360),
    "Credit_History": st.selectbox("Credit History", [0.0, 1.0]),
    "Gender": st.selectbox("Gender (0=Female, 1=Male)", [0, 1]),
    "Married": st.selectbox("Married (0=No, 1=Yes)", [0, 1]),
    "Education": st.selectbox("Education (0=Graduate, 1=Not Graduate)", [0, 1]),
    "Self_Employed": st.selectbox("Self Employed (0=No, 1=Yes)", [0, 1]),
    "Property_Area": st.selectbox("Property Area (0=Rural, 1=Semiurban, 2=Urban)", [0, 1, 2]),
    "Dependents": st.selectbox("Number of Dependents (use 3 for ‘3+’)", [0, 1, 2, 3])
}

if st.button("Check Eligibility"):
    try:
        input_df = pd.DataFrame([input_data])[feature_order]
        result = model.predict(input_df)[0]
        st.success("Loan Approved" if result == 1 else "Loan Rejected")
    except Exception as e:
        st.error(f"Prediction failed: {e}")

