import streamlit as st
import pickle
import numpy as np

# ==========================
# Load Saved Model
# ==========================

model = pickle.load(open("xgboost_model.pkl", "rb"))
label_encoder = pickle.load(open("gender_encoder.pkl", "rb"))

st.set_page_config(
    page_title="Customer Retention Prediction",
    page_icon="🏦",
    layout="centered"
)

st.title("🏦 Customer Retention Prediction")
st.write("Predict whether a customer will leave the bank or stay.")

st.divider()

# ==========================
# User Inputs
# ==========================

credit_score = st.number_input(
    "Credit Score",
    min_value=300,
    max_value=900,
    value=600
)

geography = st.selectbox(
    "Country",
    ["France", "Germany", "Spain"]
)

gender = st.selectbox(
    "Gender",
    ["Male", "Female"]
)

age = st.slider(
    "Age",
    18,
    100,
    35
)

tenure = st.slider(
    "Years with Bank",
    0,
    10,
    5
)

balance = st.number_input(
    "Bank Balance",
    min_value=0.0,
    value=50000.0
)

num_products = st.slider(
    "Number of Products",
    1,
    4,
    1
)

has_credit_card = st.selectbox(
    "Has Credit Card?",
    ["Yes", "No"]
)

is_active = st.selectbox(
    "Is Active Member?",
    ["Yes", "No"]
)

estimated_salary = st.number_input(
    "Estimated Salary",
    min_value=0.0,
    value=50000.0
)

st.divider()

# ==========================
# Prediction Button
# ==========================

if st.button("Predict"):

    # One-Hot Encoding Geography
    geo_france = 1 if geography == "France" else 0
    geo_germany = 1 if geography == "Germany" else 0
    geo_spain = 1 if geography == "Spain" else 0

    # Label Encoding Gender
    gender = label_encoder.transform([gender])[0]

    has_credit_card = 1 if has_credit_card == "Yes" else 0
    is_active = 1 if is_active == "Yes" else 0

    features = np.array([[
        geo_france,
        geo_germany,
        geo_spain,
        credit_score,
        gender,
        age,
        tenure,
        balance,
        num_products,
        has_credit_card,
        is_active,
        estimated_salary
    ]])

    prediction = model.predict(features)

    probability = model.predict_proba(features)

    st.divider()

    if prediction[0] == 1:
        st.error("❌ Customer is likely to LEAVE the bank.")
    else:
        st.success("✅ Customer is likely to STAY with the bank.")

    st.write(f"**Retention Probability:** {probability[0][0]*100:.2f}%")
    st.write(f"**Churn Probability:** {probability[0][1]*100:.2f}%")