import streamlit as st
import numpy as np
import joblib

model = joblib.load("best_model.pkl")
scaler = joblib.load("scaler.pkl")
feature_columns = joblib.load("feature_columns.pkl")
model_name = joblib.load("model_name.pkl")

st.set_page_config(
    page_title="Medical Cost Prediction",
    page_icon="🏥",
    layout="wide"
)

st.title("🏥 Medical Cost Prediction System")
st.write("Enter patient details to predict annual medical cost.")

st.divider()

st.subheader("Patient Details")

col1, col2 = st.columns(2)

with col1:
    age = st.number_input(
        "Age",
        min_value=1,
        max_value=100,
        value=30
    )

    income = st.number_input(
        "Income",
        min_value=0.0,
        value=50000.0
    )

    household_size = st.number_input(
        "Household Size",
        min_value=1,
        max_value=20,
        value=3
    )

    dependents = st.number_input(
        "Dependents",
        min_value=0,
        max_value=20,
        value=1
    )

with col2:
    bmi = st.number_input(
        "BMI",
        min_value=10.0,
        max_value=60.0,
        value=25.0
    )

    visits_last_year = st.number_input(
        "Visits Last Year",
        min_value=0,
        max_value=50,
        value=2
    )

    risk_score = st.number_input(
        "Risk Score",
        min_value=0.0,
        max_value=100.0,
        value=20.0
    )

st.divider()

if st.button("🔍 Predict Medical Cost", use_container_width=True):

    sample = np.zeros(len(feature_columns))

    values = {
        "age": age,
        "income": income,
        "household_size": household_size,
        "dependents": dependents,
        "bmi": bmi,
        "visits_last_year": visits_last_year,
        "risk_score": risk_score
    }

    for i, column in enumerate(feature_columns):
        if column in values:
            sample[i] = values[column]

    sample = sample.reshape(1, -1)

    if model_name in [
        "Linear Regression",
        "Ridge Regression",
        "Lasso Regression"
    ]:
        sample = scaler.transform(sample)

    prediction = model.predict(sample)[0]

    st.success(
        "Predicted Annual Medical Cost: ₹{:,.2f}".format(prediction)
    )

    st.info("Model Used: " + model_name)

st.divider()

st.caption("Medical Cost Prediction System | Built with Streamlit")