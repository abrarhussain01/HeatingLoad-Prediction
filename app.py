import streamlit as st
import joblib
import numpy as np

# 🌡️ Load trained model and polynomial transformer
poly_path = "abrar/poly_transformer.pkl"  # your trained LinearRegression model
model_path = "abrar/polynomial_regression_model.pkl"    # your PolynomialFeatures transformer

model = joblib.load(open(model_path, "rb"))
poly = joblib.load(open(poly_path, "rb"))

# 🎨 Streamlit Page Setup
st.set_page_config(page_title="Heating Load Prediction", layout="centered")

st.title("🏠 Heating Load Prediction App")
st.write("Provide the building parameters below to estimate the **Heating Load (Energy Efficiency)**.")

# 🧾 Input fields (only selected features)
Relative_Compactness = st.number_input("Relative Compactness", min_value=0.0, max_value=1.0, value=0.75)
Surface_Area = st.number_input("Surface Area (m²)", min_value=200.0, max_value=1000.0, value=500.0)
Wall_Area = st.number_input("Wall Area (m²)", min_value=100.0, max_value=400.0, value=200.0)
Roof_Area = st.number_input("Roof Area (m²)", min_value=100.0, max_value=400.0, value=150.0)
Overall_Height = st.number_input("Overall Height (m)", min_value=2.0, max_value=10.0, value=3.5)

# 🧮 Predict Button
if st.button("Predict Heating Load"):
    input_data = np.array([[Relative_Compactness, Surface_Area, Wall_Area, Roof_Area, Overall_Height]])

    # Apply polynomial transformation


    # Predict using trained model
    prediction = model.predict(input_data)[0]
    st.success(f"Predicted Heating Load: {prediction:.2f}")

st.markdown("---")

