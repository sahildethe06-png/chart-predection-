import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import LabelEncoder

st.title("Iris Species Prediction")

# Load the trained model and LabelEncoder
model = joblib.load('iris_model.pkl')
le = joblib.load('label_encoder.pkl')

# Input labels
sepal_length = st.number_input("Sepal Length (cm)", min_value=0.0, max_value=10.0, value=5.0)
sepal_width = st.number_input("Sepal Width (cm)", min_value=0.0, max_value=10.0, value=3.0)
petal_length = st.number_input("Petal Length (cm)", min_value=0.0, max_value=10.0, value=4.0)
petal_width = st.number_input("Petal Width (cm)", min_value=0.0, max_value=10.0, value=1.5)

# Prediction button
if st.button("Predict"):
    input_data = np.array([[sepal_length,
                             sepal_width, petal_length,
                             petal_width]])
    prediction_numeric = model.predict(input_data)
    # Inverse transform the numerical prediction back to the original species name
    species_predicted = le.inverse_transform(prediction_numeric)
    st.success(f"The predicted species is: {species_predicted[0]}")
