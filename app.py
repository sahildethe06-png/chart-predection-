import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import LabelEncoder

st.title("Iris Species Prediction")

# Load the trained model
model = joblib.load('iris_model.pkl')

# Initialize and fit LabelEncoder (assuming 'y' contains the original species labels)
# This is crucial for inverse_transform
# For deployment, you would typically save and load the fitted LabelEncoder as well.
# For this example, we'll refit it using the original dataset's unique species.
# In a real application, ensure consistency with the training phase.
le = LabelEncoder()
# Fit the LabelEncoder with the known species from the dataset
# Assuming df (or y) is available, or you know the unique species beforehand.
# For this example, we'll hardcode them since we know the Iris species.
le.fit(['Iris-setosa', 'Iris-versicolor', 'Iris-virginica'])

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
    species_predicted = le.inverse_transform(prediction_numeric)
    st.success(f"The predicted species is: {species_predicted[0]}")
