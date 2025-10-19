import numpy as np 
import pandas as pd
import joblib
import streamlit as st
from tensorflow.keras.models import load_model


st.set_page_config(page_title="Breast Cancer Prediction", layout="wide")
st.title("🧬 Breast Cancer Prediction App")
st.write("Enter the cell nucleus feature measurements below to predict if the tumor is **Malignant** or **Benign**.")
model = load_model('models/breast_cancer_prediction.h5')

feature_names = [
    "radius_mean", "texture_mean", "perimeter_mean", "area_mean", "smoothness_mean", "compactness_mean",
    "concavity_mean", "concave points_mean", "symmetry_mean", "fractal_dimension_mean",
    "radius_se", "texture_se", "perimeter_se", "area_se", "smoothness_se", "compactness_se",
    "concavity_se", "concave points_se", "symmetry_se", "fractal_dimension_se",
    "radius_worst", "texture_worst", "perimeter_worst", "area_worst", "smoothness_worst", "compactness_worst",
    "concavity_worst", "concave points_worst", "symmetry_worst", "fractal_dimension_worst"
]

col1, col2, col3 = st.columns(3)


with col1:
    radius_mean = st.number_input("Mean Radius", 0.0, help="Average distance from center to perimeter points")
    texture_mean = st.number_input("Mean Texture", 0.0, help="Standard deviation of gray-scale values")
    perimeter_mean = st.number_input("Mean Perimeter", 0.0, help="Average length of nucleus boundary")
    area_mean = st.number_input("Mean Area", 0.0, help="Average area of the cell nucleus")
    smoothness_mean = st.number_input("Mean Smoothness", 0.0, help="Local variation in radius lengths")
    compactness_mean = st.number_input("Mean Compactness", 0.0, help="(Perimeter² / Area - 1.0)")
    concavity_mean = st.number_input("Mean Concavity", 0.0, help="Severity of concave portions of the contour")
    concave_points_mean = st.number_input("Mean Concave Points", 0.0, help="Number of concave portions on contour")
    symmetry_mean = st.number_input("Mean Symmetry", 0.0, help="Measure of how symmetrical the cell is")
    fractal_dimension_mean = st.number_input("Mean Fractal Dimension", 0.0, help="Cell boundary complexity")

with col2:
    radius_se = st.number_input("Radius SE", 0.0, help="Standard error of mean radius")
    texture_se = st.number_input("Texture SE", 0.0, help="Standard error of mean texture")
    perimeter_se = st.number_input("Perimeter SE", 0.0, help="Standard error of mean perimeter")
    area_se = st.number_input("Area SE", 0.0, help="Standard error of mean area")
    smoothness_se = st.number_input("Smoothness SE", 0.0, help="Standard error of mean smoothness")
    compactness_se = st.number_input("Compactness SE", 0.0, help="Standard error of mean compactness")
    concavity_se = st.number_input("Concavity SE", 0.0, help="Standard error of mean concavity")
    concave_points_se = st.number_input("Concave Points SE", 0.0, help="Standard error of mean concave points")
    symmetry_se = st.number_input("Symmetry SE", 0.0, help="Standard error of mean symmetry")
    fractal_dimension_se = st.number_input("Fractal Dimension SE", 0.0, help="Standard error of mean fractal dimension")

with col3:
    radius_worst = st.number_input("Worst Radius", 0.0, help="Largest mean value for radius")
    texture_worst = st.number_input("Worst Texture", 0.0, help="Largest mean value for texture")
    perimeter_worst = st.number_input("Worst Perimeter", 0.0, help="Largest mean value for perimeter")
    area_worst = st.number_input("Worst Area", 0.0, help="Largest mean value for area")
    smoothness_worst = st.number_input("Worst Smoothness", 0.0, help="Largest mean value for smoothness")
    compactness_worst = st.number_input("Worst Compactness", 0.0, help="Largest mean value for compactness")
    concavity_worst = st.number_input("Worst Concavity", 0.0, help="Largest mean value for concavity")
    concave_points_worst = st.number_input("Worst Concave Points", 0.0, help="Largest mean value for concave points")
    symmetry_worst = st.number_input("Worst Symmetry", 0.0, help="Largest mean value for symmetry")
    fractal_dimension_worst = st.number_input("Worst Fractal Dimension", 0.0, help="Largest mean value for fractal dimension")


input_data = pd.DataFrame([[
    radius_mean, texture_mean, perimeter_mean, area_mean, smoothness_mean, compactness_mean,
    concavity_mean, concave_points_mean, symmetry_mean, fractal_dimension_mean,
    radius_se, texture_se, perimeter_se, area_se, smoothness_se, compactness_se,
    concavity_se, concave_points_se, symmetry_se, fractal_dimension_se,
    radius_worst, texture_worst, perimeter_worst, area_worst, smoothness_worst, compactness_worst,
    concavity_worst, concave_points_worst, symmetry_worst, fractal_dimension_worst
]], columns=feature_names)


with st.expander("🔎 View Entered Data"):
    st.dataframe(input_data)

center_col = st.columns([1, 1, 1])[1]
with center_col:
    if st.button("🔍 Predict Diagnosis", use_container_width=True):
        prediction = model.predict(input_data)[0][0]
        result = "Malignant" if prediction > 0.5 else "Benign"
        if result == "Malignant":
            st.error(f'The model predicts the cancer to be {result}')
        else:
            st.success(f'The model predicts the cancer to be {result}')

