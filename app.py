from pathlib import Path
import urllib.request
import pandas as pd
import streamlit as st
import joblib

ROOT=Path(__file__).resolve().parent
DATA=ROOT/"data"/"StudentsPerformance.csv"
MODEL=ROOT/"outputs"/"linear_regression_model.joblib"
URL="https://raw.githubusercontent.com/NadavKiani/Students-Performance-in-Exams/master/StudentsPerformance.csv"
st.set_page_config(page_title="Student Performance Predictor", page_icon="🎓", layout="centered")
st.title("🎓 Student Performance Prediction")
st.write("Estimate a student's math score from background and test-preparation details using a Linear Regression model.")

@st.cache_resource
def load_model():
    if not MODEL.exists():
        st.error("Model not found. Run `python train_model.py` first.")
        st.stop()
    return joblib.load(MODEL)

model=load_model()
with st.form("prediction_form"):
    gender=st.selectbox("Gender", ["female","male"])
    race=st.selectbox("Race / ethnicity group", ["group A","group B","group C","group D","group E"])
    education=st.selectbox("Parental level of education", ["some high school","high school","some college","associate's degree","bachelor's degree","master's degree"])
    lunch=st.selectbox("Lunch type", ["standard","free/reduced"])
    prep=st.selectbox("Test preparation course", ["none","completed"])
    submitted=st.form_submit_button("Predict math score")
if submitted:
    row=pd.DataFrame([{"gender":gender,"race_ethnicity":race,"parental_level_of_education":education,"lunch":lunch,"test_preparation_course":prep}])
    score=float(model.predict(row)[0])
    score=max(0,min(100,score))
    st.metric("Estimated math score", f"{score:.1f} / 100")
    st.caption("This is a statistical estimate from a small, observational dataset—not a guaranteed result or a judgment of a student's ability.")

st.divider()
st.caption("Dataset: Students Performance in Exams (Kaggle). Predictions reflect patterns in this dataset and should not be used to make high-stakes decisions about students.")
