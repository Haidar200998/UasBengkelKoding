import streamlit as st
import numpy as np
import joblib

# Memuat model dan preprocessor yang disimpan
scaler = joblib.load('scaler.pkl')
imputer = joblib.load('imputer.pkl')
log_reg_model = joblib.load('log_reg_model.pkl')
rf_model = joblib.load('rf_model.pkl')
svm_model = joblib.load('svm_model.pkl')

# Membuat antarmuka pengguna di Streamlit
st.title("Heart Disease Prediction App")

# Membuat form input untuk fitur-fitur
with st.form("prediction_form"):
    st.write("Please enter the following information:")
    age = st.number_input("Age", min_value=0, max_value=120, value=50)
    sex = st.selectbox("Sex", options=[0, 1], format_func=lambda x: "Female" if x == 0 else "Male")
    cp = st.slider("Chest Pain Type", min_value=0, max_value=3, value=1)
    trestbps = st.number_input("Resting Blood Pressure", min_value=0, max_value=200, value=120)
    chol = st.number_input("Serum Cholesterol in mg/dl", min_value=0, max_value=600, value=200)
    fbs = st.selectbox("Fasting Blood Sugar > 120 mg/dl", options=[0, 1], format_func=lambda x: "No" if x == 0 else "Yes")
    restecg = st.selectbox("Resting Electrocardiographic Results", options=[0, 1, 2])
    thalach = st.number_input("Maximum Heart Rate Achieved", min_value=60, max_value=220, value=140)
    exang = st.selectbox("Exercise Induced Angina", options=[0, 1], format_func=lambda x: "No" if x == 0 else "Yes")
    oldpeak = st.number_input("ST Depression Induced by Exercise", min_value=0.0, max_value=6.0, value=1.0, step=0.1)
    slope = st.selectbox("Slope of the Peak Exercise ST Segment", options=[0, 1, 2])
    ca = st.slider("Number of Major Vessels Colored by Fluoroscopy", 0, 4, 1)
    thal = st.selectbox("Thalassemia", options=[1, 2, 3, 0], format_func=lambda x: "Normal" if x == 1 else "Fixed Defect" if x == 2 else "Reversable Defect" if x == 3 else "Unknown")

    submit_button = st.form_submit_button("Predict")

# Proses input dan buat prediksi menggunakan model
if submit_button:
    input_data = np.array([[age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal]])
    input_data = scaler.transform(imputer.transform(input_data))

    prediction_log_reg = log_reg_model.predict(input_data)
    prediction_rf = rf_model.predict(input_data)
    prediction_svm = svm_model.predict(input_data)

    # Tampilkan hasil prediksi
    st.subheader("Prediction Results:")
    st.write("Logistic Regression Prediction:", "Positive" if prediction_log_reg[0] == 1 else "Negative")
    st.write("Random Forest Prediction:", "Positive" if prediction_rf[0] == 1 else "Negative")
    st.write("SVM Prediction:", "Positive" if prediction_svm[0] == 1 else "Negative")
