import streamlit as st
import pandas as pd
import pickle

# Load the model
@st.cache(allow_output_mutation=True)
def load_model():
    # Load a model from the same directory as the script
    with open('random_forest_model_ht.pkl', 'rb') as file:
        model = pickle.load(file)
    return model

model = load_model()

def user_input_features():
    st.header('Stroke Prediction Model')
    st.write("Please enter the following details to predict the stroke risk:")

    age = st.number_input("Age", min_value=0, max_value=120, value=30)
    avg_glucose_level = st.number_input("Average Glucose Level (mg/dL)", min_value=50, max_value=300, value=100)
    bmi = st.number_input("Body Mass Index (kg/m²)", min_value=10, max_value=50, value=22)
    gender = st.selectbox("Gender", ("Male", "Female"))
    hypertension = st.selectbox("Hypertension", ("No", "Yes"), index=0)
    heart_disease = st.selectbox("Heart Disease", ("No", "Yes"), index=0)
    ever_married = st.selectbox("Ever Married?", ("No", "Yes"), index=1)
    work_type = st.selectbox("Work Type", ("Private", "Self-employed", "Govt_job", "Children", "Never_worked"))
    residence_type = st.selectbox("Residence Type", ("Urban", "Rural"))
    smoking_status = st.selectbox("Smoking Status", ("formerly smoked", "never smoked", "smokes", "unknown"))

    # Create a data frame of the input features
    features = pd.DataFrame({
        'age': [age],
        'avg_glucose_level': [avg_glucose_level],
        'bmi': [bmi],
        'gender_Male': [1 if gender == 'Male' else 0],
        'gender_Female': [0 if gender == 'Male' else 1],
        'hypertension_No': [1 if hypertension == 'No' else 0],
        'hypertension_Yes': [0 if hypertension == 'No' else 1],
        'heart_disease_No': [1 if heart_disease == 'No' else 0],
        'heart_disease_Yes': [0 if heart_disease == 'No' else 1],
        'ever_married_No': [1 if ever_married == 'No' else 0],
        'ever_married_Yes': [0 if ever_married == 'No' else 1],
        'work_type_Private': [1 if work_type == 'Private' else 0],
        'work_type_Self-employed': [1 if work_type == 'Self-employed' else 0],
        'work_type_Govt_job': [1 if work_type == 'Govt_job' else 0],
        'work_type_Children': [1 if work_type == 'Children' else 0],
        'work_type_Never_worked': [1 if work_type == 'Never_worked' else 0],
        'residence_type_Urban': [1 if residence_type == 'Urban' else 0],
        'residence_type_Rural': [0 if residence_type == 'Urban' else 1],
        'smoking_status_formerly_smoked': [1 if smoking_status == 'formerly smoked' else 0],
        'smoking_status_never_smoked': [1 if smoking_status == 'never smoked' else 0],
        'smoking_status_smokes': [1 if smoking_status == 'smokes' else 0],
        'smoking_status_unknown': [1 if smoking_status == 'unknown' else 0],
    })
    return features

input_df = user_input_features()

if st.button('Predict'):
    prediction = model.predict(input_df)
    prediction_proba = model.predict_proba(input_df)
    st.write(f'Prediction (0: No Stroke, 1: Stroke): {prediction[0]}')
    st.write(f'Probability [No Stroke, Stroke]: {prediction_proba[0][1]*100:.2f}%')