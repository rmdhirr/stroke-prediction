import streamlit as st
import pandas as pd
import pickle

# Set page config
st.set_page_config(page_title="Stroke Prediction", page_icon="🧠")

# Load the model
@st.cache(allow_output_mutation=True)
def load_model():
    with open('random_forest_model_nt.pkl', 'rb') as file:
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

    # Map work types and smoking status to the correct numerical encoding as used during model training
    work_type_mapping = {
        "Private": 0, "Self-employed": 1, "Govt_job": 2, "Children": 3, "Never_worked": 4
    }
    smoking_status_mapping = {
        "formerly smoked": 0, "never smoked": 1, "smokes": 2, "unknown": 3
    }

    # Create a data frame of the input features with correct naming convention as used in training
    features = pd.DataFrame({
        'age': [age],
        'avg_glucose_level': [avg_glucose_level],
        'bmi': [bmi],
        'gender_0': [1 if gender == 'Male' else 0],
        'gender_1': [0 if gender == 'Male' else 1],
        'hypertension_0': [1 if hypertension == 'No' else 0],
        'hypertension_1': [0 if hypertension == 'No' else 1],
        'heart_disease_0': [1 if heart_disease == 'No' else 0],
        'heart_disease_1': [0 if heart_disease == 'No' else 1],
        'ever_married_0': [1 if ever_married == 'No' else 0],
        'ever_married_1': [0 if ever_married == 'No' else 1],
        'work_type_0': [1 if work_type == work_type_mapping['Private'] else 0],
        'work_type_1': [1 if work_type == work_type_mapping['Self-employed'] else 0],
        'work_type_2': [1 if work_type == work_type_mapping['Govt_job'] else 0],
        'work_type_3': [1 if work_type == work_type_mapping['Children'] else 0],
        'work_type_4': [1 if work_type == work_type_mapping['Never_worked'] else 0],
        'Residence_type_0': [1 if residence_type == 'Urban' else 0],
        'Residence_type_1': [0 if residence_type == 'Urban' else 1],
        'smoking_status_0': [1 if smoking_status == smoking_status_mapping['formerly smoked'] else 0],
        'smoking_status_1': [1 if smoking_status == smoking_status_mapping['never smoked'] else 0],
        'smoking_status_2': [1 if smoking_status == smoking_status_mapping['smokes'] else 0],
        'smoking_status_3': [1 if smoking_status == smoking_status_mapping['unknown'] else 0],
    })
    return features

input_df = user_input_features()

if st.button('Predict'):
    prediction = model.predict(input_df)
    prediction_proba = model.predict_proba(input_df)
    
    # Convert probabilities to percentage and round to two decimal places
    prob_stroke = prediction_proba[0][1] * 100
    
    # Define risk levels based on the probability of stroke
    if prob_stroke < 33:
        risk_level = "Low Risk"
        color = "success"  # Green
    elif 33 <= prob_stroke < 66:
        risk_level = "Medium Risk"
        color = "warning"  # Orange
    else:
        risk_level = "High Risk"
        color = "danger"  # Red
    
    # Display predictions and probabilities with color-coded risk level
    st.write(f'Prediction (0: No Stroke, 1: Stroke): {prediction[0]}')
    st.markdown(f'**Probability of Stroke: {prob_stroke:.2f}%**')
    st.markdown(f'<h2 style="color: {color};">{risk_level}</h2>', unsafe_allow_html=True)

# Footer with credits and GitHub link
st.markdown("## Credits")
st.markdown("Developed by [Ramadhirra](https://github.com/rmdhirr)")

# Custom CSS to style the back to top button
st.markdown("""
    <style>
    .reportview-container .main footer {visibility: hidden;}
    #MainMenu {visibility: hidden;}
    footer {position: fixed; left: 0; bottom: 0; width: 100%; background-color: white; color: black; text-align: center;}
    </style>
    <footer>
        <button onclick="window.scrollTo(0,0);">Back to Top</button>
    </footer>
    """, unsafe_allow_html=True)
