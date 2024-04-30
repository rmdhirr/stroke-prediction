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

    age = st.number_input("Age", min_value=0, max_value=120, value=100)
    avg_glucose_level = st.number_input("Average Glucose Level (mg/dL)", min_value=50, max_value=600, value=100)
    bmi = st.number_input("Body Mass Index (kg/m²)", min_value=10, max_value=60, value=100)
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
    
    prob_stroke = prediction_proba[0][1] * 100
    if prob_stroke < 33:
        risk_level = "Low Risk"
        color = "green"
    elif 33 <= prob_stroke < 66:
        risk_level = "Medium Risk"
        color = "orange"
    else:
        risk_level = "High Risk"
        color = "red"

    st.markdown(f'<div style="border-radius:5px;padding:10px;color:white;background-color:{color};">{risk_level} - Probability of Stroke: {prob_stroke:.2f}%</div>', unsafe_allow_html=True)

# Custom CSS and JavaScript for Back to Top button
st.markdown("""
    <style>
    #myBtn {
        display: none;
        position: fixed;
        bottom: 20px;
        right: 30px;
        z-index: 99;
        border: none;
        outline: none;
        background-color: red;
        color: white;
        cursor: pointer;
        padding: 15px;
        border-radius: 10px;
        font-size: 18px;
    }

    #myBtn:hover {
        background-color: #555;
    }
    
    footer {visibility: hidden;}
    .reportview-container .main footer {visibility: hidden;}
    </style>
    <footer style="background-color:gray;color:white;text-align:center;padding:10px;">
        Developed by <a href="https://github.com/rmdhirr" style="color:white;">Ramadhirra</a>
    </footer>
    <button onclick="topFunction()" id="myBtn" title="Go to top">Top</button>
    <script>
    let mybutton = document.getElementById("myBtn");

    window.onscroll = function() {scrollFunction()};

    function scrollFunction() {
        if (document.body.scrollTop > 20 || document.documentElement.scrollTop > 20) {
            mybutton.style.display = "block";
        } else {
            mybutton.style.display = "none";
        }
    }

    function topFunction() {
        document.body.scrollTop = 0;
        document.documentElement.scrollTop = 0;
    }
    </script>
    """, unsafe_allow_html=True)
