import streamlit as st
import pandas as pd
import pickle

# Set page config
st.set_page_config(page_title="Stroke Prediction", page_icon="🧠", layout="wide")

# Load the model
@st.cache(allow_output_mutation=True)
def load_model():
    with open('random_forest_model_nt.pkl', 'rb') as file:
        model = pickle.load(file)
    return model

model = load_model()

def user_input_features():
    st.header('Stroke Prediction')
    st.write("Please enter the following details to predict the stroke risk:")

    # Collect inputs
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
        # ... (your DataFrame setup)
    })
    return features

input_df = user_input_features()

if st.button('Predict'):
    prediction = model.predict(input_df)
    prediction_proba = model.predict_proba(input_df)[0]

    # Define risk levels based on probability of having a stroke
    risk_level = prediction_proba[1]  # Probability of stroke
    if risk_level < 0.2:
        risk_text = "Low Risk"
        risk_color = "success"
    elif 0.2 <= risk_level < 0.5:
        risk_text = "Medium Risk"
        risk_color = "warning"
    elif 0.5 <= risk_level < 0.7:
        risk_text = "Medium High Risk"
        risk_color = "orange"
    else:
        risk_text = "High Risk"
        risk_color = "danger"
    
    st.markdown(f"### Prediction: {risk_text}")
    st.markdown(f"#### Detailed Probability of Stroke: {risk_level*100:.2f}%")
    st.markdown(f'<div style="color: {risk_color}; font-size: 24px;">{risk_text} - {risk_level*100:.2f}% chance of stroke</div>', unsafe_allow_html=True)

# Credits and GitHub link
st.sidebar.markdown("## Credits")
st.sidebar.markdown("Developed by Ramadhirra")
st.sidebar.markdown("[GitHub](https://github.com/rmdhirr)")

# Back to top button
st.sidebar.markdown('<a href="#top">Back to Top</a>', unsafe_allow_html=True)
