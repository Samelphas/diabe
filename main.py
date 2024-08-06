import streamlit as st
import pickle
import numpy as np
with open('diabetes.pkl', 'rb') as file:
    model = pickle.load(file)
   
y_pred = model.predict([[1,2,3,4,5,6,7,8]])
print(y_pred)
# Function to make predictions
def predict_diabetes(inputs):
    return model.predict(inputs)

# Streamlit app
st.title("Diabetes Prediction App")

pregnancies = st.number_input("Pregnancies", 0, 20, step=1)
glucose = st.number_input("Glucose", 0, 200, step=1)
blood_pressure = st.number_input("Blood Pressure", 0, 122, step=1)
skin_thickness = st.number_input("Skin Thickness", 0, 99, step=1)
insulin = st.number_input("Insulin", 0, 846, step=1)
bmi = st.number_input("BMI", 0.0, 67.1, step=0.1)
dpf = st.number_input("Diabetes Pedigree Function", 0.0, 2.5, step=0.01)
age = st.number_input("Age", 21, 100, step=1)
# Make prediction
if st.button("Predict"):
    inputs = np.array([[pregnancies, glucose, blood_pressure, skin_thickness, insulin, bmi, dpf, age]])
    prediction = predict_diabetes(inputs)
    if prediction[0] == 1:
        st.write("The model predicts that the person has diabetes.")
    else:
        st.write("The model predicts that the person does not have diabetes.")