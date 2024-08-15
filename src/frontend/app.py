# Import Libraries
import json
import requests
import streamlit as st

# Initialization
API_URL = 'http://localhost:8000/score'

with open('config/config.json', 'r') as file:
    input_options = json.load(file)

# Main UI
st.title("Credit Risk - Probability Being Bad Loan")

# Service Inputs
loanAmount = st.number_input("Loan Amount ($)", min_value=0)
state = st.selectbox("State", input_options["state"])
leadType = st.selectbox("Lead Type", input_options["leadType"])
leadCost = st.number_input("Lead Cost ($)", min_value=0)

# Predict Button
if st.button("Predict Probability of Being Bad Loan"):
    data = {"loanAmount": loanAmount,
            "state": state,
            "leadType": leadType,
            "leadCost": leadCost}
    response = requests.post(API_URL, json={"data": data})
    prediction = response.json()["score"]
    st.write(f"Predicted Score: {prediction}")