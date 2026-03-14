import streamlit as st
import pandas as pd
import pickle

model = pickle.load(open("model.pkl","rb"))
scaler = pickle.load(open("scaler.pkl","rb"))
columns = pickle.load(open("columns.pkl","rb"))

st.title("Used Vehicle Price Prediction using SVM")

brand = st.text_input("Brand")
model_name = st.text_input("Model")
year = st.number_input("Model Year",2000,2025)
milage = st.number_input("Milage")
fuel = st.text_input("Fuel Type")
engine = st.text_input("Engine")
transmission = st.text_input("Transmission")
ext_col = st.text_input("Exterior Color")
int_col = st.text_input("Interior Color")
accident = st.text_input("Accident")
clean = st.text_input("Clean Title")

if st.button("Predict Price"):

    data = pd.DataFrame({
        'brand':[brand],
        'model':[model_name],
        'model_year':[year],
        'milage':[milage],
        'fuel_type':[fuel],
        'engine':[engine],
        'transmission':[transmission],
        'ext_col':[ext_col],
        'int_col':[int_col],
        'accident':[accident],
        'clean_title':[clean]
    })

    data = pd.get_dummies(data)
    data = data.reindex(columns=columns,fill_value=0)

    data_scaled = scaler.transform(data)

    prediction = model.predict(data_scaled)

    st.success(f"Predicted Price: ${prediction[0]:,.2f}")