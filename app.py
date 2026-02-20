import streamlit as st
import pandas as pd
import joblib
import numpy as np

# Load model dan daftar fitur
model = joblib.load('model_scm_final.pkl')
features_list = joblib.load('features.pkl')

st.title("🚚 SCM Delivery Risk Predictor")

# Input Form
st.sidebar.header("Input Parameter")

# Buat input sesuai dengan selected_features di Colab
days_scheduled = st.sidebar.slider("Days for Shipment (Scheduled)", 0, 6, 3)
shipping_mode = st.sidebar.selectbox("Shipping Mode", options=[0, 1, 2, 3], 
                                     help="0: First Class, 1: Same Day, 2: Second Class, 3: Standard Class")
order_region = st.sidebar.number_input("Order Region ID", value=0)
sales = st.sidebar.number_input("Sales per Customer", value=150.0)
quantity = st.sidebar.number_input("Order Item Quantity", value=1)
market = st.sidebar.selectbox("Market", options=[0, 1, 2, 3, 4], 
                               help="ID Market (sesuaikan dengan label encoder)")

if st.button("Cek Risiko Keterlambatan"):
    # MENYAMAKAN FORMAT DATA (PENTING!)
    # Data harus dalam bentuk DataFrame dengan nama kolom yang persis sama
    data_input = pd.DataFrame([[days_scheduled, shipping_mode, order_region, sales, quantity, market]], 
                              columns=features_list)
    
    # Prediksi
    prediction = model.predict(data_input)
    probability = model.predict_proba(data_input)

    if prediction[0] == 1:
        st.error(f"⚠️ RISIKO TERLAMBAT: {probability[0][1]*100:.2f}%")
    else:
        st.success(f"✅ TEPAT WAKTU: {probability[0][0]*100:.2f}%")
