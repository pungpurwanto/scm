import streamlit as st
import pandas as pd
import joblib

# Load model
model = joblib.load('model_scm.pkl')

st.title("🚚 SCM Late Delivery Predictor")
st.write("Aplikasi ini memprediksi risiko keterlambatan pengiriman berdasarkan data operasional.")

# Input Form
st.sidebar.header("Input Data Pengiriman")
days_scheduled = st.sidebar.slider("Days Scheduled", 0, 7, 3)
shipping_mode = st.sidebar.selectbox("Shipping Mode", [0, 1, 2, 3]) # Sesuaikan dengan encoding Anda
order_region = st.sidebar.number_input("Order Region ID", value=0)
sales = st.sidebar.number_input("Sales Amount", value=100.0)

# Prediksi
if st.button("Prediksi Risiko"):
    # Buat dataframe sesuai urutan kolom saat training
    # Catatan: Masukkan semua kolom yang digunakan saat X_train
    input_data = pd.DataFrame([[days_scheduled, shipping_mode, order_region, sales]], 
                              columns=['Days for shipment (scheduled)', 'Shipping Mode', 'Order Region', 'Sales'])
    
    # Karena model mengharapkan fitur lengkap, pastikan input_data memiliki kolom yang sama dengan X_train
    # Untuk demo sederhana, kita asumsikan 4 fitur utama ini.
    
    prediction = model.predict(input_data)
    prob = model.predict_proba(input_data)

    if prediction[0] == 1:
        st.error(f"⚠️ RISIKO TERLAMBAT: {prob[0][1]*100:.2f}%")
    else:
        st.success(f"✅ TEPAT WAKTU: {prob[0][0]*100:.2f}%")