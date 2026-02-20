import streamlit as st
import pandas as pd
import joblib
import os

# --- KONFIGURASI HALAMAN ---
st.set_page_config(
    page_title="SCM Risk Analytics Dashboard",
    page_icon="🚚",
    layout="wide"
)

# --- FUNGSI LOAD DATA & MODEL ---
def load_resources():
    curr_path = os.path.dirname(__file__)
    model = joblib.load(os.path.join(curr_path, 'model_scm_final.pkl'))
    # Kita definisikan list fitur secara manual agar aman dari FileNotFoundError
    features = [
        'Days for shipment (scheduled)', 
        'Shipping Mode', 
        'Order Region', 
        'Sales per customer', 
        'Order Item Quantity',
        'Market'
    ]
    return model, features

try:
    model, features_list = load_resources()
except Exception as e:
    st.error(f"Gagal memuat resource aplikasi. Pastikan model_scm_final.pkl sudah di-upload. Error: {e}")
    st.stop()

# --- SIDEBAR: INPUT PARAMETER ---
st.sidebar.header("🛠️ Konfigurasi Pesanan")
st.sidebar.write("Masukkan detail pengiriman untuk dianalisis oleh AI:")

with st.sidebar:
    scheduled_days = st.slider("Target Pengiriman (Hari Terjadwal)", 0, 6, 3)
    
    # Mapping Shipping Mode (Sesuai Encoding di Riset)
    ship_mode_opt = ["First Class", "Same Day", "Second Class", "Standard Class"]
    ship_mode_choice = st.selectbox("Metode Pengiriman", ship_mode_opt)
    mode_mapping = {"First Class": 0, "Same Day": 1, "Second Class": 2, "Standard Class": 3}
    
    # Mapping Market (Sesuai Encoding di Riset)
    market_opt = ["Africa", "Europe", "LATAM", "Pacific Asia", "USCA"]
    market_choice = st.selectbox("Pasar Utama (Market)", market_opt)
    market_mapping = {"Africa": 0, "Europe": 1, "LATAM": 2, "Pacific Asia": 3, "USCA": 4}

    # Mapping Region (Paling sering muncul di DataCo)
    region_dict = {
        "Southeast Asia": 11, "South Asia": 13, "Western Europe": 22, 
        "Central America": 3, "Oceania": 12, "Eastern Asia": 5, "North Africa": 10
    }
    region_choice = st.selectbox("Wilayah Tujuan (Order Region)", list(region_dict.keys()))
    
    sales_cust = st.number_input("Nilai Penjualan ($)", value=150.0)
    quantity = st.number_input("Jumlah Barang", min_value=1, value=1)

# --- HALAMAN UTAMA ---
st.title("🚚 SCM Late Delivery Risk Predictor")
st.markdown("""
Aplikasi ini merupakan implementasi dari riset klasifikasi risiko rantai pasok menggunakan algoritma **XGBoost**.
Sistem ini memprediksi apakah sebuah pesanan akan mengalami keterlambatan berdasarkan pola historis.
""")

col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("🔮 Prediksi & Analisis")
    if st.button("Analisis Risiko Pengiriman"):
        # Menyusun data input sesuai fitur model
        input_data = pd.DataFrame([[
            scheduled_days, 
            mode_mapping[ship_mode_choice], 
            region_dict[region_choice], 
            sales_cust, 
            quantity, 
            market_mapping[market_choice]
        ]], columns=features_list)
        
        # Eksekusi Prediksi
        prediction = model.predict(input_data)
        probability = model.predict_proba(input_data)[0][1] * 100

        # Menampilkan Hasil
        if prediction[0] == 1:
            st.error(f"### ⚠️ TERDETEKSI RISIKO TINGGI ({probability:.2f}%)")
            st.markdown(f"**Status:** Pengiriman ini diprediksi akan **TERLAMBAT**.")
        else:
            st.success(f"### ✅ PREDIKSI AMAN ({100 - probability:.2f}%)")
            st.markdown(f"**Status:** Pengiriman ini diprediksi akan **TEPAT WAKTU**.")

        # Penjelasan Teknis di Web
        st.write("---")
        st.markdown("#### **Mengapa hasil ini muncul?**")
        st.write(f"""
        Model menganalisis bahwa metode **{ship_mode_choice}** dengan tujuan wilayah **{region_choice}** memiliki korelasi yang kuat terhadap risiko tersebut berdasarkan data masa lalu.
        """)

with col2:
    st.subheader("📊 Metodologi Riset")
    st.info("""
    **Fitur Utama Riset:**
    1. **Scheduled Days**: Variabel paling krusial.
    2. **Shipping Mode**: Menentukan prioritas logistik.
    3. **Region**: Menentukan tantangan infrastruktur.
    """)
    
    st.write("**Skala Risiko:**")
    st.write("- 🟢 < 30%: Aman")
    st.write("- 🟡 30-60%: Waspada")
    st.write("- 🔴 > 60%: Risiko Tinggi")

# --- VISUALISASI TAMBAHAN ---
st.divider()
st.subheader("📈 Visualisasi Faktor Pengaruh (Feature Importance)")
st.write("Grafik di bawah ini menjelaskan variabel mana yang paling menentukan dalam model prediksi ini.")



st.markdown("""
> **Insight Riset:** Berdasarkan grafik di atas, 'Scheduled Shipping Days' adalah penentu utama. 
> Jika perusahaan menjanjikan waktu yang terlalu singkat tanpa didukung moda transportasi yang tepat, 
> maka probabilitas keterlambatan akan meningkat secara drastis.
""")

st.caption("Copyright © 2024 - Riset Supply Chain Management dengan Machine Learning")
