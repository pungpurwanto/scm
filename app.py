import streamlit as st
import pandas as pd
import joblib
import os

# --- KONFIGURASI HALAMAN ---
st.set_page_config(
    page_title="SCM Analytics Dashboard",
    page_icon="🚚",
    layout="wide"
)

# --- LOAD MODEL & FITUR ---
# Menggunakan penanganan error agar aplikasi tidak crash jika file hilang
curr_path = os.path.dirname(__file__)
model_path = os.path.join(curr_path, 'model_scm_final.pkl')

@st.cache_resource # Cache agar model tidak reload terus menerus
def load_model():
    return joblib.load(model_path)

try:
    model = load_model()
    # Daftar fitur harus persis sama dengan saat training di Colab
    features_list = [
        'Days for shipment (scheduled)', 
        'Shipping Mode', 
        'Order Region', 
        'Sales per customer', 
        'Order Item Quantity',
        'Market'
    ]
except Exception as e:
    st.error(f"Gagal memuat model. Pastikan file 'model_scm_final.pkl' ada di GitHub. Error: {e}")
    st.stop()

# --- SIDEBAR INPUT ---
st.sidebar.header("🛠️ Parameter Input Pesanan")
st.sidebar.markdown("Masukkan data operasional di bawah ini:")

with st.sidebar:
    scheduled_days = st.slider("Target Pengiriman (Hari)", 0, 6, 3, 
                               help="Batas waktu yang dijanjikan sistem kepada pelanggan.")
    
    ship_mode_label = st.selectbox("Metode Pengiriman", 
                                  ["Standard Class", "Second Class", "First Class", "Same Day"])
    # Mapping sesuai encoding: First Class:0, Same Day:1, Second Class:2, Standard Class:3
    mode_mapping = {"First Class": 0, "Same Day": 1, "Second Class": 2, "Standard Class": 3}
    
    region = st.number_input("ID Wilayah (Region ID)", value=0, help="ID Numerik lokasi tujuan pengiriman.")
    sales = st.number_input("Nilai Penjualan ($)", value=150.0)
    qty = st.number_input("Jumlah Barang (Quantity)", value=1, min_value=1)
    
    market_label = st.selectbox("Pasar (Market)", 
                               ["Africa", "Europe", "LATAM", "Pacific Asia", "USCA"])
    market_mapping = {"Africa":0, "Europe":1, "LATAM":2, "Pacific Asia":3, "USCA":4}

# --- HALAMAN UTAMA ---
st.title("🚚 SCM Late Delivery Risk Predictor")
st.markdown("""
Aplikasi ini menggunakan model **XGBoost Classifier** untuk memprediksi probabilitas keterlambatan pengiriman berdasarkan pola data historis *DataCo Smart Supply Chain*.
""")

# Layout Kolom Utama
col_main, col_info = st.columns([2, 1])

with col_main:
    st.subheader("🔍 Hasil Analisis")
    if st.button("Jalankan Prediksi Risiko"):
        # Menyiapkan data untuk model
        input_data = pd.DataFrame([[
            scheduled_days, 
            mode_mapping[ship_mode_label], 
            region, 
            sales, 
            qty, 
            market_mapping[market_label]
        ]], columns=features_list)
        
        # Prediksi
        prediction = model.predict(input_data)
        probability = model.predict_proba(input_data)[0][1] * 100

        # Menampilkan Hasil dengan Desain
        if prediction[0] == 1:
            st.error(f"### ⚠️ RISIKO TERLAMBAT: {probability:.2f}%")
            st.markdown(f"""
            **Analisis:** Pesanan ini memiliki karakteristik yang sering mengalami hambatan logistik. 
            Sistem mendeteksi peluang keterlambatan sebesar **{probability:.2f}%**.
            """)
        else:
            st.success(f"### ✅ PREDIKSI AMAN: {100-probability:.2f}% Tepat Waktu")
            st.markdown(f"""
            **Analisis:** Pesanan ini diprediksi akan tiba sesuai jadwal dengan tingkat keyakinan **{100-probability:.2f}%**.
            """)
        
        # Penjelasan Teknis
        with st.expander("📚 Penjelasan Hasil (Prescriptive Insights)"):
            st.write(f"""
            1. **Faktor Utama:** Model mendeteksi bahwa kombinasi metode pengiriman '{ship_mode_label}' dan target '{scheduled_days} hari' memiliki dampak besar pada hasil ini.
            2. **Rekomendasi:** * Jika berisiko tinggi: Lakukan percepatan (expedite) di gudang atau informasikan potensi delay kepada pelanggan.
                * Jika aman: Jalankan prosedur logistik standar.
            """)
    else:
        st.info("Silakan masukkan data pada sidebar dan klik tombol prediksi.")

with col_info:
    st.subheader("ℹ️ Panduan Interpretasi")
    st.info("""
    **Cara Membaca:**
    - **Aman (< 30%):** Probabilitas telat rendah.
    - **Waspada (30% - 60%):** Perlu pengawasan.
    - **Risiko Tinggi (> 60%):** Sangat mungkin terlambat.
    """)
    
    st.write("**Fitur Paling Berpengaruh (Berdasarkan Riset):**")
    st.markdown("""
    1. **Scheduled Days** (Estimasi Sistem)
    2. **Shipping Mode** (Kelas Kurir)
    3. **Order Region** (Tantangan Geografis)
    """)

# --- FOOTER / GRAFIK PENJELAS ---
st.divider()
st.subheader("📈 Logika Prediksi (Feature Importance)")
st.write("Grafik di bawah ini menjelaskan variabel mana yang paling memengaruhi keputusan AI dalam riset ini.")



st.caption("Analisis ini didasarkan pada model yang dilatih menggunakan dataset DataCo Smart Supply Chain untuk publikasi jurnal ilmiah.")
