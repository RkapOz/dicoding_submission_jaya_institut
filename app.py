import streamlit as st
import pandas as pd
import joblib

# Set konfigurasi halaman Streamlit
st.set_page_config(
    page_title="Prediksi Mahasiswa Dropout",
    page_icon="🎓",
    layout="centered"
)

# Judul Aplikasi
st.title("🎓 Sistem Deteksi Dini Mahasiswa Dropout")
st.markdown("""
Aplikasi ini memprediksi apakah seorang mahasiswa berisiko **Dropout** atau akan **Graduate** (Lulus) 
berdasarkan performa akademik di awal semester dan status finansial.
""")

# 1. Memuat Model Machine Learning
@st.cache_resource
def load_model():
    # Pastikan file rf_model_binary.joblib berada di folder yang sama dengan app.py
    return joblib.load('rf_model_binary.joblib')

model = load_model()

# 2. Form Input Interaktif untuk User
st.header("📝 Masukkan Data Mahasiswa")

col1, col2 = st.columns(2)

with col1:
    tuition_up_to_date = st.selectbox("Status Pembayaran Kuliah", ["Tunggakan (No)", "Lunas (Yes)"])
    scholarship = st.selectbox("Penerima Beasiswa", ["Tidak (No)", "Ya (Yes)"])
    debtor = st.selectbox("Status Debitur", ["Bukan Debitur (No)", "Debitur (Yes)"])
    age = st.number_input("Usia saat mendaftar", min_value=15, max_value=80, value=20)

with col2:
    grade_sem1 = st.slider("Beban kredit yang diambil pada Semester 1", min_value=0.0, max_value=20.0, value=12.0, step=0.1)
    grade_sem2 = st.slider("Beban kredit yang diambil pada Semester 2", min_value=0.0, max_value=20.0, value=12.0, step=0.1)
    gender = st.selectbox("Jenis Kelamin", ["Perempuan (Female)", "Laki-laki (Male)"])

# 3. Proses Prediksi
if st.button("🔍 Deteksi Risiko", type="primary"):
    
    # Mapping input teks kembali ke angka (seperti saat model dilatih)
    tuition_val = 1 if tuition_up_to_date == "Lunas (Yes)" else 0
    scholar_val = 1 if scholarship == "Ya (Yes)" else 0
    debtor_val = 1 if debtor == "Debitur (Yes)" else 0
    gender_val = 1 if gender == "Laki-laki (Male)" else 0

    # Menyiapkan dictionary dengan 7 dari 36 fitur sesuai dengan Streamlit
    input_data = pd.DataFrame([{
    'Tuition fees up to date': tuition_val,
    'Scholarship holder': scholar_val,
    'Debtor': debtor_val,
    'Gender': gender_val,
    'Age at enrollment': age,
    'Curricular units 1st sem (grade)': grade_sem1,
    'Curricular units 2nd sem (grade)': grade_sem2
}])

# Prediksi sekarang akan 100% valid karena fitur model == fitur input
prediction = model.predict(input_data)
    # Mengubah dictionary menjadi DataFrame dengan 1 baris
    df_input = pd.DataFrame([input_data])

    # Melakukan prediksi
    prediction = model.predict(df_input)

    # Menampilkan hasil dengan visual yang menarik
    st.markdown("---")
    if prediction[0] == 'Dropout':
        st.error("🚨 **Peringatan Tinggi:** Mahasiswa ini memiliki risiko besar untuk **DROPOUT**.")
        st.info("💡 **Rekomendasi:** Segera jadwalkan sesi konseling akademik dan tinjau opsi bantuan finansial jika mahasiswa mengalami kesulitan pembayaran.")
    else:
        st.success("✅ **Status Aman:** Mahasiswa ini diprediksi akan **GRADUATE** (Lulus).")
        st.balloons()