import streamlit as st
import pandas as pd
import joblib

st.set_page_config(
    page_title="Prediksi Mahasiswa Dropout",
    page_icon="🎓",
    layout="centered"
)

st.title("🎓 Sistem Deteksi Dini Mahasiswa Dropout")
st.markdown("""
Aplikasi ini memprediksi risiko studi mahasiswa berdasarkan 7 indikator utama 
yang paling berpengaruh menurut analisis data Jaya Jaya Institut.
""")

# load model - pakai cache biar ga muat ulang tiap kali ada interaksi
@st.cache_resource
def load_model():
    return joblib.load('rf_model_binary.joblib')

model = load_model()

st.header("📝 Masukkan Data Mahasiswa")

col1, col2 = st.columns(2)

with col1:
    tuition_up_to_date = st.selectbox("Status Pembayaran Kuliah", ["Tunggakan (No)", "Lunas (Yes)"])
    scholarship = st.selectbox("Penerima Beasiswa", ["Tidak (No)", "Ya (Yes)"])
    debtor = st.selectbox("Status Debitur", ["Bukan Debitur (No)", "Debitur (Yes)"])
    # range usia ini mungkin bisa dipersempit, tapi biarkan dulu biar fleksibel
    age = st.number_input("Usia saat mendaftar", min_value=15, max_value=80, value=20)

with col2:
    grade_sem1 = st.slider("Nilai Rata-rata Semester 1", min_value=0.0, max_value=20.0, value=12.0, step=0.1)
    grade_sem2 = st.slider("Nilai Rata-rata Semester 2", min_value=0.0, max_value=20.0, value=12.0, step=0.1)
    gender = st.selectbox("Jenis Kelamin", ["Perempuan (Female)", "Laki-laki (Male)"])

if st.button("🔍 Deteksi Risiko", type="primary"):

    # konversi input teks ke nilai numerik sesuai encoding waktu training
    tuition_val = 1 if tuition_up_to_date == "Lunas (Yes)" else 0
    scholar_val = 1 if scholarship == "Ya (Yes)" else 0
    debtor_val  = 1 if debtor == "Debitur (Yes)" else 0
    gender_val  = 1 if gender == "Laki-laki (Male)" else 0

    # urutan kolom HARUS sama persis dengan waktu model dilatih, cek notebook bagian modeling
    df_input = pd.DataFrame([{
        'Tuition fees up to date':              tuition_val,
        'Scholarship holder':                   scholar_val,
        'Debtor':                               debtor_val,
        'Gender':                               gender_val,
        'Age at enrollment':                    age,
        'Curricular units 1st sem (grade)':     grade_sem1,
        'Curricular units 2nd sem (grade)':     grade_sem2
    }])

    prediction = model.predict(df_input)
    proba = model.predict_proba(df_input)

    st.markdown("---")

    if prediction[0] == 'Dropout':
        # index 0 = Dropout, index 1 = Graduate (sesuai urutan alphabet di sklearn)
        confidence = proba[0][0] * 100
        st.error("🚨 **Hasil Prediksi: DROPOUT**")
        st.write(f"**Tingkat Keyakinan Model:** {confidence:.2f}%")

        st.warning("⚠️ **Rekomendasi Strategis:**")
        st.write("- Segera jadwalkan sesi konseling akademik.")
        if debtor_val == 1:
            st.write("- **Intervensi Finansial:** Mahasiswa berstatus Debitur. Disarankan menawarkan skema cicilan khusus atau dana darurat.")
    else:
        confidence = proba[0][1] * 100
        st.success("✅ **Hasil Prediksi: GRADUATE (Lulus)**")
        st.write(f"**Tingkat Keyakinan Model:** {confidence:.2f}%")
        st.balloons()
