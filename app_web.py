import streamlit as st
import librosa
import librosa.display
import numpy as np
import tensorflow as tf
import noisereduce as nr
import matplotlib.pyplot as plt
from PIL import Image
import os

# --- KONFIGURASI HALAMAN ---
st.set_page_config(page_title="Animal Voice Recognition", page_icon="🐾")

st.title("🐾 Sistem Klasifikasi Suara Hewan (CNN)")
st.write("Aplikasi web berbasis Deep Learning untuk mengenali suara Ayam, Sapi, dan Kucing.")

# --- LOAD MODEL (Caching agar cepat) ---
@st.cache_resource
def load_my_model():
    model_path = 'model_hewan_pintar.keras'
    if os.path.exists(model_path):
        return tf.keras.models.load_model(model_path)
    return None

model = load_my_model()
label_list = ["Lain-lain", "Suara Ayam", "Suara Kucing", "Suara Sapi"]

# --- SIDEBAR ---
st.sidebar.header("Informasi Proyek")
st.sidebar.info("Proyek Jurnal Teknik Informatika - Institut Sosial dan Teknologi Widuri")
st.sidebar.write("Dibuat oleh: Tegar, Raffa, & Khaililul")

# --- FUNGSI PRA-PEMROSESAN ---
def siapkan_fitur(file_path):
    y, sr = librosa.load(file_path, duration=3)
    y = nr.reduce_noise(y=y, sr=sr, prop_decrease=0.8)
    y, _ = librosa.effects.trim(y)
    y = librosa.util.normalize(y)
    
    melspec = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
    log_spec = librosa.power_to_db(melspec, ref=np.max)
    
    # Simpan plot spektrogram ke objek figure
    fig, ax = plt.subplots(figsize=(10, 4))
    librosa.display.specshow(log_spec, sr=sr, x_axis='time', y_axis='mel', ax=ax, cmap='magma')
    ax.set_title("Mel-Spectrogram Terproses")
    
    # Reshape untuk input model (128x128)
    if log_spec.shape[1] > 128:
        log_spec = log_spec[:, :128]
    else:
        log_spec = np.pad(log_spec, ((0,0), (0, 128 - log_spec.shape[1])))
    
    return log_spec.reshape(1, 128, 128, 1), fig

# --- ANTARMUKA UTAMA ---
uploaded_file = st.file_uploader("Pilih file audio (format .wav)", type=["wav"])

if uploaded_file is not None:
    # 1. Putar Audio
    st.audio(uploaded_file, format='audio/wav')
    
    if model:
        with st.spinner('Sedang menganalisis suara...'):
            # 2. Proses Fitur dan Prediksi
            fitur, fig_spec = siapkan_fitur(uploaded_file)
            prediksi = model.predict(fitur)
            
            idx = np.argmax(prediksi)
            hasil = label_list[idx]
            skor = np.max(prediksi) * 100

            # 3. Tampilan Hasil
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Hasil Analisis")
                if skor < 75.0:
                    st.error("Hasil: Tidak Dikenali")
                else:
                    st.success(f"Teridentifikasi: {hasil}")
                    st.metric("Tingkat Keyakinan", f"{skor:.2f}%")
                    
                    # Tampilkan Gambar Berdasarkan Hasil
                    img_mapping = {
                        "Suara Ayam": "gambar ayam.jpg",
                        "Suara Kucing": "gambar kucing.jpg",
                        "Suara Sapi": "gambar sapi.jpg"
                    }
                    
                    img_path = img_mapping.get(hasil)
                    if img_path and os.path.exists(img_path):
                        st.image(Image.open(img_path), use_container_width=True)

            with col2:
                st.subheader("Visualisasi Spektral")
                st.pyplot(fig_spec)
    else:
        st.error("File model (.keras) tidak ditemukan!")