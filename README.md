# Klasifikasi-Suara-Hewan-Ayam-Kucing-Sapi-
Repository ini berisi implementasi sistem klasifikasi vokalisasi hewan berbasis Convolutional Neural Network (CNN). Penelitian ini mengevaluasi efektivitas ekstraksi fitur Mel-Spectrogram untuk mengenali pola suara pada tiga spesies: ayam, sapi, dan kucing. 


Aplikasi ini dikembangkan dalam bentuk Web App menggunakan framework Streamlit untuk memastikan aksesibilitas lintas perangkat tanpa kendala arsitektur instruksi CPU (AVX). 

Fitur Utama
Pra-pemrosesan Otomatis: Melakukan normalisasi amplitudo dan reduksi kebisingan (noise reduction) menggunakan library noisereduce.
Visualisasi Spektral: Mengubah sinyal audio satu dimensi menjadi representasi visual Mel-Spectrogram dua dimensi secara real-time. 
Analisis Spasial CNN: Memanfaatkan lapisan konvolusi untuk menangkap tekstur kecerahan yang mewakili intensitas amplitudo pada frekuensi tertentu. 
Antarmuka Interaktif: Pengguna dapat mengunggah file .wav, mendengarkan audio, dan mendapatkan hasil identifikasi beserta tingkat keyakinannya (confidence score). 


Teknologi yang Digunakan
Bahasa Pemrograman: Python 3.11 
Deep Learning Framework: TensorFlow & Keras 
Audio Processing: Librosa 
Web Framework: Streamlit

Struktur Proyek
app_web.py: Script utama aplikasi web Streamlit.
model_hewan_pintar.keras: Model CNN yang telah dilatih (Pre-trained model). 
gambar ayam/sapi/kucing.jpg: Aset visual untuk feedback identifikasi. 
requirements.txt: Daftar dependensi library Python.




Instal dependensi:
Bash
pip install -r requirements.txt

Jalankan aplikasi:
Bash
streamlit run app_web.py

Konteks Penelitian
Proyek ini merupakan bagian dari penelitian mahasiswa Teknik Informatika di Institut Sosial dan Teknologi Widuri, Jakarta (2025). Fokus utama penelitian adalah menganalisis bagaimana perubahan amplitudo terhadap waktu (envelope) menjadi fitur kunci yang membedakan jenis suara binatang secara objektif. 


