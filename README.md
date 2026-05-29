# Proyek Akhir: Menyelesaikan Permasalahan Institusi Pendidikan

Nama: Aqil Aziz

## Business Understanding

Jaya Jaya Institut menghadapi tingkat dropout yang perlu ditekan. Institusi membutuhkan cara untuk memahami faktor yang berkaitan dengan dropout dan mendeteksi mahasiswa berisiko lebih awal agar tim akademik dapat melakukan intervensi.

## Permasalahan Bisnis

- Banyak mahasiswa tidak menyelesaikan pendidikan.
- Tim akademik membutuhkan indikator risiko yang mudah dipantau.
- Institusi membutuhkan prototype prediksi dropout untuk membantu prioritas bimbingan.

## Cakupan Proyek

- Melakukan eksplorasi dataset students performance.
- Membuat dashboard monitoring faktor dropout.
- Membuat model machine learning klasifikasi risiko dropout.
- Membuat prototype Streamlit untuk prediksi risiko mahasiswa.
- Memberikan rekomendasi action items.

## Dataset

Dataset yang digunakan adalah student performance dataset yang disediakan pada kelas Dicoding dan berasal dari UCI Machine Learning Repository.

Sumber data:

- File lokal: `data/students_performance.csv`
- Referensi dataset: https://archive.ics.uci.edu/dataset/697/predict+students+dropout+and+academic+success

Target model dibuat biner:

- `1`: Dropout
- `0`: Graduate

Catatan penting: data dengan status `Enrolled` tidak digunakan untuk training model karena status akhirnya belum diketahui. Data `Enrolled` hanya digunakan sebagai data inferensi pada prototype untuk melihat potensi risiko dropout.

## Persiapan Proyek

### 1. Membuat dan Mengaktifkan Virtual Environment

Windows:

```bash
python -m venv .venv
.venv\Scripts\activate
```

Linux/macOS:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 2. Menginstal Dependensi

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

### 3. Menjalankan Notebook

Buka `notebook.ipynb` menggunakan Jupyter Notebook, JupyterLab, Google Colab, atau VS Code, lalu jalankan seluruh cell dari atas ke bawah.

### 4. Menjalankan Prototype Python

File utama prototype adalah `app.py`. Jalankan perintah berikut dari root folder proyek:

```bash
streamlit run app.py
```

Prototype lokal akan terbuka di `http://localhost:8501`.

## Dashboard

Dashboard disediakan dalam bentuk file HTML, screenshot, dan data ringkasan:

- `dashboard/dashboard.html`
- `dashboard/dashboard.png`
- `aqilaziz_dicoding-dashboard.png`
- `dashboard/status_summary.csv`
- `dashboard/dropout_factor_summary.csv`
- `dashboard/top_dropout_risk.csv`

Dashboard memuat distribusi status mahasiswa, dropout rate berdasarkan pembayaran tuition, performa akademik semester 2, dan sebaran usia saat pendaftaran.

## Prototype Machine Learning

Model disimpan pada:

- `model/dropout_prediction_model.joblib`

Metrik validasi model:

- Accuracy: 0.920
- Precision: 0.912
- Recall: 0.880
- F1-score: 0.896
- Data training: 3630 baris dengan status Dropout dan Graduate
- Data Enrolled yang dikeluarkan dari training: 794 baris

### Cara Menjalankan Prototype Lokal

```bash
pip install -r requirements.txt
streamlit run app.py
```

Prototype akan terbuka di `http://localhost:8501`.

### Streamlit Community Cloud

Prototype dibuat menggunakan Streamlit dengan entrypoint `app.py`.

Link prototype Streamlit Community Cloud:

- https://jaya-jaya-institut-dropout-aqilaziz.streamlit.app

Jika reviewer menjalankan ulang secara lokal, gunakan perintah `streamlit run app.py`.

## Conclusion

Kesimpulan EDA dan dashboard:

Mahasiswa dropout cenderung memiliki indikator akademik yang lebih rendah, terutama pada jumlah unit kurikulum yang disetujui dan nilai semester awal. Status pembayaran tuition, status debtor, scholarship holder, dan usia saat pendaftaran juga terlihat berkaitan dengan risiko dropout. Dashboard menunjukkan bahwa pemantauan status pembayaran dan performa akademik semester awal dapat membantu institusi menemukan kelompok mahasiswa yang perlu ditindaklanjuti.

Kesimpulan model machine learning:

Model Random Forest dilatih sebagai klasifikasi biner menggunakan data `Dropout` dan `Graduate` saja. Model memperoleh accuracy 0.920, precision 0.912, recall 0.880, dan F1-score 0.896. Fitur paling berpengaruh pada model adalah:

- `Curricular_units_2nd_sem_approved`: 0.2430
- `Curricular_units_1st_sem_approved`: 0.1400
- `Curricular_units_2nd_sem_grade`: 0.1244
- `Curricular_units_1st_sem_grade`: 0.0736
- `Tuition_fees_up_to_date`: 0.0564

Model ini dapat digunakan sebagai alat bantu prioritas intervensi, bukan sebagai satu-satunya dasar keputusan akademik.

## Action Items

1. Prioritaskan bimbingan untuk mahasiswa dengan probabilitas dropout tinggi.
2. Pantau mahasiswa yang memiliki tunggakan tuition atau status debtor.
3. Buat program remedial untuk mahasiswa dengan jumlah unit kurikulum yang disetujui rendah.
4. Kombinasikan dashboard monitoring dengan review akademik bulanan.
5. Evaluasi ulang model setiap periode akademik agar tetap sesuai dengan data terbaru.
