import json
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st


st.set_page_config(page_title="Jaya Jaya Institut - Dropout Risk", layout="wide")

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "model" / "dropout_prediction_model.joblib"
METADATA_PATH = BASE_DIR / "model" / "model_metadata.json"
DATA_PATH = BASE_DIR / "data" / "students_performance.csv"


@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


@st.cache_data
def load_metadata():
    with open(METADATA_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


@st.cache_data
def load_sample_data():
    return pd.read_csv(DATA_PATH, sep=";")


model = load_model()
metadata = load_metadata()
df = load_sample_data()
features = metadata["features"]

st.title("Prototype Prediksi Risiko Dropout")
st.caption("Jaya Jaya Institut")
st.info("Model dilatih hanya menggunakan data mahasiswa berstatus Dropout dan Graduate. Data Enrolled digunakan sebagai contoh inferensi risiko.")

left, right = st.columns([1, 1])
with left:
    status_filter = st.selectbox("Filter data contoh", ["Semua", "Enrolled", "Dropout", "Graduate"], index=1)
    sample_df = df if status_filter == "Semua" else df[df["Status"] == status_filter]
    if sample_df.empty:
        sample_df = df
    selected_index = st.selectbox("Pilih indeks data contoh", sample_df.index.tolist())
    row = df.loc[int(selected_index), features].copy()
    original_status = df.loc[int(selected_index), "Status"]

    st.subheader("Profil Mahasiswa")
    st.caption(f"Status data contoh: {original_status}")
    age = st.number_input("Age at enrollment", 15, 80, int(row["Age_at_enrollment"]))
    tuition = st.selectbox("Tuition fees up to date", [0, 1], index=int(row["Tuition_fees_up_to_date"]))
    debtor = st.selectbox("Debtor", [0, 1], index=int(row["Debtor"]))
    scholarship = st.selectbox("Scholarship holder", [0, 1], index=int(row["Scholarship_holder"]))
    sem1_approved = st.number_input(
        "Curricular units 1st sem approved",
        0,
        30,
        int(row["Curricular_units_1st_sem_approved"]),
    )
    sem2_approved = st.number_input(
        "Curricular units 2nd sem approved",
        0,
        30,
        int(row["Curricular_units_2nd_sem_approved"]),
    )
    sem1_grade = st.number_input("Curricular units 1st sem grade", 0.0, 20.0, float(row["Curricular_units_1st_sem_grade"]))
    sem2_grade = st.number_input("Curricular units 2nd sem grade", 0.0, 20.0, float(row["Curricular_units_2nd_sem_grade"]))

row["Age_at_enrollment"] = age
row["Tuition_fees_up_to_date"] = tuition
row["Debtor"] = debtor
row["Scholarship_holder"] = scholarship
row["Curricular_units_1st_sem_approved"] = sem1_approved
row["Curricular_units_2nd_sem_approved"] = sem2_approved
row["Curricular_units_1st_sem_grade"] = sem1_grade
row["Curricular_units_2nd_sem_grade"] = sem2_grade

input_df = pd.DataFrame([row], columns=features)
prob = model.predict_proba(input_df)[0, 1]
prediction = "Berisiko Dropout" if prob >= metadata.get("threshold", 0.5) else "Tidak Berisiko Tinggi"

with right:
    st.subheader("Hasil Prediksi")
    st.metric("Probabilitas Dropout", f"{prob:.1%}")
    st.metric("Klasifikasi", prediction)
    st.write("Metrik validasi model")
    st.json({
        "accuracy": round(metadata["accuracy"], 3),
        "precision": round(metadata["precision"], 3),
        "recall": round(metadata["recall"], 3),
        "f1": round(metadata["f1"], 3),
        "training_rows": metadata["training_rows"],
        "excluded_enrolled_rows": metadata["excluded_from_training"]["Enrolled"],
    })

st.divider()
col_a, col_b = st.columns([1, 1])
with col_a:
    st.subheader("Data Input Model")
    st.dataframe(input_df, width="stretch")
with col_b:
    st.subheader("Fitur Paling Berpengaruh")
    top_features = pd.DataFrame(metadata.get("top_features", []))
    if not top_features.empty:
        st.dataframe(top_features, width="stretch")
