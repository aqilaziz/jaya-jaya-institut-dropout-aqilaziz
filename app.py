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


def dropout_probability(estimator, input_data, feature_names):
    proba = estimator.predict_proba(input_data[feature_names])
    classes = list(getattr(estimator, "classes_", [0, 1]))
    dropout_index = classes.index(1) if 1 in classes else 1
    return proba[:, dropout_index]


def build_enrolled_risk_table(data, estimator, feature_names):
    enrolled = data[data["Status"] == "Enrolled"].copy()
    if enrolled.empty:
        return enrolled

    enrolled["Dropout_Risk"] = dropout_probability(estimator, enrolled, feature_names)
    columns = [
        "Dropout_Risk",
        "Age_at_enrollment",
        "Tuition_fees_up_to_date",
        "Debtor",
        "Scholarship_holder",
        "Curricular_units_1st_sem_approved",
        "Curricular_units_2nd_sem_approved",
        "Curricular_units_1st_sem_grade",
        "Curricular_units_2nd_sem_grade",
    ]
    return enrolled[columns].sort_values("Dropout_Risk", ascending=False)


model = load_model()
metadata = load_metadata()
df = load_sample_data()
features = metadata["features"]
threshold = metadata.get("threshold", 0.5)
enrolled_risk = build_enrolled_risk_table(df, model, features)

st.title("Dashboard dan Prototype Risiko Dropout")
st.caption("Jaya Jaya Institut")
st.info(
    "Model dilatih hanya menggunakan data mahasiswa berstatus Dropout dan Graduate. "
    "Data Enrolled digunakan sebagai data monitoring dan inferensi risiko."
)

dashboard_tab, prototype_tab = st.tabs(["Dashboard Monitoring", "Prototype Prediksi"])

with dashboard_tab:
    st.subheader("Monitoring Risiko Dropout")

    filter_col, threshold_col = st.columns([2, 1])
    with filter_col:
        selected_status = st.multiselect(
            "Filter status mahasiswa",
            options=["Dropout", "Graduate", "Enrolled"],
            default=["Dropout", "Graduate", "Enrolled"],
        )
    with threshold_col:
        risk_threshold = st.slider(
            "Ambang risiko Enrolled",
            min_value=0.0,
            max_value=1.0,
            value=float(threshold),
            step=0.05,
        )

    filtered_df = df[df["Status"].isin(selected_status)] if selected_status else df.iloc[0:0]
    completed_df = filtered_df[filtered_df["Status"].isin(["Dropout", "Graduate"])]
    dropout_count = int((filtered_df["Status"] == "Dropout").sum())
    graduate_count = int((filtered_df["Status"] == "Graduate").sum())
    enrolled_count = int((filtered_df["Status"] == "Enrolled").sum())
    dropout_rate = dropout_count / len(completed_df) if len(completed_df) else 0

    metric_cols = st.columns(5)
    metric_cols[0].metric("Total Mahasiswa", f"{len(filtered_df):,}")
    metric_cols[1].metric("Dropout", f"{dropout_count:,}")
    metric_cols[2].metric("Graduate", f"{graduate_count:,}")
    metric_cols[3].metric("Enrolled", f"{enrolled_count:,}")
    metric_cols[4].metric("Dropout Rate", f"{dropout_rate:.1%}")

    chart_col, factor_col = st.columns([1, 1])
    with chart_col:
        st.markdown("**Distribusi Status Mahasiswa**")
        status_order = ["Dropout", "Graduate", "Enrolled"]
        status_counts = (
            filtered_df["Status"]
            .value_counts()
            .reindex(status_order, fill_value=0)
            .reset_index()
        )
        status_counts.columns = ["Status", "Jumlah"]
        st.bar_chart(status_counts, x="Status", y="Jumlah")

    with factor_col:
        st.markdown("**Rata-rata Performa Akademik per Status**")
        academic_features = [
            "Curricular_units_1st_sem_approved",
            "Curricular_units_2nd_sem_approved",
            "Curricular_units_1st_sem_grade",
            "Curricular_units_2nd_sem_grade",
        ]
        selected_metric = st.selectbox(
            "Metrik akademik",
            options=academic_features,
            index=1,
        )
        factor_summary = (
            filtered_df.groupby("Status", as_index=False)[selected_metric]
            .mean()
            .rename(columns={selected_metric: "Rata-rata"})
        )
        st.bar_chart(factor_summary, x="Status", y="Rata-rata")

    risk_factor_col, table_col = st.columns([1, 1])
    with risk_factor_col:
        st.markdown("**Dropout Rate Berdasarkan Faktor Intervensi**")
        intervention_factor = st.selectbox(
            "Faktor",
            options=["Tuition_fees_up_to_date", "Debtor", "Scholarship_holder"],
        )
        factor_completed = df[df["Status"].isin(["Dropout", "Graduate"])].copy()
        factor_completed["is_dropout"] = factor_completed["Status"].eq("Dropout")
        factor_rate = (
            factor_completed.groupby(intervention_factor, as_index=False)["is_dropout"]
            .mean()
            .rename(columns={intervention_factor: "Kategori", "is_dropout": "Dropout Rate"})
        )
        st.bar_chart(factor_rate, x="Kategori", y="Dropout Rate")

    with table_col:
        st.markdown("**Mahasiswa Enrolled Berisiko Tinggi**")
        high_risk = enrolled_risk[enrolled_risk["Dropout_Risk"] >= risk_threshold].copy()
        high_risk["Dropout_Risk"] = high_risk["Dropout_Risk"].map(lambda value: f"{value:.1%}")
        st.dataframe(high_risk.head(20), width="stretch")

with prototype_tab:
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
        sem1_grade = st.number_input(
            "Curricular units 1st sem grade",
            0.0,
            20.0,
            float(row["Curricular_units_1st_sem_grade"]),
        )
        sem2_grade = st.number_input(
            "Curricular units 2nd sem grade",
            0.0,
            20.0,
            float(row["Curricular_units_2nd_sem_grade"]),
        )

    row["Age_at_enrollment"] = age
    row["Tuition_fees_up_to_date"] = tuition
    row["Debtor"] = debtor
    row["Scholarship_holder"] = scholarship
    row["Curricular_units_1st_sem_approved"] = sem1_approved
    row["Curricular_units_2nd_sem_approved"] = sem2_approved
    row["Curricular_units_1st_sem_grade"] = sem1_grade
    row["Curricular_units_2nd_sem_grade"] = sem2_grade

    input_df = pd.DataFrame([row], columns=features)
    prob = dropout_probability(model, input_df, features)[0]
    prediction = "Berisiko Dropout" if prob >= threshold else "Tidak Berisiko Tinggi"

    with right:
        st.subheader("Hasil Prediksi")
        st.metric("Probabilitas Dropout", f"{prob:.1%}")
        st.metric("Klasifikasi", prediction)
        st.write("Metrik validasi model")
        st.json(
            {
                "accuracy": round(metadata["accuracy"], 3),
                "precision": round(metadata["precision"], 3),
                "recall": round(metadata["recall"], 3),
                "f1": round(metadata["f1"], 3),
                "training_rows": metadata["training_rows"],
                "excluded_enrolled_rows": metadata["excluded_from_training"]["Enrolled"],
            }
        )

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
