from datetime import datetime, timezone
from io import BytesIO
import os
from pathlib import Path
import sqlite3
from uuid import uuid4

import numpy as np
import pandas as pd
import streamlit as st

try:
    import psycopg
    from psycopg.rows import dict_row
except ImportError:  # Local SQLite mode does not require the PostgreSQL driver.
    psycopg = None
    dict_row = None

BASE = Path(__file__).resolve().parent
DATA = BASE / "data"
DB_PATH = DATA / "healthcare_patients.db"
APP_VERSION = "2.2.0-clinician-review"
NUMERIC = ["age_years", "glucose_mg_dl", "systolic_bp_mmhg", "diastolic_bp_mmhg", "cholesterol_mg_dl", "heart_rate_bpm"]
BATCH_COLUMNS = ["patient_id", "age_years", "condition", "glucose_mg_dl", "systolic_bp_mmhg", "diastolic_bp_mmhg", "cholesterol_mg_dl", "heart_rate_bpm", "recorded_allergy", "family_history", "adherence_level"]

st.set_page_config(page_title="Healthcare QML", page_icon="🧬", layout="wide")


@st.cache_data
def sheet(file, name):
    return pd.read_excel(DATA / file, sheet_name=name)


@st.cache_data
def load_data():
    integration = "end_to_end_system_integration_results.xlsx"
    safety = "patient_risk_safety_layer_results.xlsx"
    ranking = "personalized_medicine_ranking_results.xlsx"
    benchmark = "quantum_model_analysis_benchmarking.xlsx"
    reliability = "robustness_reliability_analysis_results.xlsx"
    explain = "explainability_uncertainty_analysis.xlsx"
    return {
        "patients": sheet(integration, "Patient_Results"),
        "ranked": sheet(integration, "Ranked_Recommendations"),
        "metrics": sheet(integration, "System_Metrics"),
        "checks": sheet(integration, "Integration_Checks"),
        "risk": sheet(safety, "Patient_Risk_Profiles"),
        "disease_map": sheet(safety, "Disease_Medicine_Map"),
        "ranking_metrics": sheet(ranking, "Ranking_Metrics"),
        "models": sheet("final_model_performance_evaluation.xlsx", "Final_Model_Ranking"),
        "benchmark": sheet(benchmark, "Performance_Benchmark"),
        "circuit": sheet(benchmark, "Circuit_Analysis"),
        "reliability": sheet(reliability, "Reliability_Summary"),
        "perturbation": sheet(reliability, "Perturbation_Summary"),
        "importance": sheet(explain, "Permutation_Importance"),
    }


def database_url():
    """Return a secret managed-PostgreSQL URL without exposing it in the UI."""
    if os.getenv("DATABASE_URL"):
        return os.environ["DATABASE_URL"]
    try:
        return str(st.secrets.get("DATABASE_URL", "")).strip()
    except Exception:
        return ""


def uses_cloud_database():
    return bool(database_url())


def database_connection():
    """Open managed PostgreSQL in cloud mode or SQLite as a local fallback."""
    url = database_url()
    if url:
        if psycopg is None:
            raise RuntimeError("PostgreSQL driver is unavailable; reinstall requirements.txt")
        return psycopg.connect(
            url,
            row_factory=dict_row,
            connect_timeout=15,
            prepare_threshold=None,
        )
    connection = sqlite3.connect(DB_PATH, timeout=30)
    connection.row_factory = sqlite3.Row
    return connection


def sql_parameters(count):
    marker = "%s" if uses_cloud_database() else "?"
    return ", ".join([marker] * count)


def initialize_database():
    DATA.mkdir(parents=True, exist_ok=True)
    with database_connection() as connection:
        connection.execute("""
            CREATE TABLE IF NOT EXISTS saved_patients (
                patient_id TEXT PRIMARY KEY,
                age_years INTEGER NOT NULL,
                condition TEXT NOT NULL,
                glucose_mg_dl REAL NOT NULL,
                systolic_bp_mmhg REAL NOT NULL,
                diastolic_bp_mmhg REAL NOT NULL,
                cholesterol_mg_dl REAL NOT NULL,
                heart_rate_bpm REAL NOT NULL,
                recorded_allergy TEXT NOT NULL,
                family_history TEXT NOT NULL,
                adherence_level TEXT NOT NULL,
                top_medication_class TEXT,
                research_score REAL,
                risk_score INTEGER NOT NULL,
                risk_level TEXT NOT NULL,
                confidence_level TEXT NOT NULL,
                created_at_utc TEXT NOT NULL,
                updated_at_utc TEXT NOT NULL
            )
        """)
        connection.execute("""
            CREATE TABLE IF NOT EXISTS clinician_prescription_records (
                prescription_id TEXT PRIMARY KEY,
                patient_id TEXT NOT NULL,
                medicine_name TEXT NOT NULL,
                dose TEXT NOT NULL,
                route TEXT NOT NULL,
                frequency TEXT NOT NULL,
                duration TEXT NOT NULL,
                clinical_rationale TEXT NOT NULL,
                reviewer_identifier TEXT NOT NULL,
                verification_status TEXT NOT NULL,
                recorded_at_utc TEXT NOT NULL
            )
        """)


def patient_exists(patient_id):
    marker = "%s" if uses_cloud_database() else "?"
    with database_connection() as connection:
        row = connection.execute(
            f"SELECT 1 FROM saved_patients WHERE patient_id = {marker}", (patient_id,)
        ).fetchone()
    return row is not None


def save_patient(record, result):
    now = datetime.now(timezone.utc).isoformat(timespec="seconds")
    top = result["table"].iloc[0] if not result["table"].empty else None
    values = [record[column] for column in BATCH_COLUMNS]
    values += [
        None if top is None else top["Medication class"],
        None if top is None else float(top["Research score"]),
        int(result["risk_score"]), result["risk_level"],
        result["confidence_level"], now, now,
    ]
    with database_connection() as connection:
        connection.execute(f"""
            INSERT INTO saved_patients (
                patient_id, age_years, condition, glucose_mg_dl,
                systolic_bp_mmhg, diastolic_bp_mmhg, cholesterol_mg_dl,
                heart_rate_bpm, recorded_allergy, family_history,
                adherence_level, top_medication_class, research_score,
                risk_score, risk_level, confidence_level,
                created_at_utc, updated_at_utc
            ) VALUES ({sql_parameters(18)})
        """, values)


def load_saved_patients():
    with database_connection() as connection:
        rows = connection.execute(
            "SELECT * FROM saved_patients ORDER BY created_at_utc DESC"
        ).fetchall()
    return pd.DataFrame([dict(row) for row in rows])


def save_clinician_prescription(record):
    values = [
        record["prescription_id"], record["patient_id"], record["medicine_name"],
        record["dose"], record["route"], record["frequency"], record["duration"],
        record["clinical_rationale"], record["reviewer_identifier"],
        record["verification_status"], record["recorded_at_utc"],
    ]
    with database_connection() as connection:
        connection.execute(f"""
            INSERT INTO clinician_prescription_records (
                prescription_id, patient_id, medicine_name, dose, route,
                frequency, duration, clinical_rationale, reviewer_identifier,
                verification_status, recorded_at_utc
            ) VALUES ({sql_parameters(11)})
        """, values)


def load_clinician_prescriptions(patient_id=None):
    query = "SELECT * FROM clinician_prescription_records"
    parameters = ()
    if patient_id:
        marker = "%s" if uses_cloud_database() else "?"
        query += f" WHERE patient_id = {marker}"
        parameters = (patient_id,)
    query += " ORDER BY recorded_at_utc DESC"
    with database_connection() as connection:
        rows = connection.execute(query, parameters).fetchall()
    return pd.DataFrame([dict(row) for row in rows])


def pct(x):
    return f"{float(x):.1%}"


def validate(record, conditions):
    errors = []
    ranges = {
        "age_years": (18, 100), "glucose_mg_dl": (40, 500),
        "systolic_bp_mmhg": (60, 260), "diastolic_bp_mmhg": (35, 160),
        "cholesterol_mg_dl": (80, 500), "heart_rate_bpm": (30, 220),
    }
    for field, (low, high) in ranges.items():
        try:
            value = float(record[field])
            if not low <= value <= high:
                errors.append(f"{field} must be between {low} and {high}")
        except (KeyError, TypeError, ValueError):
            errors.append(f"{field} must be numeric")
    if not str(record.get("patient_id", "")).strip(): errors.append("patient_id is required")
    if record.get("condition") not in conditions: errors.append("condition is not recognized")
    if record.get("family_history") not in {"Yes", "No"}: errors.append("family_history must be Yes or No")
    if record.get("adherence_level") not in {"Low", "Medium", "High"}: errors.append("invalid adherence_level")
    return errors


def risk_review(r):
    score, reasons = 0, []
    g, s, d, c, h, age = [float(r[x]) for x in ["glucose_mg_dl", "systolic_bp_mmhg", "diastolic_bp_mmhg", "cholesterol_mg_dl", "heart_rate_bpm", "age_years"]]
    if g < 55 or g > 300: score += 4; reasons.append("Critical glucose review")
    elif g > 200: score += 2; reasons.append("High glucose review")
    if s >= 180 or d >= 120: score += 4; reasons.append("Critical blood-pressure review")
    elif s < 90 or d < 60: score += 2; reasons.append("Low blood-pressure review")
    elif s >= 140 or d >= 90: score += 2; reasons.append("Elevated blood-pressure review")
    if c >= 240: score += 2; reasons.append("High cholesterol review")
    if h < 50 or h > 120: score += 2; reasons.append("Heart-rate review")
    if age >= 65: score += 1; reasons.append("Older-adult review")
    if str(r["recorded_allergy"]).lower() != "none recorded": score += 2; reasons.append("Allergy verification required")
    if r["adherence_level"] == "Low": score += 1; reasons.append("Low adherence review")
    if r["family_history"] == "Yes": score += 1; reasons.append("Positive family history")
    level = "Critical Review" if score >= 8 else "High Review" if score >= 5 else "Moderate Review" if score >= 2 else "Routine Review"
    return score, level, reasons or ["No rule-based risk flags"]


def similar_profiles(record, profiles, n=12):
    pool = profiles[profiles.condition == record["condition"]].copy()
    if pool.empty: pool = profiles.copy()
    matrix = pool[NUMERIC].astype(float)
    scale = matrix.std(ddof=0).replace(0, 1)
    query = pd.Series({x: float(record[x]) for x in NUMERIC})
    distance = np.sqrt((((matrix - query) / scale) ** 2).mean(axis=1))
    pool["similarity"] = 1 / (1 + distance)
    return pool.nlargest(n, "similarity")


def recommend(record, data):
    neighbors = similar_profiles(record, data["risk"])
    ranked = data["ranked"][data["ranked"].patient_id.isin(neighbors.patient_id)].copy()
    weights = neighbors.set_index("patient_id")["similarity"]
    ranked["weighted_score"] = ranked.model_score.astype(float) * ranked.patient_id.map(weights)
    allowed = data["disease_map"]
    allowed = allowed[(allowed.disease == record["condition"]) & allowed.disease_compatible.astype(bool)]
    allowed = allowed.recommended_medication_class.unique().tolist()
    ranked = ranked[ranked.recommended_medication_class.isin(allowed)]
    scores = ranked.groupby("recommended_medication_class").weighted_score.sum().sort_values(ascending=False)
    if scores.empty: scores = pd.Series({x: 1.0 for x in allowed}, dtype=float)
    scores = scores.head(3); scores = scores / scores.sum()
    risk_score, risk_level, reasons = risk_review(record)
    rows = []
    for rank, (name, score) in enumerate(scores.items(), 1):
        review = risk_score >= 5 or str(record["recorded_allergy"]).lower() != "none recorded" or score < .40
        rows.append({"Rank": rank, "Medication class": name, "Research score": float(score), "Compatibility": "Disease-compatible", "Professional review": "Required" if review else "Standard"})
    table = pd.DataFrame(rows)
    confidence = 0 if table.empty else float(table.iloc[0]["Research score"])
    return {"table": table, "neighbors": neighbors.head(5), "risk_score": risk_score, "risk_level": risk_level, "reasons": reasons, "confidence": confidence, "confidence_level": "High" if confidence >= .70 else "Moderate" if confidence >= .40 else "Low"}


def pdf_report(record, result):
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import getSampleStyleSheet
    from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle
    stream = BytesIO(); styles = getSampleStyleSheet(); story = []
    story += [Paragraph("Personalized Healthcare Research Report", styles["Title"]), Paragraph("Synthetic similarity-based demonstration — not a diagnosis or prescription.", styles["Italic"]), Spacer(1, 10)]
    story.append(Table([["Field", "Value"], *[[k.replace("_", " ").title(), str(v)] for k, v in record.items()]], repeatRows=1))
    story += [Spacer(1, 10), Paragraph("Ranked medication classes", styles["Heading2"])]
    table = result["table"].copy(); table["Research score"] = table["Research score"].map(pct)
    rec_table = Table([table.columns.tolist(), *table.astype(str).values.tolist()], repeatRows=1)
    rec_table.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#0F766E")), ("TEXTCOLOR", (0, 0), (-1, 0), colors.white), ("GRID", (0, 0), (-1, -1), .4, colors.grey), ("FONTSIZE", (0, 0), (-1, -1), 7)]))
    story += [rec_table, Spacer(1, 10), Paragraph(f"Safety: {result['risk_level']} (score {result['risk_score']})", styles["Heading2"]), Paragraph("; ".join(result["reasons"]), styles["BodyText"]), Spacer(1, 12), Paragraph("Research use only. Synthetic historical patterns are not clinical advice. Professional review is required.", styles["BodyText"])]
    SimpleDocTemplate(stream, pagesize=A4).build(story)
    return stream.getvalue()


def log(event, **details):
    st.session_state.audit.append({"timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"), "event": event, **details})


def page_header(title, subtitle):
    st.markdown(f"## {title}")
    st.markdown(f'<p class="page-subtitle">{subtitle}</p>', unsafe_allow_html=True)


def section_note(text):
    st.markdown(f'<div class="section-note">{text}</div>', unsafe_allow_html=True)


st.markdown("""<style>
.stApp{background:radial-gradient(circle at 92% 4%,rgba(14,165,164,.10),transparent 27rem),#f7fafc;color:#173b54}.block-container{max-width:1380px;padding-top:1.4rem}[data-testid="stSidebar"]{background:linear-gradient(180deg,#10233f,#173b54 55%,#0f5960)}[data-testid="stSidebar"] *{color:#eef9fb}.sidebar-title{display:flex;align-items:center;gap:.55rem;font-size:1.35rem;font-weight:750;color:#fff!important;margin:.25rem 0 .65rem}.sidebar-title span{color:#fff!important}[data-testid="stSidebar"] h1,[data-testid="stSidebar"] h2,[data-testid="stSidebar"] h3,[data-testid="stSidebar"] h4,[data-testid="stSidebar"] strong{color:#f8ffff!important}[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p{color:#eef9fb!important}[data-testid="stSidebar"] [data-testid="stCaptionContainer"] p{color:#a9c7d0!important}.hero{padding:1.6rem 1.8rem;border-radius:22px;color:#fff!important;background:linear-gradient(125deg,#10233f,#155e75 52%,#0f9d91);box-shadow:0 14px 35px rgba(16,35,63,.18)}.hero h1{margin:0;color:#fff!important}.hero p{color:#d7f5f4!important}.notice{padding:.85rem 1rem;margin:.8rem 0;border-radius:12px;border:1px solid #fed7aa;border-left:5px solid #f59e0b;background:#fff7ed;color:#7c2d12!important}.notice *{color:#7c2d12!important}[data-testid="stMetric"]{background:#fff;border:1px solid #dbe7ec;padding:.8rem;border-radius:14px;box-shadow:0 6px 18px rgba(16,35,63,.05)}[data-testid="stMetricValue"]{color:#0f766e!important}[data-testid="stMetricLabel"] p{color:#506779!important}h1,h2,h3,h4{color:#173b54!important}.page-subtitle{font-size:1rem;color:#587184!important;margin-top:-.6rem;margin-bottom:1.2rem;max-width:1050px}.section-note{padding:.8rem 1rem;margin:.4rem 0 1rem;border-radius:10px;background:#eaf5f5;border-left:4px solid #0f9d91;color:#173b54!important}.section-note *{color:#173b54!important}.definition-card{padding:1rem;border:1px solid #dbe7ec;border-radius:14px;background:#fff;min-height:140px;box-shadow:0 5px 16px rgba(16,35,63,.04);color:#173b54!important}.definition-card *{color:#173b54!important}[data-testid="stMainBlockContainer"] p,[data-testid="stMainBlockContainer"] label,[data-testid="stMainBlockContainer"] li{color:#173b54}[data-testid="stMainBlockContainer"] [data-testid="stCaptionContainer"] p{color:#60788a!important}[data-testid="stMainBlockContainer"] [data-baseweb="select"] *{color:#173b54}
</style>""", unsafe_allow_html=True)

try:
    initialize_database()
    data = load_data()
except Exception as exc: st.error(f"Could not load deployment data: {exc}"); st.stop()
if "audit" not in st.session_state: st.session_state.audit = []
if "last" not in st.session_state: st.session_state.last = None
conditions = sorted(data["risk"].condition.unique()); allergies = sorted(data["risk"].recorded_allergy.unique())

st.markdown('<div class="hero"><h1>Personalized Healthcare & Medicine Recommendation</h1><p>Hybrid QML research dashboard with ranking, safety, explainability and monitoring</p></div>', unsafe_allow_html=True)
st.markdown('<div class="notice"><b>Research Use Only:</b> This application uses synthetic data and historical medication-class patterns. It is not a diagnostic or prescribing tool. A qualified healthcare professional must review every output.</div>', unsafe_allow_html=True)

st.sidebar.markdown(
    '<div class="sidebar-title">🧬 <span>Healthcare QML</span></div>',
    unsafe_allow_html=True,
)
st.sidebar.caption("Personalized medication-class research platform")
role = st.sidebar.selectbox(
    "Select demonstration role",
    ["Researcher", "Healthcare Reviewer", "Administrator"],
    help="The selected role changes which analytical and administrative pages are visible. This is a demonstration, not secure authentication.",
)
pages = [
    "🏠 Executive Overview",
    "🩺 Generate Recommendation",
    "👥 Patient Records Explorer",
    "💊 Medication Class Comparison",
    "🛡️ Safety & Reliability",
    "⚛️ Quantum & Benchmark Analysis",
    "📦 Batch Recommendation Processing",
]
if role in {"Healthcare Reviewer", "Administrator"}:
    pages.insert(3, "🧾 Clinician Prescription Record")
if role in {"Researcher", "Administrator"}:
    pages.append("📈 Model Evaluation")
if role == "Administrator":
    pages += ["📡 Operational Monitoring", "✅ System Validation & Dictionary"]
page = st.sidebar.radio("Dashboard sections", pages)
st.sidebar.markdown("---")
st.sidebar.caption("Synthetic data · Research use only · Professional review required")
st.sidebar.caption(
    "Storage: Managed PostgreSQL (durable)"
    if uses_cloud_database()
    else "Storage: Local SQLite fallback"
)
st.sidebar.caption(f"Application version: {APP_VERSION}")

if page == "🏠 Executive Overview":
    page_header(
        "Executive Overview",
        "A guided summary of the dataset, recommendation workflow, safety controls and analytical evidence available in this synthetic healthcare research prototype.",
    )
    saved_count = len(load_saved_patients())
    cols = st.columns(5)
    cols[0].metric("Baseline synthetic records", len(data["patients"]), help="Synthetic profiles included in the integrated test results.")
    cols[1].metric("Persistently saved patients", saved_count, help="New patient profiles stored in managed PostgreSQL when DATABASE_URL is configured, otherwise local SQLite.")
    cols[2].metric("Supported conditions", data["patients"].condition.nunique(), help="Distinct condition labels represented in the synthetic dataset.")
    cols[3].metric("Medication classes", data["ranked"].recommended_medication_class.nunique(), help="Historical medication categories predicted by the research models—not individual drug prescriptions.")
    cols[4].metric("Integration checks passed", f"{(data['checks'].status == 'PASS').sum()}/{len(data['checks'])}", help="Automated checks covering required files, feature compatibility and output integrity.")

    st.markdown("### What this system does")
    section_note(
        "The platform ranks historical medication classes for synthetic patient profiles, applies disease-compatibility and rule-based safety checks, presents uncertainty and similar-profile evidence, and compares quantum and classical research models."
    )
    a, b, c = st.columns(3)
    with a:
        st.markdown('<div class="definition-card"><b>1 · Patient information</b><br><br>Validated demographic, condition, measurement, allergy, family-history and adherence fields form the synthetic patient profile.</div>', unsafe_allow_html=True)
    with b:
        st.markdown('<div class="definition-card"><b>2 · Research ranking</b><br><br>Condition-matched synthetic profiles contribute weighted evidence for the top three compatible historical medication classes.</div>', unsafe_allow_html=True)
    with c:
        st.markdown('<div class="definition-card"><b>3 · Safety review</b><br><br>Risk rules, allergy verification, uncertainty and confidence determine whether professional review is required.</div>', unsafe_allow_html=True)

    st.markdown("### Record and output definitions")
    definitions = pd.DataFrame([
        ["Patient ID", "A unique synthetic identifier used to connect profile, ranking and safety records."],
        ["Condition", "The synthetic condition category used for compatibility filtering."],
        ["Medication class", "A historical medication category; it is not a named drug, dosage or prescription."],
        ["Research score", "A normalized similarity-weighted ranking score. It is not clinical probability or treatment effectiveness."],
        ["Risk score", "A rule-based sum of review flags for measurements, age, allergy, adherence and family history."],
        ["Risk level", "Routine, Moderate, High or Critical Review category derived from the risk score."],
        ["Professional review", "Indicates that a qualified healthcare professional must assess the output."],
        ["Profile similarity", "Closeness between standardized synthetic measurement profiles; it does not establish medical equivalence."],
        ["Coverage", "Percentage of evaluated profiles for which at least one eligible class was available."],
        ["Normalized entropy", "Prediction uncertainty measure; higher values indicate a less concentrated model output."],
        ["Hit Rate@K", "Share of test profiles where the recorded class appears within the first K ranked classes."],
        ["NDCG@3", "Ranking-quality score that rewards placing the recorded class nearer the top three positions."],
    ], columns=["Term", "Meaning in this dashboard"])
    st.dataframe(definitions, hide_index=True, use_container_width=True)
    st.info("New-patient rankings are a similarity-based demonstration over precomputed synthetic results—not live QML inference and not clinical advice.")

elif page == "🩺 Generate Recommendation":
    page_header(
        "Generate a Synthetic Research Recommendation",
        "Enter a synthetic patient profile to produce a validated top-three medication-class ranking, safety review, explanation and downloadable research report.",
    )
    storage_name = "managed PostgreSQL" if uses_cloud_database() else "local SQLite fallback"
    section_note(f"Complete every field. Validated profiles are stored in {storage_name}. Input limits are demonstration validation rules, not medical reference ranges.")
    with st.form("patient"):
        st.markdown("#### A. Patient and condition information")
        a, b, c = st.columns(3)
        patient_id = a.text_input("Patient ID", f"PAT-{uuid4().hex[:6].upper()}", help="Unique identifier stored with the permanent patient record.")
        age = b.number_input("Age (years)", 18, 100, 45)
        condition = c.selectbox("Recorded condition", conditions)
        st.markdown("#### B. Clinical measurements")
        a, b, c = st.columns(3)
        glucose = a.number_input("Glucose (mg/dL)", 40, 500, 110)
        systolic = b.number_input("Systolic blood pressure (mmHg)", 60, 260, 125)
        diastolic = c.number_input("Diastolic blood pressure (mmHg)", 35, 160, 80)
        cholesterol = a.number_input("Cholesterol (mg/dL)", 80, 500, 190)
        heart = b.number_input("Heart rate (bpm)", 30, 220, 75)
        st.markdown("#### C. Safety and behavioural information")
        a, b, c = st.columns(3)
        allergy = a.selectbox("Recorded allergy", allergies)
        family = b.selectbox("Positive family history", ["No", "Yes"])
        adherence = c.selectbox("Historical adherence level", ["High", "Medium", "Low"])
        consent = st.checkbox(
            "I confirm this is synthetic/research data and may be stored in the application database.",
            help="Do not enter identifiable real-world patient information in this research prototype.",
        )
        submitted = st.form_submit_button("Save patient and generate recommendation", type="primary")
    if submitted:
        record = dict(patient_id=patient_id.strip(), age_years=age, condition=condition, glucose_mg_dl=glucose, systolic_bp_mmhg=systolic, diastolic_bp_mmhg=diastolic, cholesterol_mg_dl=cholesterol, heart_rate_bpm=heart, recorded_allergy=allergy, family_history=family, adherence_level=adherence)
        errors = validate(record, conditions)
        if not consent:
            errors.append("Confirm the research-data storage statement before saving")
        if patient_exists(record["patient_id"]):
            errors.append("patient_id already exists in permanent storage; use a unique Patient ID")
        if errors:
            for error in errors:
                st.error(error)
        else:
            result = recommend(record, data)
            save_patient(record, result)
            st.session_state.last = (record, result)
            log("patient_saved_and_recommendation_generated", patient_id=patient_id, condition=condition, risk_level=result["risk_level"], confidence=round(result["confidence"], 4))
            st.success(f"Patient {record['patient_id']} was saved in {storage_name}.")
    if st.session_state.last:
        record, result = st.session_state.last
        st.markdown("### Recommendation summary")
        cols = st.columns(4)
        cols[0].metric("Safety category", result["risk_level"], help="Rule-based review category; not a diagnosis.")
        cols[1].metric("Risk score", result["risk_score"], help="Sum of triggered review-rule weights.")
        cols[2].metric("Top research score", pct(result["confidence"]), help="Normalized ranking score; not clinical probability.")
        cols[3].metric("Ranking confidence", result["confidence_level"])
        shown = result["table"].copy()
        shown["Research score"] = shown["Research score"].map(pct)
        st.markdown("#### Disease-compatible top-three classes")
        st.dataframe(shown, hide_index=True, use_container_width=True)
        st.caption("Ranks are based on synthetic similarity-weighted evidence and compatibility filtering. They are not prescriptions.")
        left, right = st.columns(2)
        with left:
            st.markdown("#### Triggered safety-review reasons")
            for reason in result["reasons"]:
                st.write(f"- {reason}")
        with right:
            st.markdown("#### How the result was produced")
            st.write(f"- Condition-matched comparison for **{record['condition']}**")
            st.write(f"- Evidence from the **{len(result['neighbors'])} closest displayed synthetic profiles**")
            st.write("- Candidate classes restricted by the disease–medicine compatibility map")
            st.write("- Low score, high risk or an allergy flag activates professional review")
        st.markdown("#### Similar synthetic profiles used as supporting evidence")
        sim = result["neighbors"][["patient_id", "condition", "age_years", "historical_medication_class", "similarity"]].copy()
        sim.columns = ["Patient ID", "Condition", "Age", "Historical medication class", "Profile similarity"]
        sim["Profile similarity"] = sim["Profile similarity"].map(pct)
        st.dataframe(sim, hide_index=True, use_container_width=True)
        st.download_button("Download complete PDF research report", pdf_report(record, result), f"{record['patient_id']}_research_report.pdf", "application/pdf")

elif page == "👥 Patient Records Explorer":
    page_header("Patient Records Explorer", "Review baseline synthetic records or patient profiles permanently saved through the recommendation form.")
    source = st.radio("Record source", ["Permanently saved patients", "Baseline synthetic dataset"], horizontal=True)
    if source == "Permanently saved patients":
        saved = load_saved_patients()
        if saved.empty:
            st.info("No patients have been saved yet. Open Generate Recommendation to add the first patient.")
        else:
            a, b = st.columns(2)
            saved_conditions = sorted(saved.condition.unique())
            condition_filter = a.selectbox("Filter saved records by condition", ["All conditions", *saved_conditions])
            available = saved if condition_filter == "All conditions" else saved[saved.condition == condition_filter]
            selected = b.selectbox("Select saved patient ID", available.patient_id.tolist())
            row = saved[saved.patient_id == selected].iloc[0]
            cols = st.columns(4)
            cols[0].metric("Recorded condition", row.condition)
            cols[1].metric("Safety category", row.risk_level)
            cols[2].metric("Top research score", "Unavailable" if pd.isna(row.research_score) else pct(row.research_score))
            cols[3].metric("Top medication class", row.top_medication_class or "Unavailable")
            st.markdown("#### Permanently stored patient profile")
            profile = pd.DataFrame(
                [[field.replace("_", " ").title(), row[field]] for field in BATCH_COLUMNS],
                columns=["Field", "Stored value"],
            )
            st.dataframe(profile, hide_index=True, use_container_width=True)
            st.caption(f"Created (UTC): {row.created_at_utc} · Last updated (UTC): {row.updated_at_utc}")
    else:
        a, b = st.columns(2)
        condition_filter = a.selectbox("Filter records by condition", ["All conditions", *conditions])
        available = data["patients"] if condition_filter == "All conditions" else data["patients"][data["patients"].condition == condition_filter]
        selected = b.selectbox("Select synthetic patient ID", available.patient_id.tolist())
        row = data["patients"][data["patients"].patient_id == selected].iloc[0]
        recs = data["ranked"][data["ranked"].patient_id == selected].copy()
        cols = st.columns(4)
        cols[0].metric("Recorded condition", row.condition)
        cols[1].metric("Safety category", row.risk_level)
        cols[2].metric("Raw model confidence", pct(row.raw_prediction_confidence))
        cols[3].metric("Professional review", "Required" if row.requires_professional_review else "Standard")
        st.markdown("#### Safety-adjusted ranked classes")
        st.dataframe(recs, hide_index=True, use_container_width=True)
    with st.expander("Understand these record fields"):
        st.write("**Raw model confidence:** confidence before safety and compatibility adjustments.")
        st.write("**Safety-adjusted rank:** order after removing incompatible candidates and applying review rules.")
        st.write("**Model score:** model-generated class score used for ranking; not treatment effectiveness.")

elif page == "🧾 Clinician Prescription Record":
    page_header(
        "Clinician Medication Review and Prescription Recording",
        "Record a qualified reviewer's independently selected medicine and regimen for a saved synthetic patient. The application does not generate or select the medicine, dose, frequency or duration.",
    )
    st.error(
        "Restricted demonstration workflow: this page does not verify professional identity and must not be used for real prescribing or identifiable patient data."
    )
    saved = load_saved_patients()
    if saved.empty:
        st.info("No saved synthetic patients are available. Save a patient through Generate Recommendation first.")
    else:
        patient_id = st.selectbox("Select a saved synthetic patient", saved.patient_id.tolist())
        patient = saved[saved.patient_id == patient_id].iloc[0]
        a, b, c, d = st.columns(4)
        a.metric("Condition", patient.condition)
        b.metric("Safety category", patient.risk_level)
        c.metric("Recorded allergy", patient.recorded_allergy)
        d.metric("Research medication class", patient.top_medication_class or "Unavailable")
        st.caption(
            "The research medication class is contextual evidence only. It is not a medicine selection and must not determine the clinician-entered record."
        )

        with st.form("clinician_prescription"):
            st.markdown("#### Independent clinician-entered medication decision")
            medicine_name = st.text_input(
                "Medicine name selected independently by the reviewer",
                help="The system does not suggest or autocomplete medicine names.",
            )
            a, b = st.columns(2)
            dose = a.text_input("Dose and unit entered by reviewer", placeholder="Clinician entry required")
            route = b.selectbox(
                "Route entered by reviewer",
                ["Select route", "Oral", "Topical", "Inhaled", "Injection", "Other"],
            )
            a, b = st.columns(2)
            frequency = a.text_input("Frequency entered by reviewer", placeholder="Clinician entry required")
            duration = b.text_input("Duration entered by reviewer", placeholder="Clinician entry required")
            clinical_rationale = st.text_area(
                "Clinical rationale and independent review basis",
                help="Explain the reviewer's independent reasoning and any verification performed.",
            )
            reviewer_identifier = st.text_input(
                "Reviewer identifier",
                help="Use a synthetic staff identifier for this demonstration; do not enter personal registration details.",
            )
            attested = st.checkbox(
                "I attest that a qualified healthcare professional independently selected and verified this entry; it was not generated by the model."
            )
            prescription_submitted = st.form_submit_button(
                "Record clinician-entered decision", type="primary"
            )

        if prescription_submitted:
            required = {
                "Medicine name": medicine_name,
                "Dose": dose,
                "Frequency": frequency,
                "Duration": duration,
                "Clinical rationale": clinical_rationale,
                "Reviewer identifier": reviewer_identifier,
            }
            missing = [name for name, value in required.items() if not str(value).strip()]
            if route == "Select route":
                missing.append("Route")
            if not attested:
                missing.append("Professional attestation")
            if missing:
                st.error(f"Complete the following required fields: {', '.join(missing)}")
            else:
                prescription = {
                    "prescription_id": f"RX-{uuid4().hex[:10].upper()}",
                    "patient_id": patient_id,
                    "medicine_name": medicine_name.strip(),
                    "dose": dose.strip(),
                    "route": route,
                    "frequency": frequency.strip(),
                    "duration": duration.strip(),
                    "clinical_rationale": clinical_rationale.strip(),
                    "reviewer_identifier": reviewer_identifier.strip(),
                    "verification_status": "Clinician-entered and independently attested",
                    "recorded_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                }
                save_clinician_prescription(prescription)
                log(
                    "clinician_prescription_recorded",
                    prescription_id=prescription["prescription_id"],
                    patient_id=patient_id,
                    reviewer_identifier=prescription["reviewer_identifier"],
                )
                st.success(
                    f"Clinician-entered record {prescription['prescription_id']} was stored successfully."
                )

        history = load_clinician_prescriptions(patient_id)
        st.markdown("#### Prescription-record history for this synthetic patient")
        if history.empty:
            st.info("No clinician-entered prescription records exist for this patient.")
        else:
            st.dataframe(history, hide_index=True, use_container_width=True)
            st.download_button(
                "Download clinician-entered history",
                history.to_csv(index=False),
                f"{patient_id}_clinician_prescription_history.csv",
                "text/csv",
            )

elif page == "💊 Medication Class Comparison":
    page_header("Medication Class Comparison", "Compare descriptive synthetic patient characteristics associated with two historical medication classes.")
    section_note("This comparison describes the synthetic dataset. It does not compare drug efficacy, safety, dosage or clinical superiority.")
    classes = sorted(data["risk"].historical_medication_class.unique())
    a, b = st.columns(2)
    class_a = a.selectbox("First historical medication class", classes)
    class_b = b.selectbox("Second historical medication class", classes, index=1)
    rows = []
    for name in [class_a, class_b]:
        group = data["risk"][data["risk"].historical_medication_class == name]
        rows.append({"Medication class": name, "Historical records": len(group), "Mean age (years)": group.age_years.mean(), "Mean glucose (mg/dL)": group.glucose_mg_dl.mean(), "Mean systolic BP (mmHg)": group.systolic_bp_mmhg.mean(), "Mean cholesterol (mg/dL)": group.cholesterol_mg_dl.mean(), "Allergy-review rate": pct(group.allergy_review_flag.mean())})
    st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)

elif page == "🛡️ Safety & Reliability":
    page_header("Safety, Reliability and Uncertainty", "Inspect rule-based professional-review demand, recommendation coverage, prediction uncertainty and robustness under controlled perturbations.")
    cols = st.columns(3)
    cols[0].metric("Professional-review rate", pct(data["patients"].requires_professional_review.mean()), help="Share of evaluated profiles marked for professional assessment.")
    cols[1].metric("Recommendation coverage", pct(data["patients"].recommendation_available.mean()), help="Share of profiles with at least one eligible class.")
    cols[2].metric("Mean normalized entropy", f"{data['patients'].normalized_entropy.mean():.3f}", help="Higher entropy generally indicates greater uncertainty.")
    left, right = st.columns(2)
    with left:
        st.markdown("#### Distribution of safety-review categories")
        st.bar_chart(data["patients"].risk_level.value_counts())
    with right:
        st.markdown("#### Reliability summary")
        st.dataframe(data["reliability"], hide_index=True, use_container_width=True)
    st.markdown("#### Robustness under input perturbation")
    st.dataframe(data["perturbation"], hide_index=True, use_container_width=True)
    st.caption("Robustness measures output stability under controlled synthetic changes; they do not demonstrate clinical reliability.")

elif page == "⚛️ Quantum & Benchmark Analysis":
    page_header("Quantum Model and Benchmark Analysis", "Review circuit complexity, QML performance, classical baselines and feature sensitivity using the completed simulator-based experiments.")
    st.warning("These are simulator-based research benchmarks. The results do not establish quantum advantage or clinical effectiveness.")
    left, right = st.columns(2)
    with left:
        st.markdown("#### Quantum circuit characteristics")
        st.dataframe(data["circuit"], hide_index=True, use_container_width=True)
    with right:
        st.markdown("#### Classical and quantum performance benchmark")
        st.dataframe(data["benchmark"], hide_index=True, use_container_width=True)
    st.markdown("#### Sensitivity of encoded quantum-angle features")
    st.bar_chart(data["importance"].set_index("feature")[["mean_macro_f1_drop"]])
    st.caption("A larger macro-F1 decrease means performance was more sensitive to permutation of that encoded feature.")

elif page == "📦 Batch Recommendation Processing":
    page_header("Batch Recommendation Processing", "Validate and process up to 500 synthetic patient profiles from a CSV file, then download valid rankings and validation failures.")
    section_note("Use the provided template without renaming columns. Every row is independently validated before ranking.")
    template = pd.DataFrame([["BATCH-001", 45, conditions[0], 110, 125, 80, 190, 75, "None recorded", "No", "Medium"]], columns=BATCH_COLUMNS)
    st.download_button("1 · Download CSV input template", template.to_csv(index=False), "patient_batch_template.csv", "text/csv")
    upload = st.file_uploader("2 · Upload the completed CSV file", type="csv")
    if upload:
        batch = pd.read_csv(upload)
        missing = [x for x in BATCH_COLUMNS if x not in batch.columns]
        if missing:
            st.error(f"Missing required columns: {missing}")
        elif batch.patient_id.duplicated().any():
            st.error("Duplicate patient IDs were found. Every patient_id must be unique.")
        elif len(batch) > 500:
            st.error("The demonstration limit is 500 records per upload.")
        else:
            output, invalid = [], []
            for _, row in batch.iterrows():
                record = row[BATCH_COLUMNS].to_dict()
                errors = validate(record, conditions)
                if errors:
                    invalid.append({"patient_id": record.get("patient_id"), "validation_errors": "; ".join(errors)})
                    continue
                result = recommend(record, data)
                top = result["table"].iloc[0] if not result["table"].empty else None
                output.append({"patient_id": record["patient_id"], "condition": record["condition"], "top_medication_class": "" if top is None else top["Medication class"], "research_score": 0 if top is None else top["Research score"], "risk_score": result["risk_score"], "risk_level": result["risk_level"], "professional_review": True if top is None else top["Professional review"] == "Required"})
            st.success(f"Processing completed: {len(output)} valid records and {len(invalid)} invalid records.")
            if output:
                out = pd.DataFrame(output)
                st.markdown("#### Valid batch results")
                st.dataframe(out, hide_index=True, use_container_width=True)
                st.download_button("3 · Download valid results", out.to_csv(index=False), "batch_recommendation_results.csv", "text/csv")
                log("batch_processed", valid=len(output), invalid=len(invalid))
            if invalid:
                st.markdown("#### Rejected rows and validation reasons")
                st.dataframe(pd.DataFrame(invalid), hide_index=True, use_container_width=True)

elif page == "📈 Model Evaluation":
    page_header("Model Evaluation and Ranking Quality", "Compare final QML models, independent classical baselines and top-K ranking usefulness on the held-out synthetic test profiles.")
    metric = dict(zip(data["ranking_metrics"].metric, data["ranking_metrics"].value))
    cols = st.columns(4)
    cols[0].metric("Hit Rate@1", pct(metric.get("Hit Rate@1", 0)), help="Recorded class appears at rank 1.")
    cols[1].metric("Hit Rate@2", pct(metric.get("Hit Rate@2", 0)), help="Recorded class appears within the first two ranks.")
    cols[2].metric("Hit Rate@3", pct(metric.get("Hit Rate@3", 0)), help="Recorded class appears within the first three ranks.")
    cols[3].metric("NDCG@3", f"{metric.get('NDCG@3', 0):.3f}", help="Position-sensitive ranking quality within the top three.")
    st.markdown("#### Final QML model comparison")
    st.dataframe(data["models"], hide_index=True, use_container_width=True)
    st.markdown("#### Independent classical and quantum benchmark")
    st.dataframe(data["benchmark"], hide_index=True, use_container_width=True)
    st.info("The hybrid model leads the final QML comparison, while the RBF SVM remains the strongest overall independent benchmark.")

elif page == "📡 Operational Monitoring":
    page_header("Operational Monitoring and Session Audit", "Track recommendation and batch-processing actions generated during the current browser session and export the audit table.")
    audit = pd.DataFrame(st.session_state.audit)
    cols = st.columns(3)
    cols[0].metric("Session events", len(audit))
    recommendation_events = {"recommendation_generated", "patient_saved_and_recommendation_generated"}
    cols[1].metric("Recommendations generated", 0 if audit.empty else audit.event.isin(recommendation_events).sum())
    cols[2].metric("Batch jobs processed", 0 if audit.empty else (audit.event == "batch_processed").sum())
    if audit.empty:
        st.info("No recommendation or batch-processing events have been recorded in this session.")
    else:
        st.dataframe(audit, hide_index=True, use_container_width=True)
        st.download_button("Download current session audit CSV", audit.to_csv(index=False), "session_audit_log.csv", "text/csv")
    st.caption("This demonstration audit is stored in session memory only and resets when the session ends.")

else:
    page_header("System Validation and Technical Data Dictionary", "Confirm end-to-end integration health and understand the system-level metrics used by the deployed research dashboard.")
    passed = data["checks"].status.astype(str).str.upper().eq("PASS").sum()
    cols = st.columns(3)
    cols[0].metric("Integration checks passed", f"{passed}/{len(data['checks'])}")
    cols[1].metric("Evaluated synthetic profiles", len(data["patients"]))
    cols[2].metric("Recommendation coverage", pct(data["patients"].recommendation_available.mean()))
    st.markdown("#### Automated integration checks")
    st.dataframe(data["checks"], hide_index=True, use_container_width=True)
    st.markdown("#### End-to-end system metrics")
    st.dataframe(data["metrics"], hide_index=True, use_container_width=True)
    st.markdown("#### Technical input-field dictionary")
    dictionary = pd.DataFrame([
        ["patient_id", "Text", "Unique synthetic record identifier"], ["age_years", "Integer", "Synthetic patient age in years"],
        ["condition", "Category", "Recorded synthetic condition label"], ["glucose_mg_dl", "Number", "Glucose measurement in mg/dL"],
        ["systolic_bp_mmhg", "Number", "Systolic blood pressure in mmHg"], ["diastolic_bp_mmhg", "Number", "Diastolic blood pressure in mmHg"],
        ["cholesterol_mg_dl", "Number", "Cholesterol measurement in mg/dL"], ["heart_rate_bpm", "Number", "Heart rate in beats per minute"],
        ["recorded_allergy", "Category", "Synthetic recorded allergy status"], ["family_history", "Yes/No", "Synthetic family-history indicator"],
        ["adherence_level", "Category", "Historical adherence category: Low, Medium or High"],
    ], columns=["Field name", "Expected type", "Dashboard meaning"])
    st.dataframe(dictionary, hide_index=True, use_container_width=True)

st.divider()
st.caption("Synthetic research prototype · Historical medication-class patterns · Not clinically validated · Professional review required")
