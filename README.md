# Personalized Healthcare QML Dashboard

Application version: **2.3.0-patient-portal**

A Streamlit deployment of the personalized healthcare and medicine recommendation research prototype.

## Included functionality

- Interactive synthetic patient form with validation and SQLite storage
- Restricted clinician medication review and prescription-recording workflow
- Independent-review attestation and timestamped prescription audit history
- Patient demonstration role and plain-language saved-profile summary
- Condition education and educational medication-class information
- Condition-matched synthetic percentile comparison
- Interactive professional-visit preparation checklist and questions
- Downloadable patient visit-preparation PDF
- Persistent usability feedback and administrator feedback monitoring
- Top-three disease-compatible medication-class ranking
- Rule-based risk and professional-review layer
- Similar synthetic patient profiles and explanations
- Downloadable PDF research report
- Permanently saved-patient explorer, baseline-record review and medicine-class comparison
- Safety, reliability, robustness and uncertainty views
- Quantum circuit, QML and classical benchmarking
- CSV batch processing with validation and export
- Demonstration roles, session monitoring and audit export

New-patient results are a similarity-based demonstration over synthetic,
precomputed profiles. They are not live QML inference or clinical advice.
The application never generates a medicine name, dose, frequency or duration.
Those fields can only be entered manually through the demonstration clinician
review workflow.

## Run locally

```powershell
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```

## Deploy on Streamlit Community Cloud

1. Push this folder to a GitHub repository.
2. Open https://share.streamlit.io and choose **Create app**.
3. Select the repository and branch.
4. Set **Main file path** to `app.py`.
5. Click **Deploy**.

No secrets are required. Keep the `data` directory in the repository.

## Durable PostgreSQL patient storage

The deployed app supports a managed PostgreSQL database such as Supabase. The
database table is created automatically on first startup. Submitting the form
saves the validated profile, recommendation summary, risk result and timestamps.
Patient IDs are unique and existing records cannot be overwritten accidentally.

### Supabase and Streamlit Community Cloud setup

1. Create a Supabase project.
2. In Supabase, select **Connect** and copy the **Session pooler** PostgreSQL
   connection string. Replace the password placeholder with your database
   password.
3. In Streamlit Community Cloud, open **Manage app → Settings → Secrets**.
4. Add the following secret and save it:

```toml
DATABASE_URL = "postgresql://postgres.PROJECT_REF:PASSWORD@POOLER_HOST:5432/postgres?sslmode=require"
```

5. Reboot the Streamlit app. The sidebar must display
   **Storage: Managed PostgreSQL (durable)**.

Never commit the real connection string or password to GitHub. If
`DATABASE_URL` is missing, the application uses `data/healthcare_patients.db`
as a local-development fallback. Streamlit Community Cloud may reset that
fallback file. Do not store identifiable real patient information in this
synthetic research prototype.

## Important limitation

This application uses synthetic data and precomputed, safety-adjusted research results. It predicts historical medication classes; it is not a diagnostic or prescribing system and is not clinically validated.
