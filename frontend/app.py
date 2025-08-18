# streamlit_app.py
import os
import json
import requests
import streamlit as st

# ----------------------------
# Config: API URL
# ----------------------------
DEFAULT_API_URL = "https://churn-api-587466025917.asia-southeast2.run.app/predict"

def get_api_url() -> str:
    # Bisa override lewat st.secrets["api_url"] atau env CHURN_API_URL
    return (
        st.secrets.get("api_url", None)
        if hasattr(st, "secrets")
        else None
    ) or os.environ.get("CHURN_API_URL", DEFAULT_API_URL)

API_URL = get_api_url()

# ----------------------------
# Helper
# ----------------------------
def post_predict(payload: dict, timeout: int = 20) -> dict:
    """Panggil API /predict. Return dict hasil atau raise Exception."""
    headers = {"Content-Type": "application/json"}
    resp = requests.post(API_URL, headers=headers, json=payload, timeout=timeout)
    if resp.status_code != 200:
        # coba ambil error detail dari body
        try:
            err = resp.json()
        except Exception:
            err = resp.text
        raise RuntimeError(f"API error {resp.status_code}: {err}")
    return resp.json()

def as_int(x):  # safety cast
    try:
        return int(x)
    except Exception:
        return x

def as_float(x):
    try:
        return float(x)
    except Exception:
        return x

# ----------------------------
# UI
# ----------------------------
st.set_page_config(page_title="Churn Predictor", page_icon="📉", layout="centered")

st.title("📉 Bank Customer Churn – UI")
st.caption("Front-end Streamlit yang memanggil Cloud Run API untuk prediksi churn.")

with st.sidebar:
    st.subheader("⚙️ Pengaturan")
    st.write("Endpoint API yang dipakai:")
    st.code(API_URL, language="text")
    st.info(
        "Kamu bisa override lewat `st.secrets['api_url']` "
        "atau env `CHURN_API_URL` saat deploy."
    )
    st.markdown("---")
    st.markdown("**Tips:**")
    st.markdown("- 0 = **No Churn**, 1 = **Churn**")
    st.markdown("- Probability = peluang ke kelas **Churn**")

st.markdown("### Masukkan Fitur")

with st.form("churn_form"):
    col1, col2 = st.columns(2)

    with col1:
        credit_score = st.number_input("Credit Score", min_value=0, value=650, step=1)
        country = st.selectbox("Country", ["France", "Germany", "Spain"], index=0)
        gender = st.selectbox("Gender", ["Male", "Female"], index=0)
        age = st.number_input("Age", min_value=0, max_value=120, value=40, step=1)
        tenure = st.number_input("Tenure (years)", min_value=0, max_value=20, value=3, step=1)

    with col2:
        balance = st.number_input("Balance", min_value=0.0, value=50000.0, step=100.0, format="%.2f")
        products_number = st.selectbox("Products Number", [1, 2, 3, 4], index=0)
        credit_card_yn = st.selectbox("Has Credit Card?", ["Yes", "No"], index=0)
        active_member_yn = st.selectbox("Active Member?", ["Yes", "No"], index=0)
        estimated_salary = st.number_input("Estimated Salary", min_value=0.0, value=90000.0, step=100.0, format="%.2f")

    submitted = st.form_submit_button("🔮 Prediksi")

if submitted:
    # map Yes/No -> 1/0
    credit_card = 1 if credit_card_yn == "Yes" else 0
    active_member = 1 if active_member_yn == "Yes" else 0

    payload = {
        "credit_score": as_int(credit_score),
        "country": country,
        "gender": gender,
        "age": as_int(age),
        "tenure": as_int(tenure),
        "balance": as_float(balance),
        "products_number": as_int(products_number),
        "credit_card": as_int(credit_card),
        "active_member": as_int(active_member),
        "estimated_salary": as_float(estimated_salary),
    }

    with st.expander("🔎 Payload yang dikirim", expanded=False):
        st.json(payload)

    with st.spinner("Memanggil API…"):
        try:
            result = post_predict(payload)
        except Exception as e:
            st.error(f"Gagal memanggil API: {e}")
        else:
            # Contoh response:
            # { "prediction": 0, "label": 0, "probability": 0.0904... }
            prediction = result.get("prediction", result.get("label"))
            probability = float(result.get("probability", 0.0))

            label_text = "No Churn" if prediction == 0 else "Churn"
            color = "green" if prediction == 0 else "red"

            st.markdown("### Hasil")
            st.metric(
                label="Prediksi",
                value=label_text,
                delta=f"Prob: {probability:.2%}",
                help="Probabilitas menuju kelas Churn"
            )

            # Progress bar untuk probabilitas churn
            st.progress(min(max(probability, 0.0), 1.0))

            with st.expander("📦 Raw Response"):
                st.json(result)

# Footer kecil
st.markdown("---")
st.caption("Made with ❤️ Streamlit • Model served via Cloud Run")
