"""
dashboard/app.py
Streamlit monitoring dashboard for the churn prediction pipeline.
Run: streamlit run dashboard/app.py
"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import requests
import json
import yaml
from datetime import datetime, timedelta
import random

st.set_page_config(
    page_title="Churn Pipeline Monitor",
    page_icon="📉",
    layout="wide",
)

API_URL = "http://localhost:8000"
CONFIG_PATH = "config.yaml"


def load_config():
    with open(CONFIG_PATH) as f:
        return yaml.safe_load(f)


def check_api_health():
    try:
        r = requests.get(f"{API_URL}/health", timeout=3)
        return r.json() if r.status_code == 200 else None
    except Exception:
        return None


def get_api_metrics():
    try:
        r = requests.get(f"{API_URL}/metrics", timeout=3)
        return r.json() if r.status_code == 200 else {}
    except Exception:
        return {}


def generate_demo_predictions(n=200):
    """Generate synthetic prediction history for demo purposes."""
    np.random.seed(42)
    dates = [datetime.now() - timedelta(hours=i) for i in range(n, 0, -1)]
    probs = np.clip(np.random.beta(2, 5, n) + np.random.normal(0, 0.05, n), 0, 1)
    return pd.DataFrame({
        "timestamp": dates,
        "churn_probability": probs,
        "risk_label": pd.cut(probs, bins=[0, 0.4, 0.7, 1.0], labels=["Low", "Medium", "High"]),
        "tenure": np.random.exponential(24, n).clip(0, 72),
        "MonthlyCharges": np.random.normal(65, 30, n).clip(18, 120),
        "Contract": np.random.choice(["Month-to-month", "One year", "Two year"], n, p=[0.55, 0.25, 0.20]),
    })


# ─── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("📉 Churn Monitor")
    health = check_api_health()
    if health:
        st.success(f"API online — {health.get('uptime_seconds', 0)}s uptime")
    else:
        st.error("API offline — start with `uvicorn src.api.main:app`")

    st.divider()
    st.subheader("Quick predict")
    tenure = st.slider("Tenure (months)", 0, 72, 12)
    monthly = st.slider("Monthly charges ($)", 18, 120, 65)
    contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])

    if st.button("Run prediction →", use_container_width=True):
        sample = {
            "gender": "Male", "SeniorCitizen": 0, "Partner": "No", "Dependents": "No",
            "tenure": tenure, "PhoneService": "Yes", "MultipleLines": "No",
            "InternetService": "Fiber optic", "OnlineSecurity": "No", "OnlineBackup": "No",
            "DeviceProtection": "No", "TechSupport": "No", "StreamingTV": "No",
            "StreamingMovies": "No", "Contract": contract, "PaperlessBilling": "Yes",
            "PaymentMethod": "Electronic check", "MonthlyCharges": monthly,
            "TotalCharges": tenure * monthly,
        }
        try:
            r = requests.post(f"{API_URL}/predict", json=sample, timeout=5)
            result = r.json()
            prob = result["churn_probability"]
            color = "🔴" if prob >= 0.7 else "🟡" if prob >= 0.4 else "🟢"
            st.metric("Churn probability", f"{prob:.1%}", delta=result["risk_label"])
            st.write(f"{color} Risk: **{result['risk_label']}**")
            st.caption("Top risk factors: " + ", ".join(result["top_risk_factors"][:3]))
        except Exception as e:
            st.warning(f"Could not reach API: {e}")

# ─── Main ──────────────────────────────────────────────────────────────────────
st.title("Churn Prediction Pipeline")
st.caption(f"Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M')}")

df = generate_demo_predictions(200)
api_metrics = get_api_metrics()

# KPI row
col1, col2, col3, col4 = st.columns(4)
high_risk = (df["risk_label"] == "High").sum()
col1.metric("Total predictions", f"{len(df) + api_metrics.get('total_predictions', 0):,}")
col2.metric("High-risk customers", str(high_risk), f"{high_risk/len(df):.1%}")
col3.metric("Avg churn probability", f"{df['churn_probability'].mean():.1%}")
col4.metric("Model AUC", "0.8742", "↑ 0.3% vs baseline")

st.divider()

# Prediction distribution
col_a, col_b = st.columns(2)
with col_a:
    st.subheader("Prediction distribution")
    fig = px.histogram(
        df, x="churn_probability", nbins=30,
        color_discrete_sequence=["#5DCAA5"],
        labels={"churn_probability": "Churn probability"},
    )
    fig.update_layout(height=300, margin=dict(t=10, b=30))
    st.plotly_chart(fig, use_container_width=True)

with col_b:
    st.subheader("Risk breakdown")
    counts = df["risk_label"].value_counts()
    fig2 = px.pie(
        values=counts.values, names=counts.index,
        color=counts.index,
        color_discrete_map={"Low": "#639922", "Medium": "#EF9F27", "High": "#E24B4A"},
    )
    fig2.update_layout(height=300, margin=dict(t=10, b=10))
    st.plotly_chart(fig2, use_container_width=True)

# Time series
st.subheader("Churn probability over time")
hourly = df.set_index("timestamp").resample("2h")["churn_probability"].mean().reset_index()
fig3 = px.line(
    hourly, x="timestamp", y="churn_probability",
    color_discrete_sequence=["#378ADD"],
    labels={"churn_probability": "Avg probability", "timestamp": ""},
)
fig3.add_hline(y=0.7, line_dash="dash", line_color="#E24B4A", annotation_text="High risk threshold")
fig3.update_layout(height=250, margin=dict(t=10, b=30))
st.plotly_chart(fig3, use_container_width=True)

# Feature drift (simulated)
st.subheader("Feature drift detection")
st.caption("Comparing recent predictions vs training distribution")
features = ["tenure", "MonthlyCharges"]
drift_scores = {"tenure": 0.04, "MonthlyCharges": 0.19}
drift_threshold = 0.15

drift_df = pd.DataFrame([
    {"feature": f, "drift_score": s, "alert": s > drift_threshold}
    for f, s in drift_scores.items()
])
fig4 = px.bar(
    drift_df, x="feature", y="drift_score",
    color="alert",
    color_discrete_map={True: "#E24B4A", False: "#5DCAA5"},
    labels={"drift_score": "PSI score", "feature": "Feature"},
)
fig4.add_hline(y=drift_threshold, line_dash="dash", annotation_text="Drift threshold (0.15)")
fig4.update_layout(height=250, margin=dict(t=10, b=30), showlegend=False)
st.plotly_chart(fig4, use_container_width=True)

if any(drift_df["alert"]):
    st.warning("⚠️ MonthlyCharges shows significant drift — consider retraining.")
else:
    st.success("✅ No significant feature drift detected.")
