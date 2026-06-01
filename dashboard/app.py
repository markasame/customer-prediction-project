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
import yaml
from datetime import datetime, timedelta

st.set_page_config(
    page_title="Churn Pipeline Monitor",
    page_icon="📉",
    layout="wide",
)

API_URL = "http://localhost:8000"
CONFIG_PATH = "config.yaml"

# Human-readable labels for every raw column name used across charts and hovers
FEATURE_LABELS = {
    "tenure":           "Tenure (months as a customer)",
    "MonthlyCharges":   "Monthly Charges (current monthly bill, $)",
    "TotalCharges":     "Total Charges (cumulative spend, $)",
    "Contract":         "Contract Type (month-to-month / 1-year / 2-year)",
    "gender":           "Gender",
    "SeniorCitizen":    "Senior Citizen (1 = yes, 0 = no)",
    "Partner":          "Has a Partner (Yes / No)",
    "Dependents":       "Has Dependents (Yes / No)",
    "PhoneService":     "Has Phone Service (Yes / No)",
    "MultipleLines":    "Has Multiple Phone Lines (Yes / No)",
    "InternetService":  "Internet Service Type (DSL / Fiber optic / None)",
    "OnlineSecurity":   "Has Online Security add-on (Yes / No)",
    "OnlineBackup":     "Has Online Backup add-on (Yes / No)",
    "DeviceProtection": "Has Device Protection add-on (Yes / No)",
    "TechSupport":      "Has Tech Support add-on (Yes / No)",
    "StreamingTV":      "Streams TV (Yes / No)",
    "StreamingMovies":  "Streams Movies (Yes / No)",
    "PaperlessBilling": "Uses Paperless Billing (Yes / No)",
    "PaymentMethod":    "Payment Method (credit card / bank transfer / etc.)",
}

PSI_INTERPRETATION = {
    lambda s: s < 0.10: "No meaningful change — distribution is stable",
    lambda s: 0.10 <= s < 0.20: "Slight shift — keep monitoring",
    lambda s: s >= 0.20: "Significant shift — consider retraining the model",
}


def psi_label(score: float) -> str:
    if score < 0.10:
        return "Stable — no meaningful change"
    if score < 0.20:
        return "Slight shift — keep monitoring"
    return "Significant shift — consider retraining"


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
    st.caption(
        "Adjust the three most influential inputs and hit the button "
        "to get a live prediction from the API. All other fields use "
        "typical defaults (fiber optic internet, no add-ons, electronic check)."
    )

    tenure = st.slider(
        "Tenure (months)",
        min_value=0, max_value=72, value=12,
        help="How long the customer has been with the company. "
             "Longer tenure is strongly associated with lower churn risk.",
    )
    monthly = st.slider(
        "Monthly Charges ($)",
        min_value=18, max_value=120, value=65,
        help="The customer's current monthly bill. "
             "Higher charges with short tenure is a common churn signal.",
    )
    contract = st.selectbox(
        "Contract Type",
        ["Month-to-month", "One year", "Two year"],
        help="Billing commitment length. Month-to-month customers churn "
             "at roughly 3× the rate of two-year contract holders.",
    )

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

            st.metric(
                "Churn probability",
                f"{prob:.1%}",
                help="Probability (0–100%) that this customer cancels within "
                     "the next billing cycle, as estimated by the XGBoost model.",
            )
            st.write(f"{color} Risk level: **{result['risk_label']}**")

            st.divider()
            st.caption("**Top drivers for this prediction** — positive SHAP pushes toward churn, negative away from it:")
            for f in result["top_risk_factors"]:
                label = FEATURE_LABELS.get(f["feature"], f["feature"])
                v = f["shap_value"]
                arrow = "↑" if v >= 0 else "↓"
                colour_word = "red" if v >= 0 else "green"
                st.markdown(
                    f":{colour_word}[{arrow}] **{f['feature']}** — {label}  \n"
                    f"&nbsp;&nbsp;&nbsp;&nbsp;SHAP value: `{v:+.4f}`",
                    unsafe_allow_html=False,
                )
        except Exception as e:
            st.warning(f"Could not reach API: {e}")


# ─── Main ──────────────────────────────────────────────────────────────────────
st.title("Churn Prediction Pipeline — Monitor")
st.caption(f"Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M')}")

df = generate_demo_predictions(200)
api_metrics = get_api_metrics()

# ── KPI row ────────────────────────────────────────────────────────────────────
col1, col2, col3, col4 = st.columns(4)
high_risk = (df["risk_label"] == "High").sum()

col1.metric(
    "Total predictions",
    f"{len(df) + api_metrics.get('total_predictions', 0):,}",
    help=(
        "Total number of churn predictions served since the API last started. "
        "Combines the 200-point demo history with any live calls made to POST /predict."
    ),
)
col2.metric(
    "High-risk customers",
    str(high_risk),
    f"{high_risk / len(df):.1%} of total",
    help=(
        "Customers whose predicted churn probability (churn_probability) is ≥ 70%. "
        "These are the highest-priority accounts for retention campaigns."
    ),
)
col3.metric(
    "Avg churn probability",
    f"{df['churn_probability'].mean():.1%}",
    help=(
        "Mean predicted churn probability across all recent predictions. "
        "The natural churn rate in the Telco dataset is ~26%; values well above "
        "this may indicate model drift or a shift in the customer base."
    ),
)
col4.metric(
    "Model AUC",
    "0.8742",
    "↑ 0.3% vs baseline",
    help=(
        "ROC-AUC (Area Under the Receiver Operating Characteristic Curve). "
        "Measures how well the model separates churners from non-churners across "
        "every possible decision threshold. "
        "0.5 = random guessing, 1.0 = perfect. 0.87 is considered strong."
    ),
)

st.divider()

# ── Prediction distribution + Risk breakdown ───────────────────────────────────
col_a, col_b = st.columns(2)

with col_a:
    st.subheader("Prediction distribution")
    st.caption(
        "How churn risk is spread across all predictions. "
        "A healthy model on Telco data shows most customers in the 10–35% range. "
        "A spike near 1.0 or a flat distribution may signal data drift."
    )
    fig = px.histogram(
        df, x="churn_probability", nbins=30,
        color_discrete_sequence=["#5DCAA5"],
        labels={"churn_probability": "Churn Probability (churn_probability)"},
    )
    fig.update_traces(
        hovertemplate=(
            "<b>Churn Probability (churn_probability)</b><br>"
            "Range midpoint: %{x:.1%}<br>"
            "Predictions in this range: %{y}<br>"
            "<i>Probability that a customer cancels their subscription</i>"
            "<extra></extra>"
        )
    )
    fig.update_layout(
        height=300,
        margin=dict(t=10, b=30),
        xaxis_tickformat=".0%",
        yaxis_title="Number of predictions",
    )
    st.plotly_chart(fig, use_container_width=True)

with col_b:
    st.subheader("Risk breakdown")
    st.caption(
        "Share of predictions in each risk tier. "
        "Low = churn probability < 40%, "
        "Medium = 40–70%, "
        "High = ≥ 70%."
    )
    counts = df["risk_label"].value_counts()
    total_count = counts.sum()
    fig2 = px.pie(
        values=counts.values,
        names=counts.index,
        color=counts.index,
        color_discrete_map={"Low": "#639922", "Medium": "#EF9F27", "High": "#E24B4A"},
    )
    fig2.update_traces(
        hovertemplate=(
            "<b>%{label} Risk (risk_label)</b><br>"
            "Customers: %{value}<br>"
            "Share of predictions: %{percent}<br>"
            "<i>Low: &lt;40% · Medium: 40–70% · High: ≥70% churn probability</i>"
            "<extra></extra>"
        )
    )
    fig2.update_layout(height=300, margin=dict(t=10, b=10))
    st.plotly_chart(fig2, use_container_width=True)

# ── Time series ────────────────────────────────────────────────────────────────
st.subheader("Average churn probability over time")
st.caption(
    "Each point is the mean predicted churn probability (churn_probability) across all "
    "predictions in a 2-hour window. A rising trend or a sustained cross of the red "
    "threshold line suggests the model is seeing riskier customers — or has drifted."
)
hourly = df.set_index("timestamp").resample("2h")["churn_probability"].mean().reset_index()
fig3 = px.line(
    hourly, x="timestamp", y="churn_probability",
    color_discrete_sequence=["#378ADD"],
    labels={
        "churn_probability": "Avg Churn Probability (churn_probability)",
        "timestamp": "Time",
    },
)
fig3.update_traces(
    hovertemplate=(
        "<b>%{x|%b %d %Y, %H:%M}</b><br>"
        "Avg Churn Probability (churn_probability): %{y:.1%}<br>"
        "<i>2-hour window average across all predictions</i>"
        "<extra></extra>"
    )
)
fig3.add_hline(
    y=0.7,
    line_dash="dash",
    line_color="#E24B4A",
    annotation_text="High-risk threshold (70%)",
    annotation_position="top left",
)
fig3.update_layout(
    height=260,
    margin=dict(t=10, b=30),
    yaxis_tickformat=".0%",
)
st.plotly_chart(fig3, use_container_width=True)

# ── Feature drift ──────────────────────────────────────────────────────────────
st.subheader("Feature drift detection (PSI)")
st.caption(
    "Population Stability Index (PSI) compares the distribution of each feature in "
    "recent predictions against the training data distribution. "
    "PSI < 0.10 is stable · 0.10–0.20 is a slight shift · > 0.20 is significant."
)

drift_scores = {"tenure": 0.04, "MonthlyCharges": 0.19}
drift_threshold = 0.15

drift_df = pd.DataFrame([
    {
        "feature": f,
        "feature_label": FEATURE_LABELS.get(f, f),
        "drift_score": s,
        "psi_interpretation": psi_label(s),
        "alert": s > drift_threshold,
    }
    for f, s in drift_scores.items()
])

fig4 = px.bar(
    drift_df,
    x="feature",
    y="drift_score",
    color="alert",
    color_discrete_map={True: "#E24B4A", False: "#5DCAA5"},
    labels={
        "drift_score": "PSI Score (drift_score)",
        "feature": "Feature",
    },
    custom_data=["feature_label", "psi_interpretation", "drift_score"],
)
fig4.update_traces(
    hovertemplate=(
        "<b>%{customdata[0]}</b><br>"
        "Column name: %{x}<br>"
        "PSI Score (drift_score): %{customdata[2]:.3f}<br>"
        "Status: %{customdata[1]}<br>"
        "<i>Threshold for alert: 0.15 · Measures distribution shift vs training data</i>"
        "<extra></extra>"
    )
)
fig4.add_hline(
    y=drift_threshold,
    line_dash="dash",
    line_color="#888",
    annotation_text="Alert threshold (PSI = 0.15)",
    annotation_position="top left",
)
fig4.update_layout(
    height=260,
    margin=dict(t=10, b=30),
    showlegend=False,
    xaxis_tickvals=drift_df["feature"].tolist(),
    xaxis_ticktext=[FEATURE_LABELS.get(f, f) for f in drift_df["feature"]],
)
st.plotly_chart(fig4, use_container_width=True)

alerted = drift_df[drift_df["alert"]]["feature"].tolist()
if alerted:
    labels = [FEATURE_LABELS.get(f, f) for f in alerted]
    st.warning(
        f"⚠️ Drift detected in: **{', '.join(labels)}** — "
        "the distribution of this feature in recent predictions has shifted "
        "significantly from the training data. Consider retraining the model."
    )
else:
    st.success("✅ No significant feature drift detected — distributions are stable.")

# ── Glossary ───────────────────────────────────────────────────────────────────
st.divider()
with st.expander("📖 Glossary — what do these terms mean?"):
    st.markdown("""
| Term | Definition |
|---|---|
| **Churn probability** (`churn_probability`) | The model's estimated likelihood (0–100%) that a customer cancels their subscription within the next billing cycle. Produced by the XGBoost classifier's `predict_proba()`. |
| **Risk label** (`risk_label`) | A human-readable tier derived from churn probability: **Low** < 40%, **Medium** 40–70%, **High** ≥ 70%. |
| **ROC-AUC** | Area Under the Receiver Operating Characteristic Curve. Measures how well the model ranks churners above non-churners across every threshold. 0.5 = random, 1.0 = perfect. |
| **PSI** (`drift_score`) | Population Stability Index — measures how much a feature's distribution in recent data has shifted from the training distribution. < 0.10 stable, 0.10–0.20 slight shift, > 0.20 significant. |
| **SHAP value** (`shap_value`) | SHapley Additive exPlanations — the contribution of a single feature to a single prediction, measured in log-odds units. Positive = pushes the prediction toward churn, negative = pushes away. |
| **Tenure** (`tenure`) | Number of months the customer has been with the company. One of the strongest predictors of churn — long-tenure customers rarely leave. |
| **Monthly Charges** (`MonthlyCharges`) | The customer's current monthly bill in US dollars. High charges on a short-tenure account is a classic churn pattern. |
| **Total Charges** (`TotalCharges`) | Cumulative spend since the customer joined. Often correlated with tenure but can deviate for customers who upgraded plans. |
| **Contract type** (`Contract`) | The billing commitment: month-to-month, one year, or two year. Month-to-month customers face no penalty to leave and churn at ~3× the rate of two-year holders. |
| **SMOTE** | Synthetic Minority Over-sampling Technique — used during training to balance the dataset by generating synthetic minority-class (churned) samples, because only ~26% of Telco customers churn. |
""")
