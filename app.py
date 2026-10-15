import streamlit as st
import pandas as pd
import joblib
from pathlib import Path

# -----------------------------
# Page configuration
# -----------------------------
st.set_page_config(
    page_title="Bank Marketing AI",
    page_icon="🏦",
    layout="wide"
)

MODEL_PATH = Path(__file__).parent / "xgboost_bank_marketing.pkl"
DATA_PATH = (
    Path(__file__).parent
    / "Data"
    / "bank-additional"
    / "bank-additional-full.csv"
)

# -----------------------------
# Load model and dataset
# -----------------------------
@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)

@st.cache_data
def load_data():
    return pd.read_csv(DATA_PATH, sep=";").drop_duplicates()

try:
    model = load_model()
except Exception as e:
    st.error(f"Could not load the trained model: {e}")
    st.stop()

try:
    df = load_data()
except Exception:
    df = None

# -----------------------------
# Dashboard styling
# -----------------------------
st.markdown("""
<style>
    .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
    }
    .hero {
        padding: 1.5rem;
        border-radius: 16px;
        background: linear-gradient(120deg, #102A43, #176B87);
        color: white;
        margin-bottom: 1.2rem;
    }
    .hero h1 {
        color: white;
        margin-bottom: 0.4rem;
    }
    div[data-testid="stMetric"] {
        background: #F2F7FA;
        padding: 15px;
        border-radius: 12px;
        border: 1px solid #D8E5EC;
    }
</style>
""", unsafe_allow_html=True)

# -----------------------------
# Header
# -----------------------------
st.markdown("""
<div class="hero">
    <h1>🏦 Bank Marketing AI</h1>
    <p>Customer subscription prediction and campaign analytics</p>
</div>
""", unsafe_allow_html=True)

st.caption(
    "Machine learning capstone project | Portuguese Bank Marketing dataset"
)

# -----------------------------
# Navigation
# -----------------------------
dashboard_tab, prediction_tab, model_tab = st.tabs([
    "📊 Dashboard",
    "🎯 Customer Prediction",
    "🤖 Model Performance"
])

# -----------------------------
# Dashboard
# -----------------------------
with dashboard_tab:
    st.subheader("Campaign Overview")

    if df is None:
        st.warning(
            "Dataset not found. Check that the Data folder is "
            "inside the project directory."
        )
    else:
        total_customers = len(df)
        yes_count = (df["y"] == "yes").sum()
        no_count = (df["y"] == "no").sum()
        subscription_rate = yes_count / total_customers * 100

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Total Records", f"{total_customers:,}")
        c2.metric("Subscribed", f"{yes_count:,}")
        c3.metric("Not Subscribed", f"{no_count:,}")
        c4.metric("Subscription Rate", f"{subscription_rate:.2f}%")

        left, right = st.columns(2)

        with left:
            st.markdown("#### Subscription Distribution")
            distribution = (
                df["y"].value_counts()
                .reindex(["yes", "no"], fill_value=0)
            )
            distribution.index = ["Subscribed", "Not Subscribed"]
            st.bar_chart(distribution)

        with right:
            st.markdown("#### Subscription Rate by Contact Method")
            contact_rates = (
                df.groupby("contact", observed=True)["y"]
                .apply(lambda s: (s == "yes").mean() * 100)
                .sort_values(ascending=False)
            )
            st.bar_chart(contact_rates)

        st.markdown("#### Monthly Campaign Performance")
        monthly_rates = (
            df.groupby("month", observed=True)["y"]
            .apply(lambda s: (s == "yes").mean() * 100)
            .reindex([
                "jan", "feb", "mar", "apr", "may", "jun",
                "jul", "aug", "sep", "oct", "nov", "dec"
            ])
            .dropna()
        )
        st.bar_chart(monthly_rates)

# -----------------------------
# Customer Prediction
# -----------------------------
with prediction_tab:
    st.subheader("Predict Term Deposit Subscription")
    st.write(
        "Enter customer and campaign details to estimate the "
        "probability of subscription."
    )

    with st.form("prediction_form"):
        col1, col2, col3 = st.columns(3)

        with col1:
            age = st.number_input(
                "Age", min_value=17, max_value=98, value=35
            )
            job = st.selectbox("Job", [
                "admin.", "blue-collar", "entrepreneur", "housemaid",
                "management", "retired", "self-employed", "services",
                "student", "technician", "unemployed", "unknown"
            ])
            marital = st.selectbox("Marital Status", [
                "married", "single", "divorced", "unknown"
            ])
            education = st.selectbox("Education", [
                "university.degree", "high.school", "basic.9y",
                "professional.course", "basic.4y", "basic.6y",
                "illiterate", "unknown"
            ])
            default = st.selectbox(
                "Has Credit in Default?", ["no", "yes", "unknown"]
            )

        with col2:
            housing = st.selectbox(
                "Housing Loan", ["no", "yes", "unknown"]
            )
            loan = st.selectbox(
                "Personal Loan", ["no", "yes", "unknown"]
            )
            contact = st.selectbox(
                "Contact Method", ["cellular", "telephone"]
            )
            month = st.selectbox("Contact Month", [
                "may", "jun", "jul", "aug", "sep", "oct",
                "nov", "dec", "mar", "apr"
            ])
            day_of_week = st.selectbox(
                "Day of Week", ["mon", "tue", "wed", "thu", "fri"]
            )

        with col3:
            campaign = st.number_input(
                "Contacts in Current Campaign",
                min_value=1, max_value=56, value=1
            )
            pdays = st.number_input(
                "Days Since Previous Contact (999 = never contacted)",
                min_value=0, max_value=999, value=999
            )
            previous = st.number_input(
                "Previous Contacts",
                min_value=0, max_value=7, value=0
            )
            poutcome = st.selectbox(
                "Previous Campaign Outcome",
                ["nonexistent", "failure", "success"]
            )
            emp_var_rate = st.number_input(
                "Employment Variation Rate", value=0.0, format="%.3f"
            )

        with col1:
            cons_price_idx = st.number_input(
                "Consumer Price Index", value=93.5, format="%.3f"
            )
            cons_conf_idx = st.number_input(
                "Consumer Confidence Index", value=-40.0, format="%.3f"
            )

        with col2:
            euribor3m = st.number_input(
                "Euribor 3-Month Rate", value=3.5, format="%.3f"
            )
            nr_employed = st.number_input(
                "Number of Employees Indicator",
                value=5100.0, format="%.3f"
            )

        submitted = st.form_submit_button(
            "Predict Subscription", type="primary", use_container_width=True
        )

    if submitted:
        customer = pd.DataFrame([{
            "age": age,
            "job": job,
            "marital": marital,
            "education": education,
            "default": default,
            "housing": housing,
            "loan": loan,
            "contact": contact,
            "month": month,
            "day_of_week": day_of_week,
            "campaign": campaign,
            "pdays": pdays,
            "previous": previous,
            "poutcome": poutcome,
            "emp.var.rate": emp_var_rate,
            "cons.price.idx": cons_price_idx,
            "cons.conf.idx": cons_conf_idx,
            "euribor3m": euribor3m,
            "nr.employed": nr_employed
        }])

        try:
            probability = model.predict_proba(customer)[0, 1]
            prediction = int(model.predict(customer)[0])

            st.markdown("### Prediction Result")
            r1, r2 = st.columns(2)
            r1.metric("Estimated Subscription Probability",
                      f"{probability:.1%}")
            r2.metric("Model Prediction",
                      "Likely to Subscribe" if prediction == 1
                      else "Unlikely to Subscribe")

            st.progress(float(probability))

            st.info(
                "This is a model estimate, not a guarantee. "
                "Validate predictions before using them for real campaigns."
            )
        except Exception as e:
            st.error(f"Prediction failed: {e}")

# -----------------------------
# Model Performance
# -----------------------------
with model_tab:
    st.subheader("Tuned XGBoost Model")
    st.write(
        "Performance from the current test-set evaluation in the notebook."
    )

    c1, c2, c3 = st.columns(3)
    c1.metric("Accuracy", "90.00%")
    c2.metric("ROC-AUC", "81.53%")
    c3.metric("Recall", "24.46%")

    st.warning(
        "The model misses many actual subscribers at its default threshold. "
        "These metrics are from the current evaluation and are not a "
        "guarantee of future performance."
    )

st.divider()
st.caption("Bank Marketing AI | Academic capstone demonstration")
