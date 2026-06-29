from agents.validation_agent import validate
from agents.fraud_agent import predict
from agents.risk_agent import calculate
from agents.decision_agent import decide

import streamlit as st

# =====================================
# PAGE CONFIG
# =====================================

st.set_page_config(
    page_title="Insurance Fraud Detection",
    page_icon="🚨",
    layout="wide"
)

st.title("🚨 Insurance Fraud Detection System")
st.subheader("Agentic AI Risk Assessment Dashboard")

st.markdown("---")

# =====================================
# USER INPUTS
# =====================================

col1, col2 = st.columns(2)

with col1:

    age = st.number_input(
        "Age",
        min_value=18,
        max_value=100,
        value=35
    )

    policy_annual_premium = st.number_input(
        "Policy Annual Premium ($)",
        min_value=0.0,
        value=1200.0
    )

    total_claim_amount = st.number_input(
        "Total Claim Amount ($)",
        min_value=0.0,
        value=5000.0
    )

    injury_claim = st.number_input(
        "Injury Claim ($)",
        min_value=0.0,
        value=1000.0
    )

    property_claim = st.number_input(
        "Property Claim ($)",
        min_value=0.0,
        value=1000.0
    )

with col2:

    vehicle_claim = st.number_input(
        "Vehicle Claim ($)",
        min_value=0.0,
        value=3000.0
    )

    incident_severity = st.selectbox(
        "Incident Severity",
        [
            "Trivial Damage",
            "Minor Damage",
            "Major Damage",
            "Total Loss"
        ]
    )

    police_report = st.selectbox(
        "Police Report Available",
        [
            "YES",
            "NO"
        ]
    )

    witnesses = st.slider(
        "Number of Witnesses",
        0,
        5,
        1
    )

    vehicles = st.slider(
        "Vehicles Involved",
        1,
        5,
        1
    )

bodily_injuries = st.slider(
    "Bodily Injuries",
    0,
    5,
    0
)

st.markdown("---")

# =====================================
# FRAUD DETECTION
# =====================================

if st.button("🔍 Detect Fraud", use_container_width=True):

    # ============================
    # Collect User Data
    # ============================

    user_data = {
        "age": age,
        "policy_annual_premium": policy_annual_premium,
        "total_claim_amount": total_claim_amount,
        "injury_claim": injury_claim,
        "property_claim": property_claim,
        "vehicle_claim": vehicle_claim,
        "incident_severity": incident_severity,
        "police_report_available": police_report,
        "witnesses": witnesses,
        "number_of_vehicles_involved": vehicles,
        "bodily_injuries": bodily_injuries
    }

    # ============================
    # Agent 1 : Validation Agent
    # ============================

    errors = validate(user_data)

    if errors:

        st.error("Input Validation Failed")

        for e in errors:
            st.write("•", e)

        st.stop()

    # ============================
    # Features for DNN
    # ============================

        # ============================
    # Encode Categorical Inputs
    # ============================

    severity_map = {
        "Trivial Damage": 0,
        "Minor Damage": 1,
        "Major Damage": 2,
        "Total Loss": 3
    }

    police_map = {
        "YES": 1,
        "NO": 0
    }

    # ============================
    # Features for DNN
    # ============================

    features = [

        age,

        policy_annual_premium,

        total_claim_amount,

        injury_claim,

        property_claim,

        vehicle_claim,

        severity_map[incident_severity],

        police_map[police_report],

        witnesses,

        vehicles,

        bodily_injuries

    ]

    # ============================
    # Agent 2 : Fraud Agent
    # ============================

    fraud_probability, anomaly_score = predict(features)

    # ============================
    # Agent 3 : Risk Agent
    # ============================

    risk_score, reasons = calculate(
        user_data,
        fraud_probability,
        anomaly_score
    )

    # ============================
    # Agent 4 : Decision Agent
    # ============================

    decision, recommendation = decide(risk_score)

    # ============================
    # Results
    # ============================

    st.markdown("---")

    c1, c2, c3 = st.columns(3)

    c1.metric(
    "Fraud Probability",
    f"{fraud_probability*100:.1f}%"
)
    c2.metric(
    "Risk Score",
    f"{risk_score:.0f}/100"
)

    c3.metric(
    "Anomaly Score",
    f"{anomaly_score:.3f}"
)

    st.progress(int(risk_score))

    st.markdown("### 🤖 Agent Execution Status")

    st.success("✅ Validation Agent Completed")

    st.success("✅ Fraud Detection Agent Completed")

    st.success("✅ Risk Assessment Agent Completed")

    st.success("✅ Decision Agent Completed")

    st.progress(int(risk_score))

    if risk_score >= 75:
        st.error(decision)

    elif risk_score >= 40:
        st.warning(decision)

    else:
        st.success(decision)

    st.subheader("Risk Factors")

    if reasons:

        for r in reasons:
            st.write("•", r)

    else:

        st.success("No suspicious indicators detected.")

    st.subheader("AI Recommendation")

    st.info(recommendation)