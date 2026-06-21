
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

    risk_score = 0
    reasons = []

    ratio = total_claim_amount / (policy_annual_premium + 1)

    # Claim vs Premium

    if ratio > 30:
        risk_score += 25
        reasons.append(
            "Very high claim-to-premium ratio"
        )

    elif ratio > 15:
        risk_score += 15
        reasons.append(
            "High claim-to-premium ratio"
        )

    # High claim amount

    if total_claim_amount > 50000:
        risk_score += 20
        reasons.append(
            "Unusually high claim amount"
        )

    # Police report

    if police_report == "NO":
        risk_score += 15
        reasons.append(
            "Police report not available"
        )

    # Severity

    if incident_severity == "Total Loss":
        risk_score += 15
        reasons.append(
            "Total vehicle loss reported"
        )

    elif incident_severity == "Major Damage":
        risk_score += 10
        reasons.append(
            "Major damage reported"
        )

    # Witnesses

    if witnesses == 0:
        risk_score += 10
        reasons.append(
            "No witnesses available"
        )

    # Vehicles involved

    if vehicles >= 3:
        risk_score += 10
        reasons.append(
            "Multiple vehicles involved"
        )

    # Bodily injuries

    if bodily_injuries > 2:
        risk_score += 5
        reasons.append(
            "High bodily injury count"
        )

    # Young customer

    if age < 25:
        risk_score += 5
        reasons.append(
            "Young policy holder"
        )

    risk_score = min(risk_score, 100)

    fraud_probability = risk_score

    confidence = 100 - abs(
        50 - risk_score
    )

    # =====================================
    # DECISION
    # =====================================

    if risk_score >= 70:

        decision = "🔴 HIGH FRAUD RISK"

    elif risk_score >= 40:

        decision = "🟡 MANUAL INVESTIGATION REQUIRED"

    else:

        decision = "🟢 LOW FRAUD RISK"

    # =====================================
    # RESULTS
    # =====================================

    st.markdown("---")

    c1, c2, c3 = st.columns(3)

    c1.metric(
        "Fraud Probability",
        f"{fraud_probability:.0f}%"
    )

    c2.metric(
        "Risk Score",
        f"{risk_score:.0f}/100"
    )

    c3.metric(
        "Claim/Premium Ratio",
        f"{ratio:.2f}"
    )

    st.progress(risk_score)

    st.subheader("Decision")

    if risk_score >= 70:
        st.error(decision)

    elif risk_score >= 40:
        st.warning(decision)

    else:
        st.success(decision)

    st.subheader("Risk Factors")

    if len(reasons) == 0:

        st.success(
            "No suspicious indicators detected."
        )

    else:

        for reason in reasons:

            st.write(f"• {reason}")

    st.subheader("AI Recommendation")

    if risk_score >= 70:

        st.error(
            "Recommend detailed fraud investigation before claim approval."
        )

    elif risk_score >= 40:

        st.warning(
            "Recommend manual review by insurance officer."
        )

    else:

        st.success(
            "Claim appears genuine and may proceed normally."
        )