def calculate(user_data, fraud_probability, anomaly_score):

    risk_score = 0
    reasons = []

    # DNN Contribution (0-70)
    
    risk_score += fraud_probability * 70

    if fraud_probability >= 0.80:
        reasons.append("Very high fraud probability predicted by DNN.")

    elif fraud_probability >= 0.60:
        reasons.append("High fraud probability predicted by DNN.")

    elif fraud_probability >= 0.40:
        reasons.append("Moderate fraud probability predicted by DNN.")

    # Autoencoder (0-15)
    
    if anomaly_score >= 1.5:
        risk_score += 15
        reasons.append("Highly anomalous claim pattern.")

    elif anomaly_score >= 0.7:
        risk_score += 10
        reasons.append("Moderately anomalous claim pattern.")

    elif anomaly_score >= 0.3:
        risk_score += 5
        reasons.append("Slight anomaly detected.")

    # ==========================
    # Rule-Based Checks (0-15)
    # ==========================

    if user_data["total_claim_amount"] > 50000:
        risk_score += 5
        reasons.append("Very high claim amount.")

    if user_data["police_report_available"] == "NO":
        risk_score += 4
        reasons.append("Police report unavailable.")

    if user_data["number_of_vehicles_involved"] >= 3:
        risk_score += 3
        reasons.append("Multiple vehicles involved.")

    if user_data["witnesses"] == 0:
        risk_score += 3
        reasons.append("No witnesses available.")

    risk_score = min(100, round(risk_score))

    return risk_score, reasons