def validate(data):

    errors = []

    if data["age"] < 18:
        errors.append("Invalid age")

    if data["policy_annual_premium"] <= 0:
        errors.append("Premium must be greater than 0")

    if data["total_claim_amount"] < 0:
        errors.append("Invalid claim amount")

    if data["injury_claim"] + data["property_claim"] + data["vehicle_claim"] > data["total_claim_amount"]:
        errors.append("Claim breakup exceeds total claim amount")

    return errors