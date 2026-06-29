def decide(score):

    if score >= 75:

        return (
            "🔴 HIGH FRAUD RISK",
            "Recommend detailed investigation."
        )

    elif score >= 40:

        return (
            "🟠 MEDIUM FRAUD RISK",
            "Manual verification recommended."
        )

    else:

        return (
            "🟢 LOW FRAUD RISK",
            "Claim appears genuine."
        )