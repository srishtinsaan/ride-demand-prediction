def calculate_surge(predicted_demand, avg_demand):
    # 1. Guard against bad inputs
    if avg_demand is None or avg_demand == 0:
        print("⚠️  avg_demand is 0 or None — returning no surge")
        return 1.0

    if predicted_demand is None or predicted_demand < 0:
        print("⚠️  Invalid predicted_demand — returning no surge")
        return 1.0

    # 2. Calculate demand ratio
    ratio = predicted_demand / avg_demand

    # 3. Apply surge tiers
    if ratio < 1.0:
        surge = 1.0
        label = "🟢 Normal"
    elif ratio < 1.5:
        surge = 1.2
        label = "🟡 Moderate"
    elif ratio < 2.0:
        surge = 1.5
        label = "🟠 High"
    elif ratio < 3.0:
        surge = 1.8
        label = "🔴 Very High"
    else:
        surge = 2.0
        label = "🚨 Extreme"

    print(f"📊 Demand ratio: {ratio:.2f} → Surge: {surge}x ({label})")

    return surge, label, round(ratio, 2)