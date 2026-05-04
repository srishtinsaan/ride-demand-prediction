def calculate_surge(predicted_demand, avg_demand):
    if avg_demand == 0:
        return 1.0

    ratio = predicted_demand / avg_demand

    if ratio < 1:
        return 1.0
    elif ratio < 1.5:
        return 1.2
    elif ratio < 2:
        return 1.5
    else:
        return 2.0