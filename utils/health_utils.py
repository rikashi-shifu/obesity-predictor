def get_bmi_category_info(bmi):
    """BMI categorization with WHO standards"""
    if bmi < 18.5:
        return "Underweight", "#0D47A1", "#E3F2FD", "Below normal weight range"
    elif 18.5 <= bmi < 25:
        return "Normal Weight", "#1B5E20", "#E8F5E8", "Healthy weight range"
    elif 25 <= bmi < 30:
        return "Overweight", "#E65100", "#FFF3E0", "Above normal weight range"
    elif 30 <= bmi < 35:
        return "Obesity Class I", "#B71C1C", "#FFEBEE", "Mild obesity"
    elif 35 <= bmi < 40:
        return "Obesity Class II", "#880E4F", "#FCE4EC", "Moderate obesity"
    else:
        return "Obesity Class III", "#4A148C", "#F3E5F5", "Severe obesity"


def generate_health_recommendations(user_input, bmi, prediction):
    """Personalized health recommendation system"""
    recommendations = []

    if bmi > 30:
        recommendations.append(
            "Consult healthcare provider for comprehensive obesity management plan"
        )
    elif bmi > 25:
        recommendations.append(
            "Consider implementing structured weight management strategies"
        )
    elif bmi < 18.5:
        recommendations.append(
            "Consult healthcare provider for healthy weight gain guidance"
        )

    if user_input["faf"] < 1:
        recommendations.append(
            "Increase physical activity to minimum 150 minutes moderate exercise weekly"
        )
    if user_input["fcvc"] < 2:
        recommendations.append(
            "Incorporate more vegetables and fruits in daily dietary intake"
        )
    if user_input["ch2o"] < 2:
        recommendations.append("Increase daily water consumption to minimum 2-3 liters")
    if user_input["favc"] == "yes":
        recommendations.append(
            "Reduce frequent consumption of high-caloric processed foods"
        )
    if user_input["smoke"] == "yes":
        recommendations.append("Consider enrolling in smoking cessation programs")
    if user_input["tue"] > 1:
        recommendations.append(
            "Limit screen time and increase active lifestyle engagement"
        )
    if user_input["mtrans"] == "Automobile":
        recommendations.append(
            "Consider active transportation methods for short distances"
        )

    return recommendations
