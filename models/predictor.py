import pandas as pd
import streamlit as st


def engineer_prediction_features(user_input, selected_features):
    """Feature engineering pipeline for prediction input"""
    data = pd.DataFrame([user_input])

    # Derived numerical features
    data["BMI"] = data["Weight"] / (data["Height"] ** 2)
    data["Weight_Height_Ratio"] = data["Weight"] / data["Height"]
    data["Physical_Activity_Score"] = data["FAF"] * data["CH2O"]
    data["Age_Weight_Interaction"] = data["Age"] * data["Weight"] / 100
    data["Lifestyle_Risk_Score"] = (
        (data["FAVC"] == 1).astype(int) * 2
        + (data["SMOKE"] == 1).astype(int) * 3
        + (data["SCC"] == 0).astype(int) * 1
    )
    data["Health_Risk_Index"] = (
        (data["FAVC"] == 1).astype(int) * 0.3
        + (data["SMOKE"] == 1).astype(int) * 0.4
        + (data["FAF"] < 1).astype(int) * 0.3
    )
    data["Eating_Pattern_Score"] = (
        data["FCVC"] * data["NCP"] / data["TUE"].clip(lower=1)
    )

    # Categorical feature binning
    age = user_input["Age"]
    if age <= 25:
        age_cat = "Young"
    elif age <= 35:
        age_cat = "Adult"
    elif age <= 50:
        age_cat = "MiddleAge"
    else:
        age_cat = "Senior"

    bmi = data["BMI"].iloc[0]
    if bmi < 18.5:
        bmi_cat = "Underweight"
    elif bmi < 25:
        bmi_cat = "Normal"
    elif bmi < 30:
        bmi_cat = "Overweight"
    elif bmi < 35:
        bmi_cat = "Obese_I"
    else:
        bmi_cat = "Obese_II"

    ch2o = user_input["CH2O"]
    if ch2o <= 1.5:
        hydration_cat = "Low"
    elif ch2o <= 2.5:
        hydration_cat = "Moderate"
    else:
        hydration_cat = "High"

    faf = user_input["FAF"]
    if faf < 1:
        activity_cat = "Sedentary"
    elif faf < 2:
        activity_cat = "Light"
    else:
        activity_cat = "Active"

    # Categorical feature assignment
    data["CAEC"] = user_input["CAEC"]
    data["CALC"] = user_input["CALC"]
    data["MTRANS"] = user_input["MTRANS"]
    data["Age_Category"] = age_cat
    data["BMI_WHO_Category"] = bmi_cat
    data["Hydration_Level"] = hydration_cat
    data["Activity_Level"] = activity_cat

    # One-hot encoding for categorical variables
    all_categories = {
        "CAEC": ["no", "Sometimes", "Frequently", "Always"],
        "CALC": ["no", "Sometimes", "Frequently", "Always"],
        "MTRANS": [
            "Public_Transportation",
            "Walking",
            "Automobile",
            "Bike",
            "Motorbike",
        ],
        "Age_Category": ["Young", "Adult", "MiddleAge", "Senior"],
        "BMI_WHO_Category": [
            "Underweight",
            "Normal",
            "Overweight",
            "Obese_I",
            "Obese_II",
        ],
        "Hydration_Level": ["Low", "Moderate", "High"],
        "Activity_Level": ["Sedentary", "Light", "Active"],
    }

    # One-hot encoding implementation (drop_first=True)
    for feature, categories in all_categories.items():
        current_value = data[feature].iloc[0]
        for category in categories[1:]:
            column_name = f"{feature}_{category}"
            data[column_name] = 1 if current_value == category else 0

    # Remove original categorical columns
    data = data.drop(
        columns=[
            "CAEC",
            "CALC",
            "MTRANS",
            "Age_Category",
            "BMI_WHO_Category",
            "Hydration_Level",
            "Activity_Level",
        ]
    )

    # Ensure all selected features are present
    for feature in selected_features:
        if feature not in data.columns:
            data[feature] = 0

    return data[selected_features]


def predict_obesity_level(artifacts, user_input):
    """Machine learning prediction pipeline"""
    try:
        # Input preprocessing
        processed_input = {
            "Age": user_input["age"],
            "Height": user_input["height"],
            "Weight": user_input["weight"],
            "Gender": 1 if user_input["gender"] == "Male" else 0,
            "family_history_with_overweight": (
                1 if user_input["family_history"] == "yes" else 0
            ),
            "FAVC": 1 if user_input["favc"] == "yes" else 0,
            "FCVC": user_input["fcvc"],
            "NCP": user_input["ncp"],
            "CAEC": user_input["caec"],
            "SMOKE": 1 if user_input["smoke"] == "yes" else 0,
            "CH2O": user_input["ch2o"],
            "SCC": 1 if user_input["scc"] == "yes" else 0,
            "FAF": user_input["faf"],
            "TUE": user_input["tue"],
            "CALC": user_input["calc"],
            "MTRANS": user_input["mtrans"],
        }

        # Feature engineering
        feature_data = engineer_prediction_features(
            processed_input, artifacts["feature_names"]
        )

        # Feature scaling for continuous variables
        continuous_features = artifacts["continuous_features"]
        existing_continuous = [
            col for col in continuous_features if col in feature_data.columns
        ]

        if existing_continuous:
            feature_data_scaled = feature_data.copy()
            feature_data_scaled[existing_continuous] = artifacts["scaler"].transform(
                feature_data[existing_continuous]
            )
        else:
            feature_data_scaled = feature_data

        # Model prediction
        prediction = artifacts["model"].predict(feature_data_scaled)[0]
        probabilities = artifacts["model"].predict_proba(feature_data_scaled)[0]

        # Probability distribution mapping
        all_probabilities = {}
        for i, prob in enumerate(probabilities):
            class_name = artifacts["target_mapping"][i].replace("_", " ").title()
            all_probabilities[class_name] = prob

        predicted_class = (
            artifacts["target_mapping"][prediction].replace("_", " ").title()
        )
        confidence = probabilities.max()
        bmi = processed_input["Weight"] / (processed_input["Height"] ** 2)

        return {
            "prediction": predicted_class,
            "confidence": confidence,
            "probabilities": all_probabilities,
            "bmi": bmi,
            "model_name": (
                artifacts["best_model_name"]
                if "best_model_name" in artifacts
                else type(artifacts["model"]).__name__
            ),
        }

    except Exception as e:
        st.error(f"Prediction error: {str(e)}")
        return None
