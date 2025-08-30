import streamlit as st
from models.predictor import predict_obesity_level
from utils.health_utils import get_bmi_category_info, generate_health_recommendations
from utils.visualization import create_probability_visualization


def render_prediction_page(artifacts):
    """Render the prediction interface page"""
    col1, col2 = st.columns([1, 1])

    with col1:
        st.markdown(
            '<h2 class="sub-header">📝 Patient Data Input</h2>', unsafe_allow_html=True
        )

        # Demographic information section
        st.markdown("**Demographic Information (Fields 1-4)**")
        gender = st.selectbox("1. Gender", ["Male", "Female"])
        age = st.slider("2. Age (years)", 14, 80, 25)
        # Height input
        height_str = st.text_input("3. Height (meters)", "1.65")
        # Weight input
        weight_str = st.text_input("4. Weight (kilograms)", "65.0")

        # Initialize validation flags
        height_valid = False
        weight_valid = False

        # Try converting safely to floats
        try:
            height = float(height_str)
            # enforce range [1.20, 2.20]
            if not (1.20 <= height <= 2.20):
                st.warning("Height must be between 1.20 and 2.20 meters")
            else:
                height_valid = True
        except ValueError:
            st.warning("Please enter a valid number for height")

        try:
            weight = float(weight_str)
            # enforce range [30.0, 200.0]
            if not (30.0 <= weight <= 200.0):
                st.warning("Weight must be between 30.0 and 200.0 kilograms")
            else:
                weight_valid = True
        except ValueError:
            st.warning("Please enter a valid number for weight")

        # Dietary and family history section
        st.markdown("**Health History & Dietary Habits (Fields 5-8)**")
        family_history = st.selectbox(
            "5. Family History with Overweight", ["no", "yes"]
        )
        favc = st.selectbox(
            "6. FAVC (Frequent high caloric food consumption)", ["no", "yes"]
        )
        fcvc = st.slider("7. FCVC (Vegetable consumption frequency)", 1, 3, 2)
        ncp = st.slider("8. NCP (Number of main meals daily)", 1.0, 4.0, 3.0, 0.5)

        # Lifestyle and behavioral patterns section
        st.markdown("**Lifestyle & Behavioral Patterns (Fields 9-12)**")
        caec = st.selectbox(
            "9. CAEC (Consumption between meals)",
            ["no", "Sometimes", "Frequently", "Always"],
        )
        smoke = st.selectbox("10. SMOKE (Smoking habit)", ["no", "yes"])
        ch2o = st.slider("11. CH2O (Daily water intake in liters)", 1.0, 3.0, 2.0, 0.1)
        scc = st.selectbox("12. SCC (Calorie consumption monitoring)", ["no", "yes"])

        # Activity and transportation section
        st.markdown("**Physical Activity & Transportation (Fields 13-16)**")
        faf = st.slider(
            "13. FAF (Physical activity frequency per week)", 0.0, 3.0, 1.0, 0.1
        )
        tue = st.slider("14. TUE (Technology device usage hours)", 0, 2, 1)
        calc = st.selectbox(
            "15. CALC (Alcohol consumption frequency)",
            ["no", "Sometimes", "Frequently", "Always"],
        )
        mtrans = st.selectbox(
            "16. MTRANS (Primary transportation mode)",
            ["Public_Transportation", "Walking", "Automobile", "Bike", "Motorbike"],
        )

    with col2:
        st.markdown(
            '<h2 class="sub-header">🎯 Prediction Results</h2>', unsafe_allow_html=True
        )

        # Compile user input data
        user_input = {
            "gender": gender,
            "age": age,
            "height": height if height_valid else None,
            "weight": weight if weight_valid else None,
            "family_history": family_history,
            "favc": favc,
            "fcvc": fcvc,
            "ncp": ncp,
            "caec": caec,
            "smoke": smoke,
            "ch2o": ch2o,
            "scc": scc,
            "faf": faf,
            "tue": tue,
            "calc": calc,
            "mtrans": mtrans,
        }

        # Primary prediction execution - disabled if inputs are invalid
        if st.button(
            "🔮 Execute Prediction Analysis",
            type="primary",
            use_container_width=True,
            disabled=not (height_valid and weight_valid),
        ):
            with st.spinner("Processing machine learning analysis..."):
                result = predict_obesity_level(artifacts, user_input)

                if result is None:
                    st.error(
                        "Prediction analysis failed. Please verify input data and retry."
                    )
                    st.stop()

                # Display results using helper functions
                _display_bmi_analysis(result)
                _display_prediction_result(result)
                _display_probability_distribution(result)
                _display_model_info(result, artifacts)
                _display_recommendations(user_input, result)


def _display_bmi_analysis(result):
    """Display BMI analysis section"""
    bmi_category, bmi_color, bmi_bg, bmi_desc = get_bmi_category_info(result["bmi"])

    st.markdown(
        f"""
    <div style="background-color: {bmi_bg}; padding: 1.5rem; border-radius: 0.7rem; border: 2px solid {bmi_color}; margin: 1rem 0;">
        <h4 style="color: {bmi_color}; margin-bottom: 1rem; font-weight: bold;">📊 Body Mass Index Analysis</h4>
        <div style="font-size: 2rem; font-weight: bold; color: {bmi_color}; margin: 0.5rem 0;">BMI: {result['bmi']:.2f}</div>
        <div style="color: {bmi_color}; font-weight: bold; font-size: 1.3rem; margin: 0.5rem 0;">{bmi_category}</div>
        <div style="color: {bmi_color}; font-style: italic; margin: 0.5rem 0;">{bmi_desc}</div>
    </div>
    """,
        unsafe_allow_html=True,
    )


def _display_prediction_result(result):
    """Display main prediction result"""
    color_map = {
        "Insufficient Weight": "#0D47A1",
        "Normal Weight": "#1B5E20",
        "Overweight Level I": "#E65100",
        "Overweight Level Ii": "#E65100",
        "Obesity Type I": "#B71C1C",
        "Obesity Type Ii": "#880E4F",
        "Obesity Type Iii": "#4A148C",
    }

    prediction_color = color_map.get(result["prediction"], "#1f77b4")

    st.markdown(
        f"""
    <div class="prediction-result" style="background-color: {prediction_color}20; border: 2px solid {prediction_color};">
        <h3 style="color: {prediction_color};">Predicted Obesity Classification</h3>
        <h2 style="color: {prediction_color};">{result['prediction']}</h2>
        <p style="color: {prediction_color};">Model Confidence: {result['confidence']:.1%}</p>
    </div>
    """,
        unsafe_allow_html=True,
    )


def _display_probability_distribution(result):
    """Display probability distribution chart"""
    st.subheader("📈 Classification Probability Distribution")
    fig = create_probability_visualization(result["probabilities"])
    st.plotly_chart(fig, use_container_width=True)


def _display_model_info(result, artifacts):
    """Display model performance information"""
    st.subheader("🤖 Model Performance Information")
    col_m1, col_m2 = st.columns(2)

    with col_m1:
        st.markdown(
            f"""
        <div class="info-box">
            <h4>Algorithm Specifications</h4>
            <p><strong>Model Type:</strong> {result['model_name']}</p>
            <p><strong>Prediction Confidence:</strong> {result['confidence']:.1%}</p>
            <p><strong>Features Analyzed:</strong> {len(artifacts['feature_names'])}</p>
        </div>
        """,
            unsafe_allow_html=True,
        )

    with col_m2:
        st.markdown(
            """
        <div class="success-box">
            <h4>✅ System Status</h4>
            <p>✓ Model: Operational</p>
            <p>✓ Features: Processed</p>
            <p>✓ Analysis: Complete</p>
            <p>✓ Validation: Successful</p>
        </div>
        """,
            unsafe_allow_html=True,
        )


def _display_recommendations(user_input, result):
    """Display health recommendations"""
    st.subheader("💡 Personalized Health Recommendations")
    recommendations = generate_health_recommendations(
        user_input, result["bmi"], result["prediction"]
    )

    if recommendations:
        rec_html = "<br>".join([f"• {rec}" for rec in recommendations])
        st.markdown(
            f"""
        <div class="warning-box">
            <h4>Health Optimization Recommendations</h4>
            <div style="line-height: 1.8;">{rec_html}</div>
        </div>
        """,
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            """
        <div class="success-box">
            <h4>✅ Optimal Health Profile</h4>
            <p>Current lifestyle choices align with healthy weight management practices.</p>
        </div>
        """,
            unsafe_allow_html=True,
        )
