import streamlit as st


def display_sidebar_information():
    """Application sidebar with usage guidance"""
    st.sidebar.markdown("## 📋 Application Guide")
    st.sidebar.markdown(
        """
    ### Usage Instructions:
    1. Complete all input fields in sequential order
    2. Verify input values using expandable section
    3. Execute prediction using primary button
    4. Review results and recommendations
    5. Explore model analysis in dedicated tab
    
    ### Important Notes:
    - Input fields follow dataset column sequence
    - All fields are required for accurate prediction
    - Results are for educational purposes only
    """
    )

    st.sidebar.markdown("---")
    st.sidebar.markdown("### 📊 Input Sequence Reference")
    st.sidebar.markdown(
        """
    <div style="font-size: 0.85rem; line-height: 1.6;">
        <strong>Demographics (1-4):</strong><br>
        Gender → Age → Height → Weight<br><br>
        <strong>Lifestyle (5-8):</strong><br>
        Family History → FAVC → FCVC → NCP<br><br>
        <strong>Habits (9-12):</strong><br>
        CAEC → Smoking → Hydration → SCC<br><br>
        <strong>Activity (13-16):</strong><br>
        Physical Activity → Technology → Alcohol → Transport
    </div>
    """,
        unsafe_allow_html=True,
    )

    st.sidebar.markdown("---")
    st.sidebar.markdown("### ⚕️ Health Disclaimer")
    st.sidebar.markdown(
        """
    This application is designed for educational and research purposes. 
    Consult qualified healthcare professionals for medical advice and treatment decisions.
    """
    )

    st.sidebar.markdown("---")
    st.sidebar.markdown("### 📈 BMI Reference Scale")
    st.sidebar.markdown(
        """
    <div style="font-size: 0.9rem; line-height: 1.8;">
        <div style="color: #0D47A1; font-weight: bold;">• Underweight: < 18.5</div>
        <div style="color: #1B5E20; font-weight: bold;">• Normal: 18.5 - 24.9</div>
        <div style="color: #E65100; font-weight: bold;">• Overweight: 25.0 - 29.9</div>
        <div style="color: #B71C1C; font-weight: bold;">• Obese: ≥ 30.0</div>
    </div>
    """,
        unsafe_allow_html=True,
    )
