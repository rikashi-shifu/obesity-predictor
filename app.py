# app.py
import streamlit as st
from config.settings import Settings
from styles.css import get_custom_css
from models.loader import load_deployment_artifacts
from components.sidebar import display_sidebar_information
from pages.prediction import render_prediction_page
from pages.analytics import render_analytics_page


def configure_app():
    """Configure Streamlit app settings"""
    st.set_page_config(
        page_title=Settings.app.PAGE_TITLE,
        page_icon=Settings.app.PAGE_ICON,
        layout=Settings.app.LAYOUT,
        initial_sidebar_state=Settings.app.SIDEBAR_STATE,
    )

    # Apply custom CSS
    st.markdown(get_custom_css(), unsafe_allow_html=True)


def main():
    """Main application interface"""
    configure_app()
    display_sidebar_information()

    st.markdown(
        '<h1 class="main-header">🏥 Obesity Level Prediction System </h1>',
        unsafe_allow_html=True,
    )

    # Load deployment artifacts
    artifacts = load_deployment_artifacts()
    if artifacts is None:
        st.stop()

    # System information panel
    with st.expander("ℹ️ System Information", expanded=False):
        col1, col2 = st.columns(2)
        with col1:
            st.markdown(
                """
            <div class="info-box">
                <h4>🎯 Classification Categories</h4>
                <ul>
                    <li>Insufficient Weight</li>
                    <li>Normal Weight</li>
                    <li>Overweight Level I & II</li>
                    <li>Obesity Type I, II & III</li>
                </ul>
            </div>
            """,
                unsafe_allow_html=True,
            )
        with col2:
            if "best_model_name" in artifacts:
                st.markdown(
                    f"""
                <div class="info-box">
                    <h4>🤖 Model Specifications</h4>
                    <ul>
                        <li>Algorithm: {artifacts['best_model_name']}</li>
                        <li>Features: {len(artifacts['feature_names'])}</li>
                        <li>Status: Production Ready</li>
                        <li>Validation: Cross-Validated</li>
                    </ul>
                </div>
                """,
                    unsafe_allow_html=True,
                )

    # Main application interface
    tab1, tab2 = st.tabs(["🔮 Prediction Interface", "📊 Model Analytics"])

    with tab1:
        render_prediction_page(artifacts)

    with tab2:
        render_analytics_page(artifacts)


if __name__ == "__main__":
    main()
