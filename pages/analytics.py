import streamlit as st
import pandas as pd
import plotly.express as px


def render_analytics_page(artifacts):
    """Render the model analytics page"""
    st.markdown(
        '<h2 class="sub-header">📊 Advanced Model Analytics</h2>',
        unsafe_allow_html=True,
    )
    display_model_performance_analysis(artifacts)

    # Technical implementation information
    col1, col2 = st.columns(2)

    with col1:
        st.markdown(
            """
        <div class="info-box">
            <h4>🔬 Technical Implementation</h4>
            <ul>
                <li><strong>Random Forest:</strong> Ensemble decision trees</li>
                <li><strong>XGBoost:</strong> Gradient boosting framework</li>
                <li><strong>SVM:</strong> Support vector classification</li>
                <li><strong>Logistic Regression:</strong> Linear probabilistic model</li>
                <li><strong>Feature Engineering:</strong> Advanced preprocessing</li>
                <li><strong>Class Balancing:</strong> SMOTE oversampling</li>
            </ul>
        </div>
        """,
            unsafe_allow_html=True,
        )

    with col2:
        st.markdown(
            """
        <div class="info-box">
            <h4>🎯 System Capabilities</h4>
            <ul>
                <li>Multi-algorithm comparison framework</li>
                <li>Comprehensive overfitting analysis</li>
                <li>Statistical significance validation</li>
                <li>Advanced feature selection methods</li>
                <li>Cross-validation optimization</li>
                <li>Production-ready deployment</li>
            </ul>
        </div>
        """,
            unsafe_allow_html=True,
        )


def display_model_performance_analysis(artifacts):
    """Model performance comparative analysis dashboard"""
    st.subheader("🔬 Model Performance Analysis")

    if "performance_metrics" in artifacts:
        performance_data = artifacts["performance_metrics"]

        if isinstance(performance_data, dict):
            models_df = pd.DataFrame(performance_data)

            if not models_df.empty:
                col1, col2 = st.columns(2)

                with col1:
                    # Test accuracy comparison
                    if "test_accuracy" in models_df.index:
                        acc_data = models_df.loc["test_accuracy"].reset_index()
                        acc_data.columns = ["Model", "Accuracy"]
                        fig_acc = px.bar(
                            acc_data,
                            x="Model",
                            y="Accuracy",
                            title="Model Test Accuracy Comparison",
                            color="Accuracy",
                            color_continuous_scale="viridis",
                        )
                        fig_acc.update_layout(xaxis_tickangle=-45)
                        st.plotly_chart(fig_acc, use_container_width=True)

                with col2:
                    # Overfitting analysis
                    if "overfitting" in models_df.index:
                        over_data = models_df.loc["overfitting"].reset_index()
                        over_data.columns = ["Model", "Overfitting"]
                        fig_over = px.bar(
                            over_data,
                            x="Model",
                            y="Overfitting",
                            title="Model Overfitting Analysis",
                            color="Overfitting",
                            color_continuous_scale="RdYlBu_r",
                        )
                        fig_over.update_layout(xaxis_tickangle=-45)
                        st.plotly_chart(fig_over, use_container_width=True)

                # Comprehensive performance metrics table
                st.dataframe(models_df.T.round(4), use_container_width=True)

        # Best model identification
        if "best_model_name" in artifacts:
            st.markdown(
                f"""
            <div class="success-box">
                <h4>🏆 Optimal Model Selected</h4>
                <p><strong>Model:</strong> {artifacts['best_model_name']}</p>
                <p>Selected based on comprehensive performance evaluation across multiple metrics.</p>
            </div>
            """,
                unsafe_allow_html=True,
            )
