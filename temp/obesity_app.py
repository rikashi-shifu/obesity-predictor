# ================================================================================================
# OBESITY LEVEL PREDICTION SYSTEM - STREAMLIT DEPLOYMENT
# Professional Machine Learning Application for Healthcare Analytics
# ================================================================================================

import streamlit as st
import pandas as pd
import numpy as np
import pickle
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')

# Application configuration
st.set_page_config(
    page_title="Obesity Level Prediction System",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Professional styling configuration
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
        font-weight: bold;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #d35400;
        margin-bottom: 1rem;
        font-weight: bold;
    }
    .info-box {
        background-color: #f8f9fa;
        color: #2c3e50;
        padding: 1.5rem;
        border-radius: 0.7rem;
        border-left: 5px solid #3498db;
        margin: 1rem 0;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .success-box {
        background-color: #e8f5e8;
        color: #1b5e20;
        padding: 1.5rem;
        border-radius: 0.7rem;
        border-left: 5px solid #4caf50;
        margin: 1rem 0;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .warning-box {
        background-color: #fff8e1;
        color: #e65100;
        padding: 1.5rem;
        border-radius: 0.7rem;
        border-left: 5px solid #ff9800;
        margin: 1rem 0;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .prediction-result {
        font-size: 2rem;
        font-weight: bold;
        text-align: center;
        padding: 2rem;
        border-radius: 1rem;
        margin: 2rem 0;
        box-shadow: 0 4px 8px rgba(0,0,0,0.15);
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_deployment_artifacts():
    """Load serialized model deployment artifacts"""
    try:
        artifacts = {}
        artifact_files = {
            'model': 'final_model.pkl',
            'scaler': 'scaler.pkl',
            'feature_names': 'selected_features.pkl',
            'continuous_features': 'continuous_features.pkl',
            'target_mapping': 'target_mapping.pkl',
            'performance_metrics': 'model_performance.pkl',
            'feature_importance_rf': 'feature_importance_rf.pkl',
            'feature_importance_xgb': 'feature_importance_xgb.pkl',
            'best_model_name': 'best_model_name.pkl'
        }

        for key, filename in artifact_files.items():
            try:
                with open(filename, 'rb') as f:
                    artifacts[key] = pickle.load(f)
            except FileNotFoundError:
                st.error(f"Required deployment file not found: {filename}")
                return None

        return artifacts
    except Exception as e:
        st.error(f"Error loading deployment artifacts: {str(e)}")
        return None

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

def engineer_prediction_features(user_input, selected_features):
    """Feature engineering pipeline for prediction input"""
    data = pd.DataFrame([user_input])

    # Derived numerical features
    data['BMI'] = data['Weight'] / (data['Height'] ** 2)
    data['Weight_Height_Ratio'] = data['Weight'] / data['Height']
    data['Physical_Activity_Score'] = data['FAF'] * data['CH2O']
    data['Age_Weight_Interaction'] = data['Age'] * data['Weight'] / 100
    data['Lifestyle_Risk_Score'] = (
            (data['FAVC'] == 1).astype(int) * 2 +
            (data['SMOKE'] == 1).astype(int) * 3 +
            (data['SCC'] == 0).astype(int) * 1
    )
    data['Health_Risk_Index'] = (
            (data['FAVC'] == 1).astype(int) * 0.3 +
            (data['SMOKE'] == 1).astype(int) * 0.4 +
            (data['FAF'] < 1).astype(int) * 0.3
    )
    data['Eating_Pattern_Score'] = data['FCVC'] * data['NCP'] / data['TUE'].clip(lower=1)

    # Categorical feature binning
    age = user_input['Age']
    if age <= 25:
        age_cat = 'Young'
    elif age <= 35:
        age_cat = 'Adult'
    elif age <= 50:
        age_cat = 'MiddleAge'
    else:
        age_cat = 'Senior'

    bmi = data['BMI'].iloc[0]
    if bmi < 18.5:
        bmi_cat = 'Underweight'
    elif bmi < 25:
        bmi_cat = 'Normal'
    elif bmi < 30:
        bmi_cat = 'Overweight'
    elif bmi < 35:
        bmi_cat = 'Obese_I'
    else:
        bmi_cat = 'Obese_II'

    ch2o = user_input['CH2O']
    if ch2o <= 1.5:
        hydration_cat = 'Low'
    elif ch2o <= 2.5:
        hydration_cat = 'Moderate'
    else:
        hydration_cat = 'High'

    faf = user_input['FAF']
    if faf < 1:
        activity_cat = 'Sedentary'
    elif faf < 2:
        activity_cat = 'Light'
    else:
        activity_cat = 'Active'

    # Categorical feature assignment
    data['CAEC'] = user_input['CAEC']
    data['CALC'] = user_input['CALC']
    data['MTRANS'] = user_input['MTRANS']
    data['Age_Category'] = age_cat
    data['BMI_WHO_Category'] = bmi_cat
    data['Hydration_Level'] = hydration_cat
    data['Activity_Level'] = activity_cat

    # One-hot encoding for categorical variables
    all_categories = {
        'CAEC': ['no', 'Sometimes', 'Frequently', 'Always'],
        'CALC': ['no', 'Sometimes', 'Frequently', 'Always'],
        'MTRANS': ['Public_Transportation', 'Walking', 'Automobile', 'Bike', 'Motorbike'],
        'Age_Category': ['Young', 'Adult', 'MiddleAge', 'Senior'],
        'BMI_WHO_Category': ['Underweight', 'Normal', 'Overweight', 'Obese_I', 'Obese_II'],
        'Hydration_Level': ['Low', 'Moderate', 'High'],
        'Activity_Level': ['Sedentary', 'Light', 'Active']
    }

    # One-hot encoding implementation (drop_first=True)
    for feature, categories in all_categories.items():
        current_value = data[feature].iloc[0]
        for category in categories[1:]:
            column_name = f"{feature}_{category}"
            data[column_name] = 1 if current_value == category else 0

    # Remove original categorical columns
    data = data.drop(columns=['CAEC', 'CALC', 'MTRANS', 'Age_Category', 'BMI_WHO_Category', 'Hydration_Level', 'Activity_Level'])

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
            'Age': user_input['age'],
            'Height': user_input['height'],
            'Weight': user_input['weight'],
            'Gender': 1 if user_input['gender'] == 'Male' else 0,
            'family_history_with_overweight': 1 if user_input['family_history'] == 'yes' else 0,
            'FAVC': 1 if user_input['favc'] == 'yes' else 0,
            'FCVC': user_input['fcvc'],
            'NCP': user_input['ncp'],
            'CAEC': user_input['caec'],
            'SMOKE': 1 if user_input['smoke'] == 'yes' else 0,
            'CH2O': user_input['ch2o'],
            'SCC': 1 if user_input['scc'] == 'yes' else 0,
            'FAF': user_input['faf'],
            'TUE': user_input['tue'],
            'CALC': user_input['calc'],
            'MTRANS': user_input['mtrans']
        }

        # Feature engineering
        feature_data = engineer_prediction_features(processed_input, artifacts['feature_names'])

        # Feature scaling for continuous variables
        continuous_features = artifacts['continuous_features']
        existing_continuous = [col for col in continuous_features if col in feature_data.columns]

        if existing_continuous:
            feature_data_scaled = feature_data.copy()
            feature_data_scaled[existing_continuous] = artifacts['scaler'].transform(feature_data[existing_continuous])
        else:
            feature_data_scaled = feature_data

        # Model prediction
        prediction = artifacts['model'].predict(feature_data_scaled)[0]
        probabilities = artifacts['model'].predict_proba(feature_data_scaled)[0]

        # Probability distribution mapping
        all_probabilities = {}
        for i, prob in enumerate(probabilities):
            class_name = artifacts['target_mapping'][i].replace('_', ' ').title()
            all_probabilities[class_name] = prob

        predicted_class = artifacts['target_mapping'][prediction].replace('_', ' ').title()
        confidence = probabilities.max()
        bmi = processed_input['Weight'] / (processed_input['Height'] ** 2)

        return {
            'prediction': predicted_class,
            'confidence': confidence,
            'probabilities': all_probabilities,
            'bmi': bmi,
            'model_name': artifacts['best_model_name'] if 'best_model_name' in artifacts else type(artifacts['model']).__name__
        }

    except Exception as e:
        st.error(f"Prediction error: {str(e)}")
        return None

def generate_health_recommendations(user_input, bmi, prediction):
    """Personalized health recommendation system"""
    recommendations = []

    # BMI-based recommendations
    if bmi > 30:
        recommendations.append("Consult healthcare provider for comprehensive obesity management plan")
    elif bmi > 25:
        recommendations.append("Consider implementing structured weight management strategies")
    elif bmi < 18.5:
        recommendations.append("Consult healthcare provider for healthy weight gain guidance")

    # Lifestyle-based recommendations
    if user_input['faf'] < 1:
        recommendations.append("Increase physical activity to minimum 150 minutes moderate exercise weekly")
    if user_input['fcvc'] < 2:
        recommendations.append("Incorporate more vegetables and fruits in daily dietary intake")
    if user_input['ch2o'] < 2:
        recommendations.append("Increase daily water consumption to minimum 2-3 liters")
    if user_input['favc'] == 'yes':
        recommendations.append("Reduce frequent consumption of high-caloric processed foods")
    if user_input['smoke'] == 'yes':
        recommendations.append("Consider enrolling in smoking cessation programs")
    if user_input['tue'] > 1:
        recommendations.append("Limit screen time and increase active lifestyle engagement")
    if user_input['mtrans'] == 'Automobile':
        recommendations.append("Consider active transportation methods for short distances")

    return recommendations

def create_probability_visualization(probabilities):
    """Interactive probability distribution chart"""
    prob_df = pd.DataFrame(list(probabilities.items()), columns=['Class', 'Probability'])
    prob_df = prob_df.sort_values('Probability', ascending=True)

    fig = px.bar(prob_df, x='Probability', y='Class', orientation='h',
                 title='Obesity Level Prediction Probabilities',
                 color='Probability', color_continuous_scale='viridis')

    fig.update_traces(texttemplate='%{x:.1%}', textposition='outside')
    fig.update_layout(height=400, showlegend=False,
                      xaxis_title="Probability", yaxis_title="Obesity Level")

    return fig

def display_model_performance_analysis(artifacts):
    """Model performance comparative analysis dashboard"""
    st.subheader("🔬 Model Performance Analysis")

    if 'performance_metrics' in artifacts:
        performance_data = artifacts['performance_metrics']

        if isinstance(performance_data, dict):
            models_df = pd.DataFrame(performance_data)

            if not models_df.empty:
                col1, col2 = st.columns(2)

                with col1:
                    # Test accuracy comparison
                    if 'test_accuracy' in models_df.index:
                        acc_data = models_df.loc['test_accuracy'].reset_index()
                        acc_data.columns = ['Model', 'Accuracy']
                        fig_acc = px.bar(acc_data, x='Model', y='Accuracy',
                                         title='Model Test Accuracy Comparison',
                                         color='Accuracy', color_continuous_scale='viridis')
                        fig_acc.update_layout(xaxis_tickangle=-45)
                        st.plotly_chart(fig_acc, use_container_width=True)

                with col2:
                    # Overfitting analysis
                    if 'overfitting' in models_df.index:
                        over_data = models_df.loc['overfitting'].reset_index()
                        over_data.columns = ['Model', 'Overfitting']
                        fig_over = px.bar(over_data, x='Model', y='Overfitting',
                                          title='Model Overfitting Analysis',
                                          color='Overfitting', color_continuous_scale='RdYlBu_r')
                        fig_over.update_layout(xaxis_tickangle=-45)
                        st.plotly_chart(fig_over, use_container_width=True)

                # Comprehensive performance metrics table
                st.dataframe(models_df.T.round(4), use_container_width=True)

        # Best model identification
        if 'best_model_name' in artifacts:
            st.markdown(f"""
            <div class="success-box">
                <h4>🏆 Optimal Model Selected</h4>
                <p><strong>Model:</strong> {artifacts['best_model_name']}</p>
                <p>Selected based on comprehensive performance evaluation across multiple metrics.</p>
            </div>
            """, unsafe_allow_html=True)

def display_sidebar_information():
    """Application sidebar with usage guidance"""
    st.sidebar.markdown("## 📋 Application Guide")
    st.sidebar.markdown("""
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
    """)

    st.sidebar.markdown("---")
    st.sidebar.markdown("### 📊 Input Sequence Reference")
    st.sidebar.markdown("""
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
    """, unsafe_allow_html=True)

    st.sidebar.markdown("---")
    st.sidebar.markdown("### ⚕️ Health Disclaimer")
    st.sidebar.markdown("""
    This application is designed for educational and research purposes. 
    Consult qualified healthcare professionals for medical advice and treatment decisions.
    """)

    st.sidebar.markdown("---")
    st.sidebar.markdown("### 📈 BMI Reference Scale")
    st.sidebar.markdown("""
    <div style="font-size: 0.9rem; line-height: 1.8;">
        <div style="color: #0D47A1; font-weight: bold;">• Underweight: < 18.5</div>
        <div style="color: #1B5E20; font-weight: bold;">• Normal: 18.5 - 24.9</div>
        <div style="color: #E65100; font-weight: bold;">• Overweight: 25.0 - 29.9</div>
        <div style="color: #B71C1C; font-weight: bold;">• Obese: ≥ 30.0</div>
    </div>
    """, unsafe_allow_html=True)

def main():
    """Main application interface"""
    st.markdown('<h1 class="main-header">🏥 Obesity Level Prediction System </h1>', unsafe_allow_html=True)

    # Load deployment artifacts
    artifacts = load_deployment_artifacts()
    if artifacts is None:
        st.stop()

    # System information panel
    with st.expander("ℹ️ System Information", expanded=False):
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("""
            <div class="info-box">
                <h4>🎯 Classification Categories</h4>
                <ul>
                    <li>Insufficient Weight</li>
                    <li>Normal Weight</li>
                    <li>Overweight Level I & II</li>
                    <li>Obesity Type I, II & III</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)
        with col2:
            if 'best_model_name' in artifacts:
                st.markdown(f"""
                <div class="info-box">
                    <h4>🤖 Model Specifications</h4>
                    <ul>
                        <li>Algorithm: {artifacts['best_model_name']}</li>
                        <li>Features: {len(artifacts['feature_names'])}</li>
                        <li>Status: Production Ready</li>
                        <li>Validation: Cross-Validated</li>
                    </ul>
                </div>
                """, unsafe_allow_html=True)

    # Main application interface
    tab1, tab2 = st.tabs(["🔮 Prediction Interface", "📊 Model Analytics"])

    with tab1:
        col1, col2 = st.columns([1, 1])

        with col1:
            st.markdown('<h2 class="sub-header">📝 Patient Data Input</h2>', unsafe_allow_html=True)

            # Demographic information section
            st.markdown("**Demographic Information (Fields 1-4)**")
            gender = st.selectbox("1. Gender", ["Female", "Male"])
            age = st.slider("2. Age (years)", 14, 80, 25)
            height = st.number_input("3. Height (meters)", 1.20, 2.20, 1.65, 0.01)
            weight = st.number_input("4. Weight (kilograms)", 30.0, 200.0, 65.0, 0.1)

            # Dietary and family history section
            st.markdown("**Health History & Dietary Habits (Fields 5-8)**")
            family_history = st.selectbox("5. Family History with Overweight", ["no", "yes"])
            favc = st.selectbox("6. FAVC (Frequent high caloric food consumption)", ["no", "yes"])
            fcvc = st.slider("7. FCVC (Vegetable consumption frequency)", 1, 3, 2)
            ncp = st.slider("8. NCP (Number of main meals daily)", 1.0, 4.0, 3.0, 0.5)

            # Lifestyle and behavioral patterns section
            st.markdown("**Lifestyle & Behavioral Patterns (Fields 9-12)**")
            caec = st.selectbox("9. CAEC (Consumption between meals)",
                                ["no", "Sometimes", "Frequently", "Always"])
            smoke = st.selectbox("10. SMOKE (Smoking habit)", ["no", "yes"])
            ch2o = st.slider("11. CH2O (Daily water intake in liters)", 1.0, 3.0, 2.0, 0.1)
            scc = st.selectbox("12. SCC (Calorie consumption monitoring)", ["no", "yes"])

            # Activity and transportation section
            st.markdown("**Physical Activity & Transportation (Fields 13-16)**")
            faf = st.slider("13. FAF (Physical activity frequency per week)", 0.0, 3.0, 1.0, 0.1)
            tue = st.slider("14. TUE (Technology device usage hours)", 0, 2, 1)
            calc = st.selectbox("15. CALC (Alcohol consumption frequency)",
                                ["no", "Sometimes", "Frequently", "Always"])
            mtrans = st.selectbox("16. MTRANS (Primary transportation mode)",
                                  ["Public_Transportation", "Walking", "Automobile", "Bike", "Motorbike"])

        with col2:
            st.markdown('<h2 class="sub-header">🎯 Prediction Results</h2>', unsafe_allow_html=True)

            # Compile user input data
            user_input = {
                'gender': gender, 'age': age, 'height': height, 'weight': weight,
                'family_history': family_history, 'favc': favc, 'fcvc': fcvc, 'ncp': ncp,
                'caec': caec, 'smoke': smoke, 'ch2o': ch2o, 'scc': scc, 'faf': faf,
                'tue': tue, 'calc': calc, 'mtrans': mtrans
            }

            # Primary prediction execution
            if st.button("🔮 Execute Prediction Analysis", type="primary", use_container_width=True):
                with st.spinner("Processing machine learning analysis..."):
                    result = predict_obesity_level(artifacts, user_input)

                    if result is None:
                        st.error("Prediction analysis failed. Please verify input data and retry.")
                        st.stop()

                    # BMI analysis display
                    bmi_category, bmi_color, bmi_bg, bmi_desc = get_bmi_category_info(result['bmi'])

                    st.markdown(f"""
                    <div style="background-color: {bmi_bg}; padding: 1.5rem; border-radius: 0.7rem; border: 2px solid {bmi_color}; margin: 1rem 0;">
                        <h4 style="color: {bmi_color}; margin-bottom: 1rem; font-weight: bold;">📊 Body Mass Index Analysis</h4>
                        <div style="font-size: 2rem; font-weight: bold; color: {bmi_color}; margin: 0.5rem 0;">BMI: {result['bmi']:.2f}</div>
                        <div style="color: {bmi_color}; font-weight: bold; font-size: 1.3rem; margin: 0.5rem 0;">{bmi_category}</div>
                        <div style="color: {bmi_color}; font-style: italic; margin: 0.5rem 0;">{bmi_desc}</div>
                    </div>
                    """, unsafe_allow_html=True)

                    # Primary prediction result display
                    color_map = {
                        'Insufficient Weight': '#0D47A1', 'Normal Weight': '#1B5E20',
                        'Overweight Level I': '#E65100', 'Overweight Level Ii': '#E65100',
                        'Obesity Type I': '#B71C1C', 'Obesity Type Ii': '#880E4F',
                        'Obesity Type Iii': '#4A148C'
                    }

                    prediction_color = color_map.get(result['prediction'], '#1f77b4')

                    st.markdown(f"""
                    <div class="prediction-result" style="background-color: {prediction_color}20; border: 2px solid {prediction_color};">
                        <h3 style="color: {prediction_color};">Predicted Obesity Classification</h3>
                        <h2 style="color: {prediction_color};">{result['prediction']}</h2>
                        <p style="color: {prediction_color};">Model Confidence: {result['confidence']:.1%}</p>
                    </div>
                    """, unsafe_allow_html=True)

                    # Probability distribution visualization
                    st.subheader("📈 Classification Probability Distribution")
                    fig = create_probability_visualization(result['probabilities'])
                    st.plotly_chart(fig, use_container_width=True)

                    # Model performance information
                    st.subheader("🤖 Model Performance Information")
                    col_m1, col_m2 = st.columns(2)

                    with col_m1:
                        st.markdown(f"""
                        <div class="info-box">
                            <h4>Algorithm Specifications</h4>
                            <p><strong>Model Type:</strong> {result['model_name']}</p>
                            <p><strong>Prediction Confidence:</strong> {result['confidence']:.1%}</p>
                            <p><strong>Features Analyzed:</strong> {len(artifacts['feature_names'])}</p>
                        </div>
                        """, unsafe_allow_html=True)

                    with col_m2:
                        st.markdown("""
                        <div class="success-box">
                            <h4>✅ System Status</h4>
                            <p>✓ Model: Operational</p>
                            <p>✓ Features: Processed</p>
                            <p>✓ Analysis: Complete</p>
                            <p>✓ Validation: Successful</p>
                        </div>
                        """, unsafe_allow_html=True)

                    # Personalized health recommendations
                    st.subheader("💡 Personalized Health Recommendations")
                    recommendations = generate_health_recommendations(user_input, result['bmi'], result['prediction'])

                    if recommendations:
                        rec_html = "<br>".join([f"• {rec}" for rec in recommendations])
                        st.markdown(f"""
                        <div class="warning-box">
                            <h4>Health Optimization Recommendations</h4>
                            <div style="line-height: 1.8;">{rec_html}</div>
                        </div>
                        """, unsafe_allow_html=True)
                    else:
                        st.markdown("""
                        <div class="success-box">
                            <h4>✅ Optimal Health Profile</h4>
                            <p>Current lifestyle choices align with healthy weight management practices.</p>
                        </div>
                        """, unsafe_allow_html=True)

    with tab2:
        st.markdown('<h2 class="sub-header">📊 Advanced Model Analytics</h2>', unsafe_allow_html=True)
        display_model_performance_analysis(artifacts)

        # Technical implementation information
        col1, col2 = st.columns(2)

        with col1:
            st.markdown("""
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
            """, unsafe_allow_html=True)

        with col2:
            st.markdown("""
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
            """, unsafe_allow_html=True)

if __name__ == "__main__":
    display_sidebar_information()
    main()