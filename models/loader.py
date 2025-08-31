import streamlit as st
import pickle
from config.settings import Settings


@st.cache_resource
def load_deployment_artifacts():
    """Load serialized model deployment artifacts"""
    try:
        artifacts = {}

        for key, filename in Settings.model.ARTIFACT_FILES.items():
            try:
                with open(f"artifacts/{filename}", "rb") as f:
                    artifacts[key] = pickle.load(f)
            except FileNotFoundError:
                st.error(f"Required deployment file not found: {filename}")
                return None

        return artifacts
    except Exception as e:
        st.error(f"Error loading deployment artifacts: {str(e)}")
        return None
