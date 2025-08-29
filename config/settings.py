from dataclasses import dataclass


@dataclass
class AppConfig:
    PAGE_TITLE: str = "Obesity Level Prediction System"
    PAGE_ICON: str = "⚖️"
    LAYOUT: str = "wide"
    SIDEBAR_STATE: str = "expanded"


@dataclass
class ModelConfig:
    ARTIFACT_FILES: dict = None

    def __post_init__(self):
        self.ARTIFACT_FILES = {
            "model": "final_model.pkl",
            "scaler": "scaler.pkl",
            "feature_names": "selected_features.pkl",
            "continuous_features": "continuous_features.pkl",
            "target_mapping": "target_mapping.pkl",
            "performance_metrics": "model_performance.pkl",
            "feature_importance_rf": "feature_importance_rf.pkl",
            "feature_importance_xgb": "feature_importance_xgb.pkl",
            "best_model_name": "best_model_name.pkl",
        }


class Settings:
    app = AppConfig()
    model = ModelConfig()
