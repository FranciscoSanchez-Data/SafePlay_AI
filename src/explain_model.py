"""
SafePlay AI - Model Explainability Pipeline

Creates global and user-level explanations for the selected supervised model.

Inputs:
- data/processed/user_features.csv
- data/processed/predictions.csv
- models/risk_model.pkl
- models/label_mapping.json

Outputs:
- reports/shap_global_importance_high.csv
- reports/shap_summary_high.png
- data/processed/top_drivers.csv

Notes:
- Explanations focus on the `high` risk class because this is the most relevant
  class for responsible gaming review workflows.
- SHAP values are used for XGBoost. If SHAP is not installed, the script raises
  a clear installation message.
- Explanations support human review. They are not clinical diagnoses and should
  not trigger automatic punitive decisions.
"""

from __future__ import annotations

import json
import pickle
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------

PROCESSED_DIR = Path("data/processed")
MODELS_DIR = Path("models")
REPORTS_DIR = Path("reports")

FEATURES_PATH = PROCESSED_DIR / "user_features.csv"
PREDICTIONS_PATH = PROCESSED_DIR / "predictions.csv"
RISK_MODEL_PATH = MODELS_DIR / "risk_model.pkl"
LABEL_MAPPING_PATH = MODELS_DIR / "label_mapping.json"

GLOBAL_IMPORTANCE_PATH = REPORTS_DIR / "shap_global_importance_high.csv"
SHAP_SUMMARY_PATH = REPORTS_DIR / "shap_summary_high.png"
TOP_DRIVERS_PATH = PROCESSED_DIR / "top_drivers.csv"

HIGH_CLASS_ID = 2
HIGH_CLASS_NAME = "high"
TOP_N_DRIVERS = 5
MAX_USERS_FOR_SHAP_PLOT = 2_000
RANDOM_SEED = 42


# -----------------------------------------------------------------------------
# Driver labels for dashboard readability
# -----------------------------------------------------------------------------

DRIVER_LABELS = {
    "sessions_7d": "aumento de sesiones en los últimos 7 días",
    "sessions_14d": "aumento de sesiones en los últimos 14 días",
    "sessions_30d": "alta frecuencia de sesiones en los últimos 30 días",
    "active_days_30d": "más días activos durante el último mes",
    "frequency_increase_ratio": "incremento reciente de frecuencia frente al histórico",
    "days_since_last_session": "actividad muy reciente",
    "total_wagered_7d": "aumento del importe apostado en los últimos 7 días",
    "total_wagered_30d": "importe apostado elevado en los últimos 30 días",
    "net_loss_7d": "pérdida neta reciente elevada",
    "net_loss_30d": "pérdida neta acumulada elevada en los últimos 30 días",
    "avg_bet_amount_30d": "ticket medio elevado en los últimos 30 días",
    "avg_wagered_per_session_7d": "importe medio por sesión elevado recientemente",
    "avg_wagered_per_session_30d": "importe medio por sesión elevado en el último mes",
    "stake_increase_ratio": "incremento del ticket medio frente al histórico",
    "deposit_amount_7d": "aumento de depósitos en los últimos 7 días",
    "deposit_count_7d": "más depósitos recientes",
    "deposit_increase_ratio": "incremento de depósitos frente al histórico",
    "loss_increase_ratio": "incremento de pérdidas frente al histórico",
    "avg_session_duration_30d": "sesiones más largas en el último mes",
    "max_session_duration_30d": "sesión máxima especialmente larga",
    "night_sessions_ratio_30d": "actividad nocturna recurrente",
    "loss_chasing_events_30d": "retornos rápidos después de pérdidas",
    "product_switch_count_30d": "cambios frecuentes entre productos",
    "channel_switch_count_30d": "cambios recientes entre canales",
    "limit_configured": "configuración de límites de juego",
    "cancelled_limit_recently": "cancelación reciente de límites",
    "self_exclusion_attempt": "uso o intento de autoexclusión",
    "cooling_off_used": "uso de herramientas de pausa o enfriamiento",
    "channel_preference_online": "preferencia por canal online",
    "channel_preference_retail": "preferencia por canal retail",
}


# -----------------------------------------------------------------------------
# Utility functions
# -----------------------------------------------------------------------------


def ensure_output_dirs() -> None:
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)



def load_pickle(path: Path) -> Any:
    if not path.exists():
        raise FileNotFoundError(f"Missing file: {path}")
    with open(path, "rb") as file:
        return pickle.load(file)



def load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Missing file: {path}")
    with open(path, "r", encoding="utf-8") as file:
        return json.load(file)



def load_inputs() -> tuple[pd.DataFrame, pd.DataFrame, Any, dict[str, Any]]:
    if not FEATURES_PATH.exists():
        raise FileNotFoundError(f"Missing input file: {FEATURES_PATH}")
    if not PREDICTIONS_PATH.exists():
        raise FileNotFoundError(f"Missing input file: {PREDICTIONS_PATH}")

    features = pd.read_csv(FEATURES_PATH)
    predictions = pd.read_csv(PREDICTIONS_PATH)
    model = load_pickle(RISK_MODEL_PATH)
    label_mapping = load_json(LABEL_MAPPING_PATH)

    return features, predictions, model, label_mapping



def prepare_model_input(features: pd.DataFrame, label_mapping: dict[str, Any]) -> pd.DataFrame:
    feature_columns = label_mapping["feature_columns"]
    missing = [col for col in feature_columns if col not in features.columns]
    if missing:
        raise ValueError(f"Missing model feature columns: {missing}")
    return features[feature_columns].copy()



def get_transformed_feature_names(model: Any, label_mapping: dict[str, Any]) -> list[str]:
    names = label_mapping.get("transformed_feature_names") or []
    if names:
        return list(names)

    preprocessor = model.named_steps["preprocessor"]
    try:
        return preprocessor.get_feature_names_out().tolist()
    except Exception:
        transformed = preprocessor.transform(pd.DataFrame(columns=label_mapping["feature_columns"]))
        return [f"feature_{idx}" for idx in range(transformed.shape[1])]



def normalize_feature_name(feature_name: str) -> str:
    """Convert transformed one-hot feature names into readable base names when possible."""
    # sklearn may return names such as channel_preference_online.
    return feature_name



def driver_to_human_label(feature_name: str) -> str:
    normalized = normalize_feature_name(feature_name)
    if normalized in DRIVER_LABELS:
        return DRIVER_LABELS[normalized]

    # Fallback for one-hot encoded variables or historical helper columns.
    for key, label in DRIVER_LABELS.items():
        if normalized.startswith(f"{key}_"):
            return label

    return normalized.replace("_", " ")


# -----------------------------------------------------------------------------
# SHAP extraction
# -----------------------------------------------------------------------------


def compute_high_class_shap_values(model: Any, X: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Return SHAP values for the high-risk class."""
    try:
        import shap
    except ImportError as exc:
        raise ImportError(
            "SHAP is not installed. Install it with `pip install shap` and rerun this script."
        ) from exc

    preprocessor = model.named_steps["preprocessor"]
    estimator = model.named_steps["model"]

    X_transformed = preprocessor.transform(X)
    feature_names = get_transformed_feature_names(model, load_json(LABEL_MAPPING_PATH))

    explainer = shap.TreeExplainer(estimator)
    shap_values_raw = explainer.shap_values(X_transformed)

    # SHAP can return different structures depending on package versions.
    if isinstance(shap_values_raw, list):
        shap_values_high = shap_values_raw[HIGH_CLASS_ID]
    elif isinstance(shap_values_raw, np.ndarray) and shap_values_raw.ndim == 3:
        # Common shape: rows x features x classes.
        shap_values_high = shap_values_raw[:, :, HIGH_CLASS_ID]
    elif isinstance(shap_values_raw, np.ndarray) and shap_values_raw.ndim == 2:
        # Binary-like fallback. Not expected for this multiclass model.
        shap_values_high = shap_values_raw
    else:
        raise ValueError(f"Unexpected SHAP values structure: {type(shap_values_raw)}")

    return np.asarray(shap_values_high), np.asarray(X_transformed), feature_names


# -----------------------------------------------------------------------------
# Reports
# -----------------------------------------------------------------------------


def build_global_importance(shap_values_high: np.ndarray, feature_names: list[str]) -> pd.DataFrame:
    mean_abs = np.abs(shap_values_high).mean(axis=0)
    importance = pd.DataFrame(
        {
            "feature": feature_names,
            "driver_label": [driver_to_human_label(feature) for feature in feature_names],
            "mean_abs_shap_high": mean_abs,
        }
    )
    importance = importance.sort_values("mean_abs_shap_high", ascending=False).reset_index(drop=True)
    importance["rank"] = np.arange(1, len(importance) + 1)
    return importance[["rank", "feature", "driver_label", "mean_abs_shap_high"]]



def build_user_top_drivers(
    features: pd.DataFrame,
    predictions: pd.DataFrame,
    shap_values_high: np.ndarray,
    feature_names: list[str],
) -> pd.DataFrame:
    """Build top positive high-risk drivers per user."""
    rows = []
    user_ids = features["user_id"].tolist()

    pred_lookup = predictions.set_index("user_id").to_dict(orient="index")

    for row_idx, user_id in enumerate(user_ids):
        user_shap = shap_values_high[row_idx]

        # Positive values push the prediction towards high risk.
        positive_indices = np.where(user_shap > 0)[0]
        if len(positive_indices) == 0:
            selected_indices = np.argsort(np.abs(user_shap))[::-1][:TOP_N_DRIVERS]
        else:
            selected_indices = positive_indices[np.argsort(user_shap[positive_indices])[::-1]][:TOP_N_DRIVERS]

        pred_info = pred_lookup.get(user_id, {})

        for rank, feature_idx in enumerate(selected_indices, start=1):
            feature_name = feature_names[int(feature_idx)]
            rows.append(
                {
                    "user_id": user_id,
                    "predicted_risk_label": pred_info.get("predicted_risk_label"),
                    "risk_score_high": pred_info.get("risk_score_high"),
                    "anomaly_score": pred_info.get("anomaly_score"),
                    "anomaly_flag": pred_info.get("anomaly_flag"),
                    "driver_rank": rank,
                    "feature": feature_name,
                    "driver_label": driver_to_human_label(feature_name),
                    "shap_value_high": float(user_shap[int(feature_idx)]),
                    "abs_shap_value_high": float(abs(user_shap[int(feature_idx)])),
                }
            )

    return pd.DataFrame(rows)



def save_shap_summary_plot(
    shap_values_high: np.ndarray,
    X_transformed: np.ndarray,
    feature_names: list[str],
) -> None:
    """Save a SHAP summary plot for the high-risk class."""
    try:
        import matplotlib.pyplot as plt
        import shap
    except ImportError as exc:
        raise ImportError(
            "SHAP and matplotlib are required for the summary plot. Install them with `pip install shap matplotlib`."
        ) from exc

    rng = np.random.default_rng(RANDOM_SEED)
    n_rows = X_transformed.shape[0]

    if n_rows > MAX_USERS_FOR_SHAP_PLOT:
        sample_idx = rng.choice(n_rows, size=MAX_USERS_FOR_SHAP_PLOT, replace=False)
        shap_plot_values = shap_values_high[sample_idx]
        X_plot = X_transformed[sample_idx]
    else:
        shap_plot_values = shap_values_high
        X_plot = X_transformed

    plt.figure()
    shap.summary_plot(
        shap_plot_values,
        X_plot,
        feature_names=feature_names,
        show=False,
        max_display=20,
    )
    plt.tight_layout()
    plt.savefig(SHAP_SUMMARY_PATH, dpi=160, bbox_inches="tight")
    plt.close()


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------


def main() -> None:
    ensure_output_dirs()

    print("Loading model, predictions and feature dataset...")
    features, predictions, model, label_mapping = load_inputs()
    X = prepare_model_input(features, label_mapping)

    print("Computing SHAP values for high-risk class...")
    shap_values_high, X_transformed, feature_names = compute_high_class_shap_values(model, X)

    print("Building global SHAP importance report...")
    global_importance = build_global_importance(shap_values_high, feature_names)
    global_importance.to_csv(GLOBAL_IMPORTANCE_PATH, index=False)

    print("Building user-level top drivers...")
    top_drivers = build_user_top_drivers(features, predictions, shap_values_high, feature_names)
    top_drivers.to_csv(TOP_DRIVERS_PATH, index=False)

    print("Saving SHAP summary plot...")
    save_shap_summary_plot(shap_values_high, X_transformed, feature_names)

    print("\nExplainability pipeline completed successfully.")
    print(f"Global importance: {GLOBAL_IMPORTANCE_PATH}")
    print(f"Top drivers:       {TOP_DRIVERS_PATH}")
    print(f"SHAP summary plot: {SHAP_SUMMARY_PATH}")

    print("\nTop 15 global drivers for high-risk class:")
    print(global_importance.head(15).to_string(index=False))

    high_alerts = predictions[predictions["predicted_risk_label"] == HIGH_CLASS_NAME]
    if not high_alerts.empty:
        sample_user_id = str(high_alerts.sort_values("risk_score_high", ascending=False).iloc[0]["user_id"])
        print(f"\nExample explanation for high-risk user {sample_user_id}:")
        print(
            top_drivers[top_drivers["user_id"] == sample_user_id]
            .sort_values("driver_rank")
            [["driver_rank", "driver_label", "shap_value_high"]]
            .to_string(index=False)
        )


if __name__ == "__main__":
    main()
