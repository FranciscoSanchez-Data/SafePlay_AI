"""
SafePlay AI - Responsible Intervention Recommendation Pipeline

Creates operational alerts and proportional responsible gaming recommendations.

Inputs:
- data/processed/user_features.csv
- data/processed/predictions.csv
- data/processed/top_drivers.csv

Outputs:
- data/processed/alerts.csv
- data/processed/intervention_recommendations.csv
- reports/intervention_summary.csv

Design principles:
- The system does not diagnose gambling addiction.
- The system does not make automatic punitive decisions.
- High-risk alerts require human review.
- Recommended actions are protective, not commercial.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------

PROCESSED_DIR = Path("data/processed")
REPORTS_DIR = Path("reports")

USER_FEATURES_PATH = PROCESSED_DIR / "user_features.csv"
PREDICTIONS_PATH = PROCESSED_DIR / "predictions.csv"
TOP_DRIVERS_PATH = PROCESSED_DIR / "top_drivers.csv"

ALERTS_PATH = PROCESSED_DIR / "alerts.csv"
RECOMMENDATIONS_PATH = PROCESSED_DIR / "intervention_recommendations.csv"
INTERVENTION_SUMMARY_PATH = REPORTS_DIR / "intervention_summary.csv"

TOP_N_DRIVERS_PER_USER = 5


# -----------------------------------------------------------------------------
# Human-readable policies
# -----------------------------------------------------------------------------

RISK_LEVEL_ES = {
    "low": "Bajo",
    "medium": "Medio",
    "high": "Alto",
}

INTERVENTION_POLICY = {
    "low": {
        "review_priority": "none",
        "recommended_action": "sin_intervencion_individual",
        "recommended_action_label": "Sin intervención individual",
        "action_detail": (
            "Mantener comunicación general sobre juego responsable y sugerencia opcional "
            "de configuración de límites."
        ),
        "human_review_required": 0,
        "commercial_suppression": 0,
        "audit_required": 0,
    },
    "medium": {
        "review_priority": "standard_monitoring",
        "recommended_action": "mensaje_preventivo_y_monitorizacion",
        "recommended_action_label": "Mensaje preventivo y monitorización",
        "action_detail": (
            "Mostrar mensaje preventivo, sugerir límites voluntarios de depósito o tiempo, "
            "reducir presión promocional y monitorizar evolución."
        ),
        "human_review_required": 0,
        "commercial_suppression": 1,
        "audit_required": 1,
    },
    "high": {
        "review_priority": "human_review",
        "recommended_action": "revision_humana_y_pausa_preventiva",
        "recommended_action_label": "Revisión humana y pausa preventiva",
        "action_detail": (
            "Excluir de campañas comerciales, activar revisión humana, mostrar mensaje de pausa, "
            "sugerir herramientas de autoexclusión o enfriamiento y registrar alerta para auditoría."
        ),
        "human_review_required": 1,
        "commercial_suppression": 1,
        "audit_required": 1,
    },
}

ANOMALY_ACTION = {
    "review_priority": "anomaly_review",
    "recommended_action": "revision_por_anomalia",
    "recommended_action_label": "Revisión por anomalía",
    "action_detail": (
        "El usuario presenta un patrón atípico frente al conjunto analizado. Revisar manualmente "
        "si la anomalía es compatible con una señal preventiva o con un caso benigno."
    ),
    "human_review_required": 1,
    "commercial_suppression": 0,
    "audit_required": 1,
}


# -----------------------------------------------------------------------------
# Loading
# -----------------------------------------------------------------------------


def ensure_output_dirs() -> None:
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)



def load_inputs() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    missing = [
        path
        for path in [USER_FEATURES_PATH, PREDICTIONS_PATH, TOP_DRIVERS_PATH]
        if not path.exists()
    ]
    if missing:
        raise FileNotFoundError(f"Missing required input files: {missing}")

    features = pd.read_csv(USER_FEATURES_PATH)
    predictions = pd.read_csv(PREDICTIONS_PATH)
    top_drivers = pd.read_csv(TOP_DRIVERS_PATH)

    return features, predictions, top_drivers


# -----------------------------------------------------------------------------
# Recommendation logic
# -----------------------------------------------------------------------------


def get_policy_for_user(predicted_risk_label: str, anomaly_flag: int) -> dict:
    """Return the intervention policy for a user."""
    policy = INTERVENTION_POLICY[predicted_risk_label].copy()

    # If a user is predicted as low but flagged as anomalous, route it to a light
    # human review queue. This avoids ignoring rare unusual patterns.
    if predicted_risk_label == "low" and int(anomaly_flag) == 1:
        policy = ANOMALY_ACTION.copy()

    return policy



def build_driver_strings(top_drivers: pd.DataFrame) -> pd.DataFrame:
    """Aggregate top SHAP drivers into one row per user."""
    top_drivers_sorted = top_drivers.sort_values(["user_id", "driver_rank"])
    top_drivers_limited = top_drivers_sorted[top_drivers_sorted["driver_rank"] <= TOP_N_DRIVERS_PER_USER]

    drivers_agg = (
        top_drivers_limited.groupby("user_id")
        .agg(
            main_driver=("driver_label", "first"),
            main_drivers=("driver_label", lambda values: " | ".join(dict.fromkeys(values))),
        )
        .reset_index()
    )

    return drivers_agg



def build_recommendations(
    features: pd.DataFrame,
    predictions: pd.DataFrame,
    top_drivers: pd.DataFrame,
) -> pd.DataFrame:
    """Create one recommendation row per user."""
    driver_strings = build_driver_strings(top_drivers)

    selected_feature_cols = [
        "user_id",
        "channel_preference",
        "sessions_7d",
        "sessions_30d",
        "deposit_amount_7d",
        "deposit_count_7d",
        "net_loss_7d",
        "net_loss_30d",
        "night_sessions_ratio_30d",
        "loss_chasing_events_30d",
        "cancelled_limit_recently",
    ]

    available_feature_cols = [col for col in selected_feature_cols if col in features.columns]

    recommendations = predictions.merge(
        features[available_feature_cols],
        on="user_id",
        how="left",
    ).merge(
        driver_strings,
        on="user_id",
        how="left",
    )

    recommendations["main_driver"] = recommendations["main_driver"].fillna("sin driver destacado")
    recommendations["main_drivers"] = recommendations["main_drivers"].fillna("sin driver destacado")

    policy_rows = []
    for row in recommendations.itertuples(index=False):
        policy = get_policy_for_user(row.predicted_risk_label, row.anomaly_flag)
        policy_rows.append(policy)

    policy_df = pd.DataFrame(policy_rows)
    recommendations = pd.concat([recommendations.reset_index(drop=True), policy_df], axis=1)

    recommendations["risk_level_es"] = recommendations["predicted_risk_label"].map(RISK_LEVEL_ES)

    # Operational prioritisation score for the alert monitor.
    # This is not a clinical score. It is a queue ordering helper.
    recommendations["operational_priority_score"] = (
        0.70 * recommendations["risk_score_high"].fillna(0)
        + 0.20 * recommendations["anomaly_flag"].fillna(0)
        + 0.10 * np.where(recommendations["cancelled_limit_recently"].fillna(0) == 1, 1, 0)
    ).clip(0, 1)

    recommendations["alert_required"] = (
        (recommendations["predicted_risk_label"].isin(["medium", "high"]))
        | (recommendations["anomaly_flag"].astype(int) == 1)
    ).astype(int)

    recommendations["alert_reason"] = np.select(
        [
            recommendations["predicted_risk_label"] == "high",
            recommendations["predicted_risk_label"] == "medium",
            (recommendations["predicted_risk_label"] == "low") & (recommendations["anomaly_flag"] == 1),
        ],
        [
            "riesgo alto predicho por el modelo",
            "señales tempranas compatibles con riesgo medio",
            "patrón atípico detectado por capa no supervisada",
        ],
        default="sin alerta operativa",
    )

    return recommendations



def build_alerts(recommendations: pd.DataFrame) -> pd.DataFrame:
    """Create the operational alert monitor table."""
    alerts = recommendations[recommendations["alert_required"] == 1].copy()

    # Highest-risk users first.
    alerts = alerts.sort_values(
        by=["predicted_risk_label", "operational_priority_score", "risk_score_high"],
        ascending=[True, False, False],
    )

    risk_order = {"high": 0, "medium": 1, "low": 2}
    alerts["risk_sort"] = alerts["predicted_risk_label"].map(risk_order)
    alerts = alerts.sort_values(
        by=["risk_sort", "operational_priority_score", "risk_score_high"],
        ascending=[True, False, False],
    ).drop(columns=["risk_sort"])

    output_columns = [
        "user_id",
        "risk_level_es",
        "predicted_risk_label",
        "risk_score_high",
        "risk_score",
        "operational_priority_score",
        "channel_preference",
        "main_driver",
        "main_drivers",
        "recommended_action_label",
        "action_detail",
        "review_priority",
        "human_review_required",
        "commercial_suppression",
        "audit_required",
        "anomaly_flag",
        "anomaly_score",
        "alert_reason",
        "sessions_7d",
        "sessions_30d",
        "deposit_amount_7d",
        "deposit_count_7d",
        "net_loss_7d",
        "net_loss_30d",
        "night_sessions_ratio_30d",
        "loss_chasing_events_30d",
        "cancelled_limit_recently",
    ]

    available_columns = [col for col in output_columns if col in alerts.columns]
    return alerts[available_columns]


# -----------------------------------------------------------------------------
# Reporting
# -----------------------------------------------------------------------------


def build_intervention_summary(recommendations: pd.DataFrame, alerts: pd.DataFrame) -> pd.DataFrame:
    summary = (
        recommendations.groupby(["predicted_risk_label", "recommended_action_label"])
        .agg(
            users=("user_id", "nunique"),
            avg_risk_score_high=("risk_score_high", "mean"),
            human_reviews=("human_review_required", "sum"),
            commercial_suppressions=("commercial_suppression", "sum"),
            audit_records=("audit_required", "sum"),
        )
        .reset_index()
    )

    summary["user_share"] = summary["users"] / recommendations["user_id"].nunique()
    summary["alerts_generated_total"] = len(alerts)
    return summary.round(4)



def print_acceptance_summary(recommendations: pd.DataFrame, alerts: pd.DataFrame, summary: pd.DataFrame) -> None:
    print("\nIntervention summary")
    print("--------------------")
    print(f"Users with recommendations: {len(recommendations):,}")
    print(f"Operational alerts:         {len(alerts):,}")
    print(f"Human reviews required:     {int(recommendations['human_review_required'].sum()):,}")
    print(f"Commercial suppressions:    {int(recommendations['commercial_suppression'].sum()):,}")
    print(f"Audit records required:     {int(recommendations['audit_required'].sum()):,}")

    print("\nRecommendations by risk level:")
    print(summary.to_string(index=False))

    if not alerts.empty:
        print("\nTop 10 alerts for monitor:")
        display_cols = [
            "user_id",
            "risk_level_es",
            "risk_score_high",
            "channel_preference",
            "main_driver",
            "recommended_action_label",
            "human_review_required",
        ]
        available_cols = [col for col in display_cols if col in alerts.columns]
        print(alerts[available_cols].head(10).to_string(index=False))


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------


def main() -> None:
    ensure_output_dirs()

    print("Loading predictions, features and top drivers...")
    features, predictions, top_drivers = load_inputs()

    print("Building responsible intervention recommendations...")
    recommendations = build_recommendations(features, predictions, top_drivers)
    alerts = build_alerts(recommendations)
    summary = build_intervention_summary(recommendations, alerts)

    recommendations.to_csv(RECOMMENDATIONS_PATH, index=False, encoding="utf-8")
    alerts.to_csv(ALERTS_PATH, index=False, encoding="utf-8")
    summary.to_csv(INTERVENTION_SUMMARY_PATH, index=False, encoding="utf-8")

    print("\nResponsible intervention pipeline completed successfully.")
    print(f"Recommendations: {RECOMMENDATIONS_PATH}")
    print(f"Alerts:          {ALERTS_PATH}")
    print(f"Summary:         {INTERVENTION_SUMMARY_PATH}")

    print_acceptance_summary(recommendations, alerts, summary)

    print("\nGovernance note:")
    print(
        "SafePlay AI prioritizes preventive review. It does not diagnose addiction, "
        "does not impose automatic sanctions and does not optimize vulnerable users for marketing."
    )


if __name__ == "__main__":
    main()
