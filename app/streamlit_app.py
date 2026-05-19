"""
SafePlay AI - Streamlit Dashboard

Internal-style dashboard for responsible gaming risk monitoring.

Pages:
1. Executive Overview
2. Alert Monitor
3. User Profile
4. Model Governance

Run from project root:
    streamlit run app/streamlit_app.py

Expected inputs:
- data/processed/user_features.csv
- data/processed/predictions.csv
- data/processed/top_drivers.csv
- data/processed/alerts.csv
- data/processed/intervention_recommendations.csv
- reports/model_comparison.csv
- reports/intervention_summary.csv
- reports/shap_global_importance_high.csv
- reports/shap_summary_high.png
- reports/confusion_matrix_xgboost.csv
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st


# -----------------------------------------------------------------------------
# Page configuration
# -----------------------------------------------------------------------------

st.set_page_config(
    page_title="SafePlay AI",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)


# -----------------------------------------------------------------------------
# Paths
# -----------------------------------------------------------------------------

ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT_DIR / "data"
PROCESSED_DIR = DATA_DIR / "processed"
SYNTHETIC_DIR = DATA_DIR / "synthetic"
REPORTS_DIR = ROOT_DIR / "reports"

USER_FEATURES_PATH = PROCESSED_DIR / "user_features.csv"
PREDICTIONS_PATH = PROCESSED_DIR / "predictions.csv"
TOP_DRIVERS_PATH = PROCESSED_DIR / "top_drivers.csv"
ALERTS_PATH = PROCESSED_DIR / "alerts.csv"
RECOMMENDATIONS_PATH = PROCESSED_DIR / "intervention_recommendations.csv"
SESSIONS_PATH = SYNTHETIC_DIR / "sessions.csv"

MODEL_COMPARISON_PATH = REPORTS_DIR / "model_comparison.csv"
INTERVENTION_SUMMARY_PATH = REPORTS_DIR / "intervention_summary.csv"
SHAP_GLOBAL_IMPORTANCE_PATH = REPORTS_DIR / "shap_global_importance_high.csv"
SHAP_SUMMARY_PATH = REPORTS_DIR / "shap_summary_high.png"
CONFUSION_MATRIX_XGB_PATH = REPORTS_DIR / "confusion_matrix_xgboost.csv"


# -----------------------------------------------------------------------------
# Visual constants
# -----------------------------------------------------------------------------

RISK_ORDER = ["low", "medium", "high"]
RISK_LABEL_ES = {
    "low": "Bajo",
    "medium": "Medio",
    "high": "Alto",
}

RISK_COLOR_MAP = {
    "low": "#2ca02c",
    "medium": "#ffbf00",
    "high": "#d62728",
    "Bajo": "#2ca02c",
    "Medio": "#ffbf00",
    "Alto": "#d62728",
}

PLOTLY_CONFIG = {
    "displayModeBar": False,
    "responsive": True,
}

BEHAVIOUR_LABELS = {
    "sessions_7d": "Sesiones últimos 7 días",
    "sessions_30d": "Sesiones últimos 30 días",
    "frequency_increase_ratio": "Incremento de frecuencia vs histórico",
    "stake_increase_ratio": "Incremento de ticket medio vs histórico",
    "deposit_amount_7d": "Importe depositado últimos 7 días",
    "deposit_count_7d": "Número de depósitos últimos 7 días",
    "net_loss_7d": "Pérdida neta últimos 7 días",
    "net_loss_30d": "Pérdida neta últimos 30 días",
    "night_sessions_ratio_30d": "Ratio de sesiones nocturnas últimos 30 días",
    "loss_chasing_events_30d": "Retornos rápidos tras pérdidas últimos 30 días",
    "avg_session_duration_30d": "Duración media de sesión últimos 30 días",
    "max_session_duration_30d": "Duración máxima de sesión últimos 30 días",
    "cancelled_limit_recently": "Canceló límites recientemente",
}

CURRENCY_FEATURES = {
    "deposit_amount_7d",
    "net_loss_7d",
    "net_loss_30d",
}

RATIO_FEATURES = {
    "frequency_increase_ratio",
    "stake_increase_ratio",
    "night_sessions_ratio_30d",
}


# -----------------------------------------------------------------------------
# Data loading
# -----------------------------------------------------------------------------

@st.cache_data(show_spinner=False)
def read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


@st.cache_data(show_spinner=False)
def load_all_data() -> dict[str, pd.DataFrame]:
    return {
        "features": read_csv(USER_FEATURES_PATH),
        "predictions": read_csv(PREDICTIONS_PATH),
        "top_drivers": read_csv(TOP_DRIVERS_PATH),
        "alerts": read_csv(ALERTS_PATH),
        "recommendations": read_csv(RECOMMENDATIONS_PATH),
        "sessions": read_csv(SESSIONS_PATH),
        "model_comparison": read_csv(MODEL_COMPARISON_PATH),
        "intervention_summary": read_csv(INTERVENTION_SUMMARY_PATH),
        "shap_importance": read_csv(SHAP_GLOBAL_IMPORTANCE_PATH),
        "confusion_matrix": read_csv(CONFUSION_MATRIX_XGB_PATH),
    }


def show_missing_data_warning(data: dict[str, pd.DataFrame]) -> bool:
    required = {
        "features": USER_FEATURES_PATH,
        "predictions": PREDICTIONS_PATH,
        "top_drivers": TOP_DRIVERS_PATH,
        "alerts": ALERTS_PATH,
        "recommendations": RECOMMENDATIONS_PATH,
        "sessions": SESSIONS_PATH,
    }

    missing = [str(path) for key, path in required.items() if data[key].empty]
    if missing:
        st.error("Faltan archivos necesarios para el dashboard.")
        st.code("\n".join(missing))
        st.info(
            "Ejecuta primero: `generate_data.py`, `build_features.py`, "
            "`train_model.py`, `explain_model.py` y `recommend_intervention.py`."
        )
        return True
    return False


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------

def format_pct(value: float) -> str:
    return f"{100 * value:.1f}%"


def format_number(value: float | int) -> str:
    return f"{value:,.0f}".replace(",", ".")


def format_currency(value: float | int) -> str:
    return f"{value:,.0f} €".replace(",", ".")


def risk_badge(label: str) -> str:
    label_es = RISK_LABEL_ES.get(label, label)
    return label_es


def get_user_sessions(sessions: pd.DataFrame, user_id: str) -> pd.DataFrame:
    user_sessions = sessions[sessions["user_id"] == user_id].copy()
    if user_sessions.empty:
        return user_sessions
    user_sessions["session_start"] = pd.to_datetime(user_sessions["session_start"])
    user_sessions["session_date"] = user_sessions["session_start"].dt.date
    user_sessions["week_start"] = user_sessions["session_start"].dt.to_period("W").dt.start_time.dt.date
    return user_sessions.sort_values("session_start")


def format_behaviour_value(feature: str, value: object) -> str:
    if pd.isna(value):
        return "-"
    if feature in CURRENCY_FEATURES:
        return format_currency(float(value))
    if feature in RATIO_FEATURES:
        return f"{float(value):.2f}"
    if feature == "cancelled_limit_recently":
        return "Sí" if int(value) == 1 else "No"
    return f"{float(value):.0f}" if isinstance(value, (int, float)) else str(value)


def show_governance_note() -> None:
    st.info(
        "SafePlay AI es un prototipo de apoyo a revisión preventiva. "
        "No diagnostica ludopatía, no impone sanciones automáticas y no debe usarse "
        "para optimizar campañas comerciales sobre usuarios vulnerables."
    )


# -----------------------------------------------------------------------------
# Page 1: Executive Overview
# -----------------------------------------------------------------------------

def page_executive_overview(data: dict[str, pd.DataFrame]) -> None:
    features = data["features"]
    predictions = data["predictions"]
    alerts = data["alerts"]
    recommendations = data["recommendations"]
    shap_importance = data["shap_importance"]

    st.title("🛡️ SafePlay AI")
    st.caption("Sistema explicable para detectar patrones tempranos de juego de riesgo y recomendar intervenciones responsables.")
    show_governance_note()

    total_users = len(predictions)
    risk_counts = predictions["predicted_risk_label"].value_counts().reindex(RISK_ORDER, fill_value=0)
    high_share = risk_counts["high"] / total_users if total_users else 0
    medium_share = risk_counts["medium"] / total_users if total_users else 0
    low_share = risk_counts["low"] / total_users if total_users else 0

    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("Usuarios analizados", format_number(total_users))
    col2.metric("Riesgo bajo", format_pct(low_share))
    col3.metric("Riesgo medio", format_pct(medium_share))
    col4.metric("Riesgo alto", format_pct(high_share))
    col5.metric("Alertas operativas", format_number(len(alerts)))

    st.divider()

    col_left, col_right = st.columns([1.1, 1])

    with col_left:
        st.subheader("Distribución de riesgo")
        risk_df = risk_counts.reset_index()
        risk_df.columns = ["risk", "users"]
        risk_df["risk_es"] = risk_df["risk"].map(RISK_LABEL_ES)
        fig = px.bar(
            risk_df,
            x="risk_es",
            y="users",
            text="users",
            color="risk_es",
            color_discrete_map=RISK_COLOR_MAP,
            labels={"risk_es": "Nivel de riesgo", "users": "Usuarios"},
        )
        fig.update_layout(showlegend=False, height=380)
        st.plotly_chart(fig, width="stretch", config=PLOTLY_CONFIG)

    with col_right:
        st.subheader("Acciones recomendadas")
        action_counts = recommendations["recommended_action_label"].value_counts().reset_index()
        action_counts.columns = ["acción", "usuarios"]
        fig = px.pie(
            action_counts,
            names="acción",
            values="usuarios",
            hole=0.45,
        )
        fig.update_layout(height=380)
        st.plotly_chart(fig, width="stretch", config=PLOTLY_CONFIG)

    col_left, col_right = st.columns([1.1, 1])

    with col_left:
        st.subheader("Canales con más alertas")
        if "channel_preference" in alerts.columns:
            channel_df = (
                alerts.groupby(["channel_preference", "risk_level_es"])
                .size()
                .reset_index(name="alerts")
            )
            fig = px.bar(
                channel_df,
                x="channel_preference",
                y="alerts",
                color="risk_level_es",
                barmode="group",
                color_discrete_map=RISK_COLOR_MAP,
                labels={"channel_preference": "Canal", "alerts": "Alertas", "risk_level_es": "Riesgo"},
            )
            fig.update_layout(height=380)
            st.plotly_chart(fig, width="stretch", config=PLOTLY_CONFIG)
        else:
            st.warning("No se encontró la columna `channel_preference` en alerts.csv.")

    with col_right:
        st.subheader("Principales drivers globales")
        if not shap_importance.empty:
            top_drivers = shap_importance.head(10).sort_values("mean_abs_shap_high", ascending=True)
            fig = px.bar(
                top_drivers,
                x="mean_abs_shap_high",
                y="driver_label",
                orientation="h",
                labels={"mean_abs_shap_high": "Impacto medio SHAP", "driver_label": "Driver"},
            )
            fig.update_layout(height=380, margin=dict(l=10, r=10, t=20, b=20))
            st.plotly_chart(fig, width="stretch", config=PLOTLY_CONFIG)
        else:
            st.warning("No se encontró shap_global_importance_high.csv.")

    st.subheader("Resumen de intervención")
    summary_cols = [
        "predicted_risk_label",
        "recommended_action_label",
        "users",
        "human_reviews",
        "commercial_suppressions",
        "audit_records",
        "user_share",
    ]
    available_cols = [col for col in summary_cols if col in data["intervention_summary"].columns]
    if available_cols:
        summary_display = data["intervention_summary"][available_cols].copy()
        summary_display["risk_order"] = summary_display["predicted_risk_label"].map({"low": 0, "medium": 1, "high": 2})
        summary_display = summary_display.sort_values("risk_order").drop(columns=["risk_order"])
        summary_display["predicted_risk_label"] = summary_display["predicted_risk_label"].map(RISK_LABEL_ES)
        summary_display["user_share"] = (summary_display["user_share"] * 100).round(2).astype(str) + "%"
        summary_display = summary_display.rename(
            columns={
                "predicted_risk_label": "Nivel de riesgo",
                "recommended_action_label": "Acción recomendada",
                "users": "Usuarios",
                "human_reviews": "Revisiones humanas",
                "commercial_suppressions": "Supresiones comerciales",
                "audit_records": "Registros de auditoría",
                "user_share": "% usuarios",
            }
        )
        st.dataframe(summary_display, width="stretch", hide_index=True)
    else:
        st.dataframe(data["intervention_summary"], width="stretch", hide_index=True)


# -----------------------------------------------------------------------------
# Page 2: Alert Monitor
# -----------------------------------------------------------------------------

def page_alert_monitor(data: dict[str, pd.DataFrame]) -> None:
    alerts = data["alerts"].copy()

    st.title("🚨 Alert Monitor")
    st.caption("Cola operativa para revisión preventiva y acciones proporcionales.")
    show_governance_note()

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Alertas", format_number(len(alerts)))
    col2.metric("Revisión humana", format_number(alerts["human_review_required"].sum()))
    col3.metric("Supresión comercial", format_number(alerts["commercial_suppression"].sum()))
    col4.metric("Registros de auditoría", format_number(alerts["audit_required"].sum()))

    st.divider()

    col_filter_1, col_filter_2, col_filter_3 = st.columns(3)

    with col_filter_1:
        selected_risk = st.multiselect(
            "Nivel de riesgo",
            options=["Alto", "Medio", "Bajo"],
            default=["Alto", "Medio"],
        )

    with col_filter_2:
        channels = sorted(alerts["channel_preference"].dropna().unique().tolist()) if "channel_preference" in alerts.columns else []
        selected_channels = st.multiselect("Canal", options=channels, default=channels)

    with col_filter_3:
        min_score = st.slider("Score mínimo alto riesgo", 0.0, 1.0, 0.0, 0.01)

    filtered = alerts.copy()
    if selected_risk:
        filtered = filtered[filtered["risk_level_es"].isin(selected_risk)]
    if selected_channels:
        filtered = filtered[filtered["channel_preference"].isin(selected_channels)]
    if "risk_score_high" in filtered.columns:
        filtered = filtered[filtered["risk_score_high"] >= min_score]

    st.subheader("Alertas priorizadas")

    display_cols = [
        "user_id",
        "risk_level_es",
        "risk_score_high",
        "operational_priority_score",
        "channel_preference",
        "main_driver",
        "recommended_action_label",
        "human_review_required",
        "commercial_suppression",
        "alert_reason",
    ]
    available_cols = [col for col in display_cols if col in filtered.columns]

    st.dataframe(
        filtered[available_cols].head(500),
        width="stretch",
        hide_index=True,
    )

    st.caption(
        "La tabla muestra las primeras 500 alertas filtradas. El CSV completo se genera en `data/processed/alerts.csv`."
    )


# -----------------------------------------------------------------------------
# Page 3: User Profile
# -----------------------------------------------------------------------------

def page_user_profile(data: dict[str, pd.DataFrame]) -> None:
    predictions = data["predictions"]
    recommendations = data["recommendations"]
    top_drivers = data["top_drivers"]
    sessions = data["sessions"]
    features = data["features"]

    st.title("👤 User Profile")
    st.caption("Vista individual para revisión preventiva explicable.")

    high_users = predictions[predictions["predicted_risk_label"] == "high"].sort_values(
        "risk_score_high", ascending=False
    )["user_id"].tolist()
    medium_users = predictions[predictions["predicted_risk_label"] == "medium"].sort_values(
        "risk_score_medium", ascending=False
    )["user_id"].tolist()
    other_users = predictions[~predictions["user_id"].isin(high_users + medium_users)]["user_id"].tolist()

    ordered_users = high_users + medium_users + other_users

    selected_user = st.selectbox(
        "Selecciona usuario",
        options=ordered_users,
        index=0 if ordered_users else None,
    )

    if not selected_user:
        st.warning("No hay usuarios disponibles.")
        return

    pred_row = predictions[predictions["user_id"] == selected_user].iloc[0]
    rec_row = recommendations[recommendations["user_id"] == selected_user].iloc[0]
    feature_row = features[features["user_id"] == selected_user].iloc[0]
    user_sessions = get_user_sessions(sessions, selected_user)
    user_drivers = top_drivers[top_drivers["user_id"] == selected_user].sort_values("driver_rank")

    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("Usuario", selected_user)
    col2.metric("Riesgo", risk_badge(pred_row["predicted_risk_label"]))
    col3.metric("Score high", f"{pred_row['risk_score_high']:.3f}")
    col4.metric("Anomalía", "Sí" if int(pred_row.get("anomaly_flag", 0)) == 1 else "No")
    col5.metric("Revisión humana", "Sí" if int(rec_row.get("human_review_required", 0)) == 1 else "No")

    st.info(str(rec_row["action_detail"]))

    st.subheader("Principales drivers de la alerta")
    if not user_drivers.empty:
        driver_display = user_drivers[["driver_rank", "driver_label", "shap_value_high"]].copy()
        driver_display.columns = ["Ranking", "Driver", "Impacto SHAP hacia riesgo alto"]
        st.dataframe(driver_display, width="stretch", hide_index=True)
    else:
        st.warning("No hay drivers disponibles para este usuario.")

    st.divider()

    if user_sessions.empty:
        st.warning("Este usuario no tiene sesiones registradas.")
        return

    daily = (
        user_sessions.groupby("session_date")
        .agg(
            amount_wagered=("amount_wagered", "sum"),
            deposit_amount=("deposit_amount", "sum"),
            net_loss=("net_loss", "sum"),
            sessions=("session_id", "count"),
            night_sessions=("night_session", "sum"),
        )
        .reset_index()
    )

    weekly = (
        user_sessions.groupby("week_start")
        .agg(
            sessions=("session_id", "count"),
            amount_wagered=("amount_wagered", "sum"),
            net_loss=("net_loss", "sum"),
        )
        .reset_index()
    )

    col_left, col_right = st.columns(2)

    with col_left:
        st.subheader("Apuestas por día")
        fig = px.line(
            daily,
            x="session_date",
            y="amount_wagered",
            markers=True,
            labels={"session_date": "Fecha", "amount_wagered": "Importe apostado"},
        )
        fig.update_layout(height=340)
        st.plotly_chart(fig, width="stretch", config=PLOTLY_CONFIG)

    with col_right:
        st.subheader("Depósitos por día")
        fig = px.bar(
            daily,
            x="session_date",
            y="deposit_amount",
            labels={"session_date": "Fecha", "deposit_amount": "Depósitos"},
        )
        fig.update_layout(height=340)
        st.plotly_chart(fig, width="stretch", config=PLOTLY_CONFIG)

    col_left, col_right = st.columns(2)

    with col_left:
        st.subheader("Sesiones por semana")
        fig = px.bar(
            weekly,
            x="week_start",
            y="sessions",
            labels={"week_start": "Semana", "sessions": "Sesiones"},
        )
        fig.update_layout(height=340)
        st.plotly_chart(fig, width="stretch", config=PLOTLY_CONFIG)

    with col_right:
        st.subheader("Pérdida neta acumulada")
        daily_loss = daily.copy()
        daily_loss["cumulative_net_loss"] = daily_loss["net_loss"].cumsum()
        fig = px.line(
            daily_loss,
            x="session_date",
            y="cumulative_net_loss",
            markers=True,
            labels={"session_date": "Fecha", "cumulative_net_loss": "Pérdida neta acumulada"},
        )
        fig.update_layout(height=340)
        st.plotly_chart(fig, width="stretch", config=PLOTLY_CONFIG)

    st.subheader("Resumen conductual del usuario")
    behaviour_cols = [
        "sessions_7d",
        "sessions_30d",
        "frequency_increase_ratio",
        "stake_increase_ratio",
        "deposit_amount_7d",
        "deposit_count_7d",
        "net_loss_7d",
        "net_loss_30d",
        "night_sessions_ratio_30d",
        "loss_chasing_events_30d",
        "avg_session_duration_30d",
        "max_session_duration_30d",
        "cancelled_limit_recently",
    ]
    available_cols = [col for col in behaviour_cols if col in features.columns]
    behaviour = pd.DataFrame(
        {
            "Métrica": [BEHAVIOUR_LABELS.get(col, col) for col in available_cols],
            "Valor": [format_behaviour_value(col, feature_row[col]) for col in available_cols],
        }
    )
    st.dataframe(behaviour, width="stretch", hide_index=True)


# -----------------------------------------------------------------------------
# Page 4: Model Governance
# -----------------------------------------------------------------------------

def page_model_governance(data: dict[str, pd.DataFrame]) -> None:
    model_comparison = data["model_comparison"]
    shap_importance = data["shap_importance"]
    confusion_matrix = data["confusion_matrix"]

    st.title("📋 Model Governance")
    st.caption("Documentación operativa, limitaciones y controles de uso responsable.")
    show_governance_note()

    st.subheader("Uso previsto")
    st.write(
        "SafePlay AI está diseñado como sistema de apoyo para priorizar revisiones preventivas "
        "de usuarios con patrones potencialmente compatibles con señales tempranas de juego de riesgo."
    )

    st.subheader("Usos no previstos")
    st.markdown(
        """
        - Diagnóstico clínico.
        - Sanciones automáticas.
        - Bloqueos automáticos sin revisión humana.
        - Segmentación comercial agresiva.
        - Decisiones regulatorias sin validación experta.
        """
    )

    st.subheader("Comparativa de modelos")
    if not model_comparison.empty:
        st.dataframe(model_comparison, width="stretch", hide_index=True)
    else:
        st.warning("No se encontró model_comparison.csv.")

    st.subheader("Matriz de confusión - XGBoost")
    if not confusion_matrix.empty:
        st.dataframe(confusion_matrix, width="stretch", hide_index=True)
    else:
        st.warning("No se encontró confusion_matrix_xgboost.csv.")

    st.subheader("Variables principales según SHAP")
    if not shap_importance.empty:
        st.dataframe(shap_importance.head(20), width="stretch", hide_index=True)
    else:
        st.warning("No se encontró shap_global_importance_high.csv.")

    if SHAP_SUMMARY_PATH.exists():
        st.subheader("SHAP summary plot")
        st.image(str(SHAP_SUMMARY_PATH), caption="Impacto global de variables sobre la clase de riesgo alto")

    col_left, col_right = st.columns(2)

    with col_left:
        st.subheader("Variables usadas")
        st.markdown(
            """
            - Frecuencia de sesiones.
            - Importe apostado.
            - Pérdidas netas.
            - Depósitos recientes.
            - Duración de sesiones.
            - Actividad nocturna.
            - Retornos tras pérdidas.
            - Cambios frente al histórico.
            - Uso de herramientas de protección.
            """
        )

    with col_right:
        st.subheader("Variables excluidas")
        st.markdown(
            """
            - Nombre.
            - Email.
            - DNI.
            - Dirección.
            - Datos personales reales.
            - `synthetic_profile`.
            - `risk_score_rule`.
            - Flags que codifican directamente la etiqueta sintética.
            """
        )

    st.subheader("Limitaciones")
    st.markdown(
        """
        1. Los datos son sintéticos y no representan población real.
        2. Las etiquetas se generan con una regla transparente de prototipo.
        3. El rendimiento alto es esperable porque el objetivo es validar el pipeline completo.
        4. En producción, las etiquetas deberían validarse con expertos, datos históricos reales y criterios regulatorios internos.
        5. El score no debe interpretarse como diagnóstico clínico.
        6. Las decisiones sensibles requieren revisión humana y trazabilidad.
        """
    )

    st.subheader("Política de revisión humana")
    st.write(
        "Los usuarios clasificados como alto riesgo requieren revisión humana antes de cualquier acción sensible. "
        "Los usuarios de riesgo medio reciben intervenciones ligeras y proporcionales, como mensajes preventivos "
        "o sugerencias de límites voluntarios."
    )


# -----------------------------------------------------------------------------
# Main app
# -----------------------------------------------------------------------------

def main() -> None:
    data = load_all_data()

    with st.sidebar:
        st.title("SafePlay AI")
        st.caption("Responsible Gaming Intelligence")
        page = st.radio(
            "Navegación",
            options=[
                "Executive Overview",
                "Alert Monitor",
                "User Profile",
                "Model Governance",
            ],
        )
        st.divider()
        st.caption("MVP construido con datos sintéticos.")

    if show_missing_data_warning(data):
        return

    if page == "Executive Overview":
        page_executive_overview(data)
    elif page == "Alert Monitor":
        page_alert_monitor(data)
    elif page == "User Profile":
        page_user_profile(data)
    elif page == "Model Governance":
        page_model_governance(data)


if __name__ == "__main__":
    main()
