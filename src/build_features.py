"""
SafePlay AI - Feature Engineering Pipeline

Reads synthetic users and sessions, then builds one modelling row per user.

Inputs:
- data/synthetic/users.csv
- data/synthetic/sessions.csv
- data/synthetic/generation_audit_profiles.csv optional, only to recover
  `cancelled_limit_recently` generated in phase 1.

Outputs:
- data/processed/user_features.csv
- data/processed/feature_summary.csv

Notes:
- `synthetic_profile` is never used as a model feature.
- `risk_label` is generated with a transparent synthetic rule for MVP purposes.
- In production, labels must be validated with domain experts, historical data and
  internal responsible gaming criteria.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------

SYNTHETIC_DIR = Path("data/synthetic")
PROCESSED_DIR = Path("data/processed")

USERS_PATH = SYNTHETIC_DIR / "users.csv"
SESSIONS_PATH = SYNTHETIC_DIR / "sessions.csv"
AUDIT_PROFILES_PATH = SYNTHETIC_DIR / "generation_audit_profiles.csv"

USER_FEATURES_PATH = PROCESSED_DIR / "user_features.csv"
FEATURE_SUMMARY_PATH = PROCESSED_DIR / "feature_summary.csv"

EPSILON = 1e-6
MAX_RATIO = 10.0


# -----------------------------------------------------------------------------
# Utility functions
# -----------------------------------------------------------------------------


def ensure_output_dir() -> None:
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)



def safe_ratio(numerator: pd.Series, denominator: pd.Series, max_value: float = MAX_RATIO) -> pd.Series:
    """Compute a stable capped ratio."""
    ratio = numerator / (denominator + EPSILON)
    ratio = ratio.replace([np.inf, -np.inf], np.nan).fillna(0)
    return ratio.clip(lower=0, upper=max_value)



def aggregate_window(
    sessions: pd.DataFrame,
    users: pd.DataFrame,
    start: pd.Timestamp,
    end: pd.Timestamp,
    suffix: str,
) -> pd.DataFrame:
    """Aggregate session behaviour for a given time window."""
    mask = (sessions["session_start"] > start) & (sessions["session_start"] <= end)
    window = sessions.loc[mask].copy()

    base = pd.DataFrame({"user_id": users["user_id"]})

    if window.empty:
        empty_cols = {
            f"sessions_{suffix}": 0,
            f"active_days_{suffix}": 0,
            f"total_wagered_{suffix}": 0.0,
            f"net_loss_{suffix}": 0.0,
            f"avg_bet_amount_{suffix}": 0.0,
            f"avg_wagered_per_session_{suffix}": 0.0,
            f"avg_session_duration_{suffix}": 0.0,
            f"max_session_duration_{suffix}": 0.0,
            f"night_sessions_ratio_{suffix}": 0.0,
            f"deposit_amount_{suffix}": 0.0,
            f"deposit_count_{suffix}": 0,
            f"loss_chasing_events_{suffix}": 0,
            f"product_switch_count_{suffix}": 0,
            f"channel_switch_count_{suffix}": 0,
        }
        return base.assign(**empty_cols)

    window["session_date"] = window["session_start"].dt.date

    agg = (
        window.groupby("user_id")
        .agg(
            **{
                f"sessions_{suffix}": ("session_id", "count"),
                f"active_days_{suffix}": ("session_date", "nunique"),
                f"total_wagered_{suffix}": ("amount_wagered", "sum"),
                f"net_loss_{suffix}": ("net_loss", "sum"),
                f"avg_bet_amount_{suffix}": ("avg_bet_amount", "mean"),
                f"avg_wagered_per_session_{suffix}": ("amount_wagered", "mean"),
                f"avg_session_duration_{suffix}": ("session_duration_min", "mean"),
                f"max_session_duration_{suffix}": ("session_duration_min", "max"),
                f"night_sessions_ratio_{suffix}": ("night_session", "mean"),
                f"deposit_amount_{suffix}": ("deposit_amount", "sum"),
                f"deposit_count_{suffix}": ("deposit_count", "sum"),
                f"loss_chasing_events_{suffix}": ("after_loss_return", "sum"),
                f"product_switch_count_{suffix}": ("product_type", "nunique"),
                f"channel_switch_count_{suffix}": ("channel", "nunique"),
            }
        )
        .reset_index()
    )

    # Convert unique counts into switch counts. If only one product/channel appears,
    # there are zero switches at aggregate level.
    agg[f"product_switch_count_{suffix}"] = (agg[f"product_switch_count_{suffix}"] - 1).clip(lower=0)
    agg[f"channel_switch_count_{suffix}"] = (agg[f"channel_switch_count_{suffix}"] - 1).clip(lower=0)

    return base.merge(agg, on="user_id", how="left").fillna(0)



def add_days_since_last_session(features: pd.DataFrame, sessions: pd.DataFrame, cutoff_date: pd.Timestamp) -> pd.DataFrame:
    last_session = sessions.groupby("user_id")["session_start"].max().reset_index(name="last_session_start")
    features = features.merge(last_session, on="user_id", how="left")
    features["days_since_last_session"] = (
        cutoff_date - features["last_session_start"]
    ).dt.total_seconds() / 86_400
    features["days_since_last_session"] = features["days_since_last_session"].fillna(999).clip(lower=0)
    features = features.drop(columns=["last_session_start"])
    return features


# -----------------------------------------------------------------------------
# Loading
# -----------------------------------------------------------------------------


def load_inputs() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame | None]:
    if not USERS_PATH.exists():
        raise FileNotFoundError(f"Missing input file: {USERS_PATH}")
    if not SESSIONS_PATH.exists():
        raise FileNotFoundError(f"Missing input file: {SESSIONS_PATH}")

    users = pd.read_csv(USERS_PATH)
    sessions = pd.read_csv(SESSIONS_PATH)
    sessions["session_start"] = pd.to_datetime(sessions["session_start"])

    audit_profiles = None
    if AUDIT_PROFILES_PATH.exists():
        audit_profiles = pd.read_csv(AUDIT_PROFILES_PATH)

    return users, sessions, audit_profiles


# -----------------------------------------------------------------------------
# Feature engineering
# -----------------------------------------------------------------------------


def build_features(users: pd.DataFrame, sessions: pd.DataFrame, audit_profiles: pd.DataFrame | None) -> pd.DataFrame:
    cutoff_date = sessions["session_start"].max()

    start_7d = cutoff_date - pd.Timedelta(days=7)
    start_14d = cutoff_date - pd.Timedelta(days=14)
    start_30d = cutoff_date - pd.Timedelta(days=30)
    start_hist_90d = cutoff_date - pd.Timedelta(days=97)
    end_hist_90d = start_7d
    start_hist_all = sessions["session_start"].min() - pd.Timedelta(seconds=1)

    features = users.copy()

    # Core temporal windows.
    for start, suffix in [
        (start_7d, "7d"),
        (start_14d, "14d"),
        (start_30d, "30d"),
    ]:
        window_features = aggregate_window(sessions, users, start=start, end=cutoff_date, suffix=suffix)
        features = features.merge(window_features, on="user_id", how="left")

    # Historical windows used only to compute change ratios.
    hist_90 = aggregate_window(
        sessions,
        users,
        start=start_hist_90d,
        end=end_hist_90d,
        suffix="hist_90d_before_7d",
    )
    hist_all = aggregate_window(
        sessions,
        users,
        start=start_hist_all,
        end=start_7d,
        suffix="hist_before_7d",
    )

    features = features.merge(hist_90, on="user_id", how="left")
    features = features.merge(hist_all, on="user_id", how="left")

    features = add_days_since_last_session(features, sessions, cutoff_date)

    # Frequency ratio:
    # sessions last 7 days / average weekly sessions over the prior 90 days.
    historical_weekly_sessions = features["sessions_hist_90d_before_7d"] / (90 / 7)
    features["frequency_increase_ratio"] = safe_ratio(
        features["sessions_7d"],
        historical_weekly_sessions,
    )

    # Stake ratio:
    # average wagered per session over last 7 days / historical average wagered per session.
    features["stake_increase_ratio"] = safe_ratio(
        features["avg_wagered_per_session_7d"],
        features["avg_wagered_per_session_hist_before_7d"],
    )

    # Deposit growth ratio:
    # deposit amount last 7 days / average weekly deposit amount over the prior 90 days.
    historical_weekly_deposit = features["deposit_amount_hist_90d_before_7d"] / (90 / 7)
    features["deposit_increase_ratio"] = safe_ratio(
        features["deposit_amount_7d"],
        historical_weekly_deposit,
    )

    # Loss growth ratio:
    # net loss last 7 days / average weekly positive net loss over the prior 90 days.
    historical_weekly_loss = features["net_loss_hist_90d_before_7d"].clip(lower=0) / (90 / 7)
    recent_loss = features["net_loss_7d"].clip(lower=0)
    features["loss_increase_ratio"] = safe_ratio(recent_loss, historical_weekly_loss)

    # Protection/governance features.
    features["self_exclusion_attempt"] = features["self_excluded"].astype(int)

    # Cooling-off is synthetic and derived conservatively for the MVP.
    # In a real system this should come from responsible gaming event logs.
    features["cooling_off_used"] = (
        (features["self_exclusion_attempt"] == 1)
        | (
            (features["limit_configured"] == 1)
            & (features["sessions_30d"] == 0)
            & (features["sessions_hist_before_7d"] > 0)
        )
    ).astype(int)

    if audit_profiles is not None and "cancelled_limit_recently" in audit_profiles.columns:
        cancelled = audit_profiles[["user_id", "cancelled_limit_recently"]].copy()
        features = features.merge(cancelled, on="user_id", how="left")
        features["cancelled_limit_recently"] = features["cancelled_limit_recently"].fillna(0).astype(int)
    else:
        features["cancelled_limit_recently"] = 0

    # Synthetic transparent label rule.
    features = add_synthetic_risk_labels(features)

    # Keep a clean modelling dataset. Historical helper columns are useful for QA but
    # should not all be necessary in the final app. We keep them for transparency.
    features = features.fillna(0)
    return features


# -----------------------------------------------------------------------------
# Synthetic labels
# -----------------------------------------------------------------------------


def add_synthetic_risk_labels(features: pd.DataFrame) -> pd.DataFrame:
    """Create transparent MVP labels from behavioural flags."""
    high_loss_growth = (
        (features["loss_increase_ratio"] >= 1.75)
        & (features["net_loss_7d"] >= 50)
    ).astype(int)

    high_frequency_growth = (
        (features["frequency_increase_ratio"] >= 1.75)
        & (features["sessions_7d"] >= 3)
    ).astype(int)

    high_deposit_growth = (
        (features["deposit_increase_ratio"] >= 1.75)
        & (features["deposit_amount_7d"] >= 50)
    ).astype(int)

    loss_chasing_events_flag = (features["loss_chasing_events_30d"] >= 2).astype(int)

    high_night_activity = (
        (features["night_sessions_ratio_30d"] >= 0.30)
        & (features["sessions_30d"] >= 3)
    ).astype(int)

    long_sessions_flag = (
        (features["avg_session_duration_30d"] >= 60)
        | (features["max_session_duration_30d"] >= 120)
    ).astype(int)

    cancelled_limit_recently = features["cancelled_limit_recently"].astype(int)

    features["risk_score_rule"] = (
        2.0 * high_loss_growth
        + 1.5 * high_frequency_growth
        + 1.5 * high_deposit_growth
        + 2.0 * loss_chasing_events_flag
        + 1.0 * high_night_activity
        + 1.5 * long_sessions_flag
        + 2.5 * cancelled_limit_recently
    )

    # MVP calibration:
    # - low: no relevant signal or only a weak isolated signal
    # - medium: at least one meaningful early-warning signal
    # - high: accumulated signals requiring human review
    #
    # The original synthetic rule used low < 3 and medium < 6. For this MVP,
    # we lower the medium threshold to make the alert monitor more realistic:
    # a single strong early-warning signal should be monitored as medium risk.
    features["risk_label"] = np.select(
        [
            features["risk_score_rule"] < 1.5,
            features["risk_score_rule"] < 6,
        ],
        ["low", "medium"],
        default="high",
    )

    # Keep flags for model governance and debugging. These should normally be excluded
    # from model training if they directly encode the synthetic rule.
    features["flag_high_loss_growth"] = high_loss_growth
    features["flag_high_frequency_growth"] = high_frequency_growth
    features["flag_high_deposit_growth"] = high_deposit_growth
    features["flag_loss_chasing_events"] = loss_chasing_events_flag
    features["flag_high_night_activity"] = high_night_activity
    features["flag_long_sessions"] = long_sessions_flag

    return features


# -----------------------------------------------------------------------------
# Reporting
# -----------------------------------------------------------------------------


def build_feature_summary(features: pd.DataFrame) -> pd.DataFrame:
    """Create a compact summary by generated risk label."""
    summary = (
        features.groupby("risk_label")
        .agg(
            users=("user_id", "nunique"),
            sessions_7d_mean=("sessions_7d", "mean"),
            sessions_30d_mean=("sessions_30d", "mean"),
            frequency_increase_ratio_mean=("frequency_increase_ratio", "mean"),
            total_wagered_30d_mean=("total_wagered_30d", "mean"),
            net_loss_30d_mean=("net_loss_30d", "mean"),
            deposit_amount_7d_mean=("deposit_amount_7d", "mean"),
            night_sessions_ratio_30d_mean=("night_sessions_ratio_30d", "mean"),
            loss_chasing_events_30d_mean=("loss_chasing_events_30d", "mean"),
            cancelled_limit_recently_rate=("cancelled_limit_recently", "mean"),
            risk_score_rule_mean=("risk_score_rule", "mean"),
        )
        .reset_index()
    )

    summary["user_share"] = summary["users"] / summary["users"].sum()
    return summary.round(4)



def print_quality_checks(features: pd.DataFrame) -> None:
    print("\nFeature quality checks")
    print("----------------------")
    print(f"Users in feature table: {len(features):,}")
    print("Risk label distribution:")
    print(features["risk_label"].value_counts(normalize=False).to_string())
    print("\nRisk label distribution (%):")
    print((features["risk_label"].value_counts(normalize=True) * 100).round(2).to_string())

    required_columns = [
        "sessions_7d",
        "sessions_14d",
        "sessions_30d",
        "active_days_30d",
        "frequency_increase_ratio",
        "days_since_last_session",
        "total_wagered_7d",
        "total_wagered_30d",
        "net_loss_7d",
        "net_loss_30d",
        "avg_bet_amount_30d",
        "stake_increase_ratio",
        "deposit_amount_7d",
        "deposit_count_7d",
        "avg_session_duration_30d",
        "max_session_duration_30d",
        "night_sessions_ratio_30d",
        "loss_chasing_events_30d",
        "product_switch_count_30d",
        "channel_switch_count_30d",
        "limit_configured",
        "cancelled_limit_recently",
        "self_exclusion_attempt",
        "cooling_off_used",
        "risk_label",
    ]
    missing = [col for col in required_columns if col not in features.columns]
    if missing:
        raise ValueError(f"Missing required feature columns: {missing}")

    leakage_columns = ["synthetic_profile"]
    present_leakage = [col for col in leakage_columns if col in features.columns]
    if present_leakage:
        raise ValueError(f"Potential leakage columns found in feature table: {present_leakage}")


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------


def main() -> None:
    ensure_output_dir()

    print("Loading synthetic inputs...")
    users, sessions, audit_profiles = load_inputs()

    print("Building user-level features...")
    features = build_features(users, sessions, audit_profiles)

    feature_summary = build_feature_summary(features)

    features.to_csv(USER_FEATURES_PATH, index=False)
    feature_summary.to_csv(FEATURE_SUMMARY_PATH, index=False)

    print("\nFeature engineering completed successfully.")
    print(f"Features: {USER_FEATURES_PATH}")
    print(f"Summary:  {FEATURE_SUMMARY_PATH}")

    print_quality_checks(features)

    print("\nFeature summary by risk label:")
    print(feature_summary.to_string(index=False))


if __name__ == "__main__":
    main()
