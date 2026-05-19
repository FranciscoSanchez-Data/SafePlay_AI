"""
SafePlay AI - Synthetic Data Generator

Generates realistic synthetic users and gaming sessions for a responsible gaming
risk detection MVP.

Outputs:
- data/synthetic/users.csv
- data/synthetic/sessions.csv
- data/synthetic/generation_audit_profiles.csv

Important:
- Data is fully synthetic.
- No personal identifiers such as name, DNI, email or address are generated.
- `generation_audit_profiles.csv` is only for validating the simulation logic.
  It must not be used as a model feature.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd


# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------

RANDOM_SEED = 42
N_USERS = 10_000
START_DATE = pd.Timestamp("2025-01-01")
END_DATE = pd.Timestamp("2025-06-30 23:59:59")
OUTPUT_DIR = Path("data/synthetic")

RiskProfile = Literal["low", "medium", "high"]

PROFILE_PROBS: dict[RiskProfile, float] = {
    "low": 0.82,
    "medium": 0.14,
    "high": 0.04,
}

AGE_BANDS = ["18-25", "26-35", "36-45", "46-55", "56-65", "65+"]
PROVINCES = [
    "Murcia",
    "Alicante",
    "Almería",
    "Valencia",
    "Madrid",
    "Barcelona",
    "Málaga",
    "Granada",
]
CHANNELS = ["online", "retail"]
PRODUCTS = ["apuestas", "casino", "bingo", "slots", "poker"]


@dataclass(frozen=True)
class ProfileParams:
    """Behavioural parameters used to simulate each synthetic risk profile."""

    monthly_sessions_mean: float
    monthly_sessions_std: float
    recent_activity_multiplier_mean: float
    recent_activity_multiplier_std: float
    wager_low: float
    wager_high: float
    duration_mean: float
    duration_std: float
    deposit_probability: float
    deposit_ratio_low: float
    deposit_ratio_high: float
    night_probability: float
    after_loss_return_probability: float
    limit_config_probability: float
    marketing_opt_in_probability: float
    cancelled_limit_probability: float
    self_excluded_probability: float
    loss_rate_mean: float
    loss_rate_std: float


PROFILE_PARAMS: dict[RiskProfile, ProfileParams] = {
    "low": ProfileParams(
        monthly_sessions_mean=2,
        monthly_sessions_std=1.2,
        recent_activity_multiplier_mean=1.00,
        recent_activity_multiplier_std=0.25,
        wager_low=20,
        wager_high=40,
        duration_mean=20,
        duration_std=8,
        deposit_probability=0.18,
        deposit_ratio_low=0.20,
        deposit_ratio_high=0.70,
        night_probability=0.08,
        after_loss_return_probability=0.03,
        limit_config_probability=0.45,
        marketing_opt_in_probability=0.65,
        cancelled_limit_probability=0.01,
        self_excluded_probability=0.002,
        loss_rate_mean=0.08,
        loss_rate_std=0.25,
    ),
    "medium": ProfileParams(
        monthly_sessions_mean=8,
        monthly_sessions_std=3.0,
        recent_activity_multiplier_mean=1.45,
        recent_activity_multiplier_std=0.45,
        wager_low=50,
        wager_high=120,
        duration_mean=45,
        duration_std=18,
        deposit_probability=0.42,
        deposit_ratio_low=0.35,
        deposit_ratio_high=0.95,
        night_probability=0.22,
        after_loss_return_probability=0.16,
        limit_config_probability=0.38,
        marketing_opt_in_probability=0.72,
        cancelled_limit_probability=0.06,
        self_excluded_probability=0.006,
        loss_rate_mean=0.14,
        loss_rate_std=0.35,
    ),
    "high": ProfileParams(
        monthly_sessions_mean=20,
        monthly_sessions_std=7.0,
        recent_activity_multiplier_mean=2.20,
        recent_activity_multiplier_std=0.65,
        wager_low=150,
        wager_high=500,
        duration_mean=90,
        duration_std=35,
        deposit_probability=0.68,
        deposit_ratio_low=0.45,
        deposit_ratio_high=1.15,
        night_probability=0.42,
        after_loss_return_probability=0.38,
        limit_config_probability=0.34,
        marketing_opt_in_probability=0.76,
        cancelled_limit_probability=0.22,
        self_excluded_probability=0.02,
        loss_rate_mean=0.22,
        loss_rate_std=0.45,
    ),
}


# -----------------------------------------------------------------------------
# Utility functions
# -----------------------------------------------------------------------------


def ensure_output_dir() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)



def random_dates(
    rng: np.random.Generator,
    start: pd.Timestamp,
    end: pd.Timestamp,
    n: int,
) -> pd.Series:
    """Generate n random timestamps between start and end."""
    if n <= 0:
        return pd.Series([], dtype="datetime64[ns]")

    start_ns = start.value
    end_ns = end.value
    random_ns = rng.integers(start_ns, end_ns, size=n, endpoint=True)
    return pd.to_datetime(random_ns).to_series(index=range(n))



def apply_night_hour(timestamp: pd.Timestamp, is_night: bool, rng: np.random.Generator) -> pd.Timestamp:
    """Adjust timestamp hour to match night/non-night behaviour."""
    if is_night:
        hour = int(rng.choice([0, 1, 2, 3, 22, 23], p=[0.16, 0.14, 0.10, 0.08, 0.24, 0.28]))
    else:
        hour = int(rng.choice(range(8, 22)))

    minute = int(rng.integers(0, 60))
    second = int(rng.integers(0, 60))
    return timestamp.replace(hour=hour, minute=minute, second=second)



def positive_normal(
    rng: np.random.Generator,
    mean: float,
    std: float,
    min_value: float,
) -> float:
    """Draw a positive value from a normal distribution with a floor."""
    return float(max(rng.normal(mean, std), min_value))



def sample_profile(rng: np.random.Generator, n_users: int) -> np.ndarray:
    profiles = list(PROFILE_PROBS.keys())
    probs = list(PROFILE_PROBS.values())
    return rng.choice(profiles, size=n_users, p=probs)


# -----------------------------------------------------------------------------
# Users
# -----------------------------------------------------------------------------


def generate_users(rng: np.random.Generator) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Generate synthetic users and a separate audit profile table."""
    user_ids = [f"U{i:06d}" for i in range(1, N_USERS + 1)]
    profiles = sample_profile(rng, N_USERS)

    registration_dates = random_dates(
        rng=rng,
        start=pd.Timestamp("2023-01-01"),
        end=START_DATE - pd.Timedelta(days=1),
        n=N_USERS,
    ).dt.date

    user_rows = []
    audit_rows = []

    for user_id, profile, registration_date in zip(user_ids, profiles, registration_dates):
        params = PROFILE_PARAMS[profile]
        preferred_channel = str(rng.choice(CHANNELS, p=[0.72, 0.28]))

        limit_configured = int(rng.random() < params.limit_config_probability)
        cancelled_limit_recently = int(limit_configured and rng.random() < params.cancelled_limit_probability)

        # Self-exclusion is rare and synthetic. It is included to test governance logic.
        self_excluded = int(rng.random() < params.self_excluded_probability)

        user_rows.append(
            {
                "user_id": user_id,
                "age_band": str(rng.choice(AGE_BANDS, p=[0.16, 0.29, 0.23, 0.17, 0.10, 0.05])),
                "province": str(rng.choice(PROVINCES)),
                "registration_date": registration_date,
                "channel_preference": preferred_channel,
                "limit_configured": limit_configured,
                "marketing_opt_in": int(rng.random() < params.marketing_opt_in_probability),
                "self_excluded": self_excluded,
            }
        )

        audit_rows.append(
            {
                "user_id": user_id,
                "synthetic_profile": profile,
                "cancelled_limit_recently": cancelled_limit_recently,
            }
        )

    users = pd.DataFrame(user_rows)
    audit_profiles = pd.DataFrame(audit_rows)
    return users, audit_profiles


# -----------------------------------------------------------------------------
# Sessions
# -----------------------------------------------------------------------------


def expected_sessions_for_period(
    rng: np.random.Generator,
    profile: RiskProfile,
    n_days: int,
    recent: bool,
) -> int:
    """Estimate the number of sessions for a profile and time window."""
    params = PROFILE_PARAMS[profile]
    months = n_days / 30.0

    base_monthly_sessions = positive_normal(
        rng=rng,
        mean=params.monthly_sessions_mean,
        std=params.monthly_sessions_std,
        min_value=0.2,
    )

    if recent:
        multiplier = positive_normal(
            rng=rng,
            mean=params.recent_activity_multiplier_mean,
            std=params.recent_activity_multiplier_std,
            min_value=0.2,
        )
        expected = base_monthly_sessions * months * multiplier
    else:
        expected = base_monthly_sessions * months

    return int(max(rng.poisson(expected), 0))



def generate_session_financials(
    rng: np.random.Generator,
    profile: RiskProfile,
    recent: bool,
) -> dict[str, float | int]:
    """Generate wager, win, loss, bets and deposits for a single session."""
    params = PROFILE_PARAMS[profile]

    amount_wagered = float(rng.uniform(params.wager_low, params.wager_high))

    # Recent escalation: medium/high users may increase stakes in the recent window.
    if recent and profile in {"medium", "high"}:
        amount_wagered *= float(rng.uniform(1.05, 1.75 if profile == "medium" else 2.30))

    amount_wagered = round(amount_wagered, 2)

    # Net loss can be negative when the user wins. The distribution is noisy by design.
    loss_rate = rng.normal(params.loss_rate_mean, params.loss_rate_std)
    net_loss = round(amount_wagered * loss_rate, 2)

    # Keep extreme wins/losses plausible for a synthetic demo.
    net_loss = float(np.clip(net_loss, -0.90 * amount_wagered, 1.00 * amount_wagered))
    amount_won = round(amount_wagered - net_loss, 2)
    amount_won = max(amount_won, 0.0)

    avg_bet_target = {
        "low": rng.uniform(5, 15),
        "medium": rng.uniform(10, 30),
        "high": rng.uniform(20, 80),
    }[profile]
    num_bets = int(max(round(amount_wagered / avg_bet_target), 1))
    avg_bet_amount = round(amount_wagered / num_bets, 2)

    has_deposit = rng.random() < params.deposit_probability
    deposit_count = 0
    deposit_amount = 0.0

    if has_deposit:
        deposit_count = int(rng.choice([1, 1, 1, 2, 2, 3] if profile == "high" else [1, 1, 1, 2]))
        deposit_amount = round(
            amount_wagered * float(rng.uniform(params.deposit_ratio_low, params.deposit_ratio_high)),
            2,
        )

        if recent and profile in {"medium", "high"}:
            deposit_amount = round(deposit_amount * float(rng.uniform(1.05, 1.80)), 2)

    return {
        "amount_wagered": amount_wagered,
        "amount_won": amount_won,
        "net_loss": round(net_loss, 2),
        "num_bets": num_bets,
        "avg_bet_amount": avg_bet_amount,
        "deposit_amount": deposit_amount,
        "deposit_count": deposit_count,
    }



def generate_sessions_for_user(
    rng: np.random.Generator,
    user_id: str,
    profile: RiskProfile,
    channel_preference: str,
    session_id_start: int,
) -> tuple[list[dict], int]:
    """Generate all sessions for a single synthetic user."""
    params = PROFILE_PARAMS[profile]
    rows: list[dict] = []
    session_id_counter = session_id_start

    recent_window_days = 30
    recent_start = END_DATE - pd.Timedelta(days=recent_window_days - 1)

    historical_days = (recent_start.normalize() - START_DATE.normalize()).days
    recent_days = recent_window_days

    n_historical = expected_sessions_for_period(rng, profile, historical_days, recent=False)
    n_recent = expected_sessions_for_period(rng, profile, recent_days, recent=True)

    timestamps_historical = random_dates(rng, START_DATE, recent_start - pd.Timedelta(seconds=1), n_historical)
    timestamps_recent = random_dates(rng, recent_start, END_DATE, n_recent)

    timestamp_records = [(ts, False) for ts in timestamps_historical] + [(ts, True) for ts in timestamps_recent]
    timestamp_records = sorted(timestamp_records, key=lambda x: x[0])

    previous_loss_session_time: pd.Timestamp | None = None

    for timestamp, recent in timestamp_records:
        is_night = bool(rng.random() < params.night_probability)
        session_start = apply_night_hour(pd.Timestamp(timestamp), is_night, rng)

        # Prefer the user's main channel but allow channel switching.
        if rng.random() < 0.82:
            channel = channel_preference
        else:
            channel = "retail" if channel_preference == "online" else "online"

        if profile == "low":
            product_probs = [0.55, 0.12, 0.12, 0.16, 0.05]
        elif profile == "medium":
            product_probs = [0.42, 0.20, 0.08, 0.24, 0.06]
        else:
            product_probs = [0.34, 0.24, 0.04, 0.32, 0.06]

        product_type = str(rng.choice(PRODUCTS, p=product_probs))

        duration = int(round(positive_normal(rng, params.duration_mean, params.duration_std, min_value=5)))
        if recent and profile in {"medium", "high"}:
            duration = int(round(duration * float(rng.uniform(1.00, 1.45))))

        financials = generate_session_financials(rng, profile, recent)

        after_loss_return = 0
        if previous_loss_session_time is not None:
            hours_since_loss = (session_start - previous_loss_session_time).total_seconds() / 3600
            if 0 < hours_since_loss <= 24 and rng.random() < params.after_loss_return_probability:
                after_loss_return = 1

        if financials["net_loss"] > 0:
            previous_loss_session_time = session_start

        rows.append(
            {
                "session_id": f"S{session_id_counter:08d}",
                "user_id": user_id,
                "session_start": session_start.strftime("%Y-%m-%d %H:%M:%S"),
                "channel": channel,
                "product_type": product_type,
                "session_duration_min": duration,
                "amount_wagered": financials["amount_wagered"],
                "amount_won": financials["amount_won"],
                "net_loss": financials["net_loss"],
                "num_bets": financials["num_bets"],
                "avg_bet_amount": financials["avg_bet_amount"],
                "deposit_amount": financials["deposit_amount"],
                "deposit_count": financials["deposit_count"],
                "night_session": int(is_night),
                "after_loss_return": after_loss_return,
            }
        )
        session_id_counter += 1

    return rows, session_id_counter



def generate_sessions(
    rng: np.random.Generator,
    users: pd.DataFrame,
    audit_profiles: pd.DataFrame,
) -> pd.DataFrame:
    """Generate synthetic gaming sessions for all users."""
    profile_map = audit_profiles.set_index("user_id")["synthetic_profile"].to_dict()

    all_rows: list[dict] = []
    session_id_counter = 1

    for user in users.itertuples(index=False):
        profile = profile_map[user.user_id]
        rows, session_id_counter = generate_sessions_for_user(
            rng=rng,
            user_id=user.user_id,
            profile=profile,
            channel_preference=user.channel_preference,
            session_id_start=session_id_counter,
        )
        all_rows.extend(rows)

    sessions = pd.DataFrame(all_rows)
    return sessions


# -----------------------------------------------------------------------------
# Validation summary
# -----------------------------------------------------------------------------


def build_generation_summary(
    users: pd.DataFrame,
    sessions: pd.DataFrame,
    audit_profiles: pd.DataFrame,
) -> pd.DataFrame:
    """Create a simple profile-level summary to validate synthetic behaviour."""
    user_profiles = users.merge(audit_profiles, on="user_id", how="left")

    session_profile = sessions.merge(
        audit_profiles[["user_id", "synthetic_profile"]],
        on="user_id",
        how="left",
    )

    per_profile_users = user_profiles.groupby("synthetic_profile")["user_id"].nunique()

    summary = (
        session_profile.groupby("synthetic_profile")
        .agg(
            sessions=("session_id", "count"),
            users_with_sessions=("user_id", "nunique"),
            avg_sessions_per_active_user=("session_id", lambda x: len(x) / x.index.to_series().map(session_profile["user_id"]).nunique()),
            avg_wagered=("amount_wagered", "mean"),
            avg_net_loss=("net_loss", "mean"),
            avg_duration=("session_duration_min", "mean"),
            night_ratio=("night_session", "mean"),
            after_loss_return_ratio=("after_loss_return", "mean"),
            avg_deposit_amount=("deposit_amount", "mean"),
        )
        .reset_index()
    )

    summary["total_users"] = summary["synthetic_profile"].map(per_profile_users).astype(int)
    summary["sessions_per_total_user"] = summary["sessions"] / summary["total_users"]

    ordered_cols = [
        "synthetic_profile",
        "total_users",
        "users_with_sessions",
        "sessions",
        "sessions_per_total_user",
        "avg_wagered",
        "avg_net_loss",
        "avg_duration",
        "night_ratio",
        "after_loss_return_ratio",
        "avg_deposit_amount",
    ]
    return summary[ordered_cols].round(3)


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------


def main() -> None:
    ensure_output_dir()
    rng = np.random.default_rng(RANDOM_SEED)

    print("Generating synthetic users...")
    users, audit_profiles = generate_users(rng)

    print("Generating synthetic sessions...")
    sessions = generate_sessions(rng, users, audit_profiles)

    users_path = OUTPUT_DIR / "users.csv"
    sessions_path = OUTPUT_DIR / "sessions.csv"
    audit_path = OUTPUT_DIR / "generation_audit_profiles.csv"
    summary_path = OUTPUT_DIR / "generation_summary.csv"

    users.to_csv(users_path, index=False)
    sessions.to_csv(sessions_path, index=False)
    audit_profiles.to_csv(audit_path, index=False)

    summary = build_generation_summary(users, sessions, audit_profiles)
    summary.to_csv(summary_path, index=False)

    print("\nSynthetic data generated successfully.")
    print(f"Users:    {len(users):,} -> {users_path}")
    print(f"Sessions: {len(sessions):,} -> {sessions_path}")
    print(f"Audit:    {len(audit_profiles):,} -> {audit_path}")
    print(f"Summary:  {summary_path}")
    print("\nGeneration summary:")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
