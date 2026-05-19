"""
SafePlay AI - Model Training Pipeline

Trains and compares:
- Logistic Regression baseline
- XGBoost classifier as the main supervised model, if installed
- Isolation Forest as an unsupervised anomaly layer

Inputs:
- data/processed/user_features.csv

Outputs:
- models/risk_model.pkl
- models/preprocessor.pkl
- models/label_mapping.json
- reports/model_comparison.csv
- reports/classification_report_logistic_regression.csv
- reports/classification_report_xgboost.csv, if XGBoost is available
- reports/confusion_matrix_*.csv
- reports/metrics.json
- data/processed/predictions.csv

Important:
- This script excludes synthetic rule columns from model training.
- `risk_score_rule` and `flag_*` columns are not used as model features.
- The objective is not automatic decision-making. The score supports human review.
"""

from __future__ import annotations

import json
import pickle
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import IsolationForest, HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.utils.class_weight import compute_sample_weight

try:
    from xgboost import XGBClassifier

    XGBOOST_AVAILABLE = True
except ImportError:
    XGBClassifier = None
    XGBOOST_AVAILABLE = False


# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------

RANDOM_SEED = 42
PROCESSED_DIR = Path("data/processed")
MODELS_DIR = Path("models")
REPORTS_DIR = Path("reports")

FEATURES_PATH = PROCESSED_DIR / "user_features.csv"
PREDICTIONS_PATH = PROCESSED_DIR / "predictions.csv"

RISK_MODEL_PATH = MODELS_DIR / "risk_model.pkl"
PREPROCESSOR_PATH = MODELS_DIR / "preprocessor.pkl"
LABEL_MAPPING_PATH = MODELS_DIR / "label_mapping.json"
METRICS_PATH = REPORTS_DIR / "metrics.json"
MODEL_COMPARISON_PATH = REPORTS_DIR / "model_comparison.csv"

LABEL_TO_ID = {"low": 0, "medium": 1, "high": 2}
ID_TO_LABEL = {v: k for k, v in LABEL_TO_ID.items()}
HIGH_CLASS_ID = LABEL_TO_ID["high"]
LABEL_ORDER = ["low", "medium", "high"]

# Columns that would leak the synthetic labelling rule or are not model inputs.
LEAKAGE_AND_METADATA_COLUMNS = {
    "user_id",
    "risk_label",
    "risk_score_rule",
    "flag_high_loss_growth",
    "flag_high_frequency_growth",
    "flag_high_deposit_growth",
    "flag_loss_chasing_events",
    "flag_high_night_activity",
    "flag_long_sessions",
}

# Keep these fields out of the first model iteration. They may be useful later,
# but they are not needed to prove the temporal behaviour modelling concept.
OPTIONAL_EXCLUDED_COLUMNS = {
    "registration_date",
}


# -----------------------------------------------------------------------------
# Utility functions
# -----------------------------------------------------------------------------


def ensure_output_dirs() -> None:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)



def make_one_hot_encoder() -> OneHotEncoder:
    """Create OneHotEncoder compatible with multiple sklearn versions."""
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=False)



def load_dataset() -> pd.DataFrame:
    if not FEATURES_PATH.exists():
        raise FileNotFoundError(
            f"Missing input file: {FEATURES_PATH}. Run `python src/build_features.py` first."
        )

    data = pd.read_csv(FEATURES_PATH)
    if "risk_label" not in data.columns:
        raise ValueError("`risk_label` column not found in user_features.csv")

    unknown_labels = set(data["risk_label"].unique()) - set(LABEL_TO_ID.keys())
    if unknown_labels:
        raise ValueError(f"Unexpected labels found: {unknown_labels}")

    return data



def split_features_target(data: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    excluded_columns = LEAKAGE_AND_METADATA_COLUMNS | OPTIONAL_EXCLUDED_COLUMNS
    feature_columns = [col for col in data.columns if col not in excluded_columns]

    X = data[feature_columns].copy()
    y = data["risk_label"].map(LABEL_TO_ID).astype(int)
    user_ids = data["user_id"].copy()

    return X, y, user_ids



def build_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    categorical_features = X.select_dtypes(include=["object", "category", "bool"]).columns.tolist()
    numeric_features = [col for col in X.columns if col not in categorical_features]

    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", make_one_hot_encoder()),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipeline, numeric_features),
            ("categorical", categorical_pipeline, categorical_features),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )



def get_feature_names(preprocessor: ColumnTransformer) -> list[str]:
    """Return transformed feature names when supported by sklearn."""
    try:
        return preprocessor.get_feature_names_out().tolist()
    except Exception:
        return []


# -----------------------------------------------------------------------------
# Models
# -----------------------------------------------------------------------------


def build_logistic_regression_pipeline(preprocessor: ColumnTransformer) -> Pipeline:
    return Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            (
                "model",
                LogisticRegression(
                    max_iter=2_000,
                    class_weight="balanced",
                    multi_class="auto",
                    random_state=RANDOM_SEED,
                    n_jobs=None,
                ),
            ),
        ]
    )



def build_xgboost_pipeline(preprocessor: ColumnTransformer) -> Pipeline:
    if not XGBOOST_AVAILABLE or XGBClassifier is None:
        raise ImportError("XGBoost is not installed. Install it with `pip install xgboost`.")

    return Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            (
                "model",
                XGBClassifier(
                    objective="multi:softprob",
                    num_class=3,
                    eval_metric="mlogloss",
                    n_estimators=350,
                    max_depth=4,
                    learning_rate=0.045,
                    subsample=0.85,
                    colsample_bytree=0.85,
                    reg_lambda=1.5,
                    min_child_weight=3,
                    random_state=RANDOM_SEED,
                    n_jobs=4,
                ),
            ),
        ]
    )



def build_hist_gradient_boosting_pipeline(preprocessor: ColumnTransformer) -> Pipeline:
    """Fallback if XGBoost is unavailable. Not the preferred final model."""
    return Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            (
                "model",
                HistGradientBoostingClassifier(
                    learning_rate=0.06,
                    max_iter=250,
                    l2_regularization=0.1,
                    random_state=RANDOM_SEED,
                ),
            ),
        ]
    )


# -----------------------------------------------------------------------------
# Evaluation
# -----------------------------------------------------------------------------


def predict_proba_safely(model: Pipeline, X: pd.DataFrame) -> np.ndarray:
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X)

    # Fallback for estimators without predict_proba.
    predictions = model.predict(X)
    probabilities = np.zeros((len(predictions), len(LABEL_TO_ID)))
    probabilities[np.arange(len(predictions)), predictions] = 1.0
    return probabilities



def evaluate_model(
    model_name: str,
    model: Pipeline,
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> tuple[dict[str, Any], pd.DataFrame, pd.DataFrame]:
    y_pred = model.predict(X_test)
    y_proba = predict_proba_safely(model, X_test)

    high_true = (y_test == HIGH_CLASS_ID).astype(int)
    high_score = y_proba[:, HIGH_CLASS_ID]

    metrics = {
        "model": model_name,
        "precision_high": float(precision_score(y_test, y_pred, labels=[HIGH_CLASS_ID], average="micro", zero_division=0)),
        "recall_high": float(recall_score(y_test, y_pred, labels=[HIGH_CLASS_ID], average="micro", zero_division=0)),
        "f1_high": float(f1_score(y_test, y_pred, labels=[HIGH_CLASS_ID], average="micro", zero_division=0)),
        "pr_auc_high": float(average_precision_score(high_true, high_score)),
        "macro_f1": float(f1_score(y_test, y_pred, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_test, y_pred, average="weighted", zero_division=0)),
    }

    report_dict = classification_report(
        y_test,
        y_pred,
        labels=[LABEL_TO_ID[label] for label in LABEL_ORDER],
        target_names=LABEL_ORDER,
        output_dict=True,
        zero_division=0,
    )
    report_df = pd.DataFrame(report_dict).transpose().reset_index().rename(columns={"index": "class"})

    cm = confusion_matrix(
        y_test,
        y_pred,
        labels=[LABEL_TO_ID[label] for label in LABEL_ORDER],
    )
    cm_df = pd.DataFrame(cm, index=[f"actual_{label}" for label in LABEL_ORDER], columns=[f"pred_{label}" for label in LABEL_ORDER])

    return metrics, report_df, cm_df



def save_model_artifacts(best_model: Pipeline, X: pd.DataFrame) -> None:
    with open(RISK_MODEL_PATH, "wb") as file:
        pickle.dump(best_model, file)

    preprocessor = best_model.named_steps["preprocessor"]
    with open(PREPROCESSOR_PATH, "wb") as file:
        pickle.dump(preprocessor, file)

    label_payload = {
        "label_to_id": LABEL_TO_ID,
        "id_to_label": {str(key): value for key, value in ID_TO_LABEL.items()},
        "feature_columns": X.columns.tolist(),
        "transformed_feature_names": get_feature_names(preprocessor),
        "excluded_columns": sorted(list(LEAKAGE_AND_METADATA_COLUMNS | OPTIONAL_EXCLUDED_COLUMNS)),
    }
    with open(LABEL_MAPPING_PATH, "w", encoding="utf-8") as file:
        json.dump(label_payload, file, indent=2, ensure_ascii=False)


# -----------------------------------------------------------------------------
# Isolation Forest anomaly layer
# -----------------------------------------------------------------------------


def add_anomaly_scores(
    train_preprocessor: ColumnTransformer,
    X_train: pd.DataFrame,
    X_all: pd.DataFrame,
    output: pd.DataFrame,
) -> pd.DataFrame:
    """Fit Isolation Forest on transformed training features and score all users."""
    X_train_transformed = train_preprocessor.transform(X_train)
    X_all_transformed = train_preprocessor.transform(X_all)

    isolation_forest = IsolationForest(
        n_estimators=250,
        contamination=0.04,
        random_state=RANDOM_SEED,
        n_jobs=-1,
    )
    isolation_forest.fit(X_train_transformed)

    # decision_function: higher means more normal. We invert it so higher = more anomalous.
    raw_anomaly_score = -isolation_forest.decision_function(X_all_transformed)
    output["anomaly_score"] = raw_anomaly_score
    output["anomaly_flag"] = (isolation_forest.predict(X_all_transformed) == -1).astype(int)

    with open(MODELS_DIR / "isolation_forest.pkl", "wb") as file:
        pickle.dump(isolation_forest, file)

    return output


# -----------------------------------------------------------------------------
# Training orchestration
# -----------------------------------------------------------------------------


def train_and_evaluate_models() -> None:
    ensure_output_dirs()

    print("Loading feature dataset...")
    data = load_dataset()
    X, y, user_ids = split_features_target(data)

    print(f"Dataset shape: {X.shape[0]:,} users x {X.shape[1]:,} features")
    print("Target distribution:")
    print(data["risk_label"].value_counts(normalize=True).mul(100).round(2).to_string())

    X_train, X_test, y_train, y_test, user_train, user_test = train_test_split(
        X,
        y,
        user_ids,
        test_size=0.25,
        random_state=RANDOM_SEED,
        stratify=y,
    )

    sample_weight = compute_sample_weight(class_weight="balanced", y=y_train)

    trained_models: dict[str, Pipeline] = {}
    comparison_rows: list[dict[str, Any]] = []
    all_reports: dict[str, pd.DataFrame] = {}
    all_confusion_matrices: dict[str, pd.DataFrame] = {}

    # Baseline model.
    print("\nTraining Logistic Regression baseline...")
    logistic_model = build_logistic_regression_pipeline(build_preprocessor(X_train))
    logistic_model.fit(X_train, y_train)
    trained_models["logistic_regression"] = logistic_model

    metrics, report_df, cm_df = evaluate_model("logistic_regression", logistic_model, X_test, y_test)
    comparison_rows.append(metrics)
    all_reports["logistic_regression"] = report_df
    all_confusion_matrices["logistic_regression"] = cm_df

    # Main model.
    if XGBOOST_AVAILABLE:
        print("Training XGBoost risk classifier...")
        xgb_model = build_xgboost_pipeline(build_preprocessor(X_train))
        xgb_model.fit(X_train, y_train, model__sample_weight=sample_weight)
        trained_models["xgboost"] = xgb_model

        metrics, report_df, cm_df = evaluate_model("xgboost", xgb_model, X_test, y_test)
        comparison_rows.append(metrics)
        all_reports["xgboost"] = report_df
        all_confusion_matrices["xgboost"] = cm_df
    else:
        print("\nWARNING: XGBoost is not installed. Using HistGradientBoosting fallback.")
        print("For the final portfolio version, install XGBoost with: pip install xgboost")
        fallback_model = build_hist_gradient_boosting_pipeline(build_preprocessor(X_train))
        fallback_model.fit(X_train, y_train)
        trained_models["hist_gradient_boosting_fallback"] = fallback_model

        metrics, report_df, cm_df = evaluate_model(
            "hist_gradient_boosting_fallback",
            fallback_model,
            X_test,
            y_test,
        )
        comparison_rows.append(metrics)
        all_reports["hist_gradient_boosting_fallback"] = report_df
        all_confusion_matrices["hist_gradient_boosting_fallback"] = cm_df

    comparison = pd.DataFrame(comparison_rows).sort_values(
        by=["recall_high", "f1_high", "pr_auc_high"],
        ascending=False,
    )

    best_model_name = str(comparison.iloc[0]["model"])
    best_model = trained_models[best_model_name]

    print("\nModel comparison:")
    print(comparison.round(4).to_string(index=False))
    print(f"\nSelected model: {best_model_name}")

    # Save reports.
    comparison.to_csv(MODEL_COMPARISON_PATH, index=False)
    for model_name, report_df in all_reports.items():
        report_df.to_csv(REPORTS_DIR / f"classification_report_{model_name}.csv", index=False)
    for model_name, cm_df in all_confusion_matrices.items():
        cm_df.to_csv(REPORTS_DIR / f"confusion_matrix_{model_name}.csv")

    metrics_payload = {
        "selected_model": best_model_name,
        "comparison": comparison.to_dict(orient="records"),
        "target_distribution": data["risk_label"].value_counts().to_dict(),
        "test_size": len(X_test),
        "random_seed": RANDOM_SEED,
        "xgboost_available": XGBOOST_AVAILABLE,
    }
    with open(METRICS_PATH, "w", encoding="utf-8") as file:
        json.dump(metrics_payload, file, indent=2, ensure_ascii=False)

    # Fit anomaly layer using the selected model preprocessor.
    print("\nScoring users and adding anomaly layer...")
    y_all_pred = best_model.predict(X)
    y_all_proba = predict_proba_safely(best_model, X)

    predictions = pd.DataFrame(
        {
            "user_id": user_ids,
            "true_risk_label": data["risk_label"],
            "predicted_risk_label": [ID_TO_LABEL[int(label_id)] for label_id in y_all_pred],
            "risk_score_low": y_all_proba[:, LABEL_TO_ID["low"]],
            "risk_score_medium": y_all_proba[:, LABEL_TO_ID["medium"]],
            "risk_score_high": y_all_proba[:, LABEL_TO_ID["high"]],
        }
    )

    predictions["risk_level"] = predictions["predicted_risk_label"]
    predictions["risk_score"] = predictions[["risk_score_low", "risk_score_medium", "risk_score_high"]].max(axis=1)

    predictions = add_anomaly_scores(
        train_preprocessor=best_model.named_steps["preprocessor"],
        X_train=X_train,
        X_all=X,
        output=predictions,
    )

    predictions.to_csv(PREDICTIONS_PATH, index=False)

    save_model_artifacts(best_model, X)

    print("\nTraining pipeline completed successfully.")
    print(f"Best model:          {RISK_MODEL_PATH}")
    print(f"Preprocessor:        {PREPROCESSOR_PATH}")
    print(f"Label mapping:       {LABEL_MAPPING_PATH}")
    print(f"Predictions:         {PREDICTIONS_PATH}")
    print(f"Model comparison:    {MODEL_COMPARISON_PATH}")
    print(f"Metrics:             {METRICS_PATH}")


if __name__ == "__main__":
    train_and_evaluate_models()
