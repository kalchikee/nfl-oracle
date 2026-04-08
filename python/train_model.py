#!/usr/bin/env python3
"""
NFL Oracle v4.0 — ML Meta-Model Training
Trains a Logistic Regression on features from build_dataset.py.
Uses walk-forward cross-validation (train 2021-N, test N+1).
Exports model artifacts to data/model/ for the TypeScript runtime.

Install dependencies:
    pip install pandas numpy scikit-learn

Run:
    python python/build_dataset.py   (first time — builds training_data.csv)
    python python/train_model.py     (trains model → data/model/*.json)
"""

import hashlib
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

try:
    from sklearn.calibration import CalibratedClassifierCV
    from sklearn.isotonic import IsotonicRegression
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score, brier_score_loss, log_loss
    from sklearn.preprocessing import StandardScaler
except ImportError:
    print("ERROR: scikit-learn not installed. Run: pip install scikit-learn")
    sys.exit(1)

PROJECT_ROOT = Path(__file__).parent.parent
MODEL_DIR = PROJECT_ROOT / "data" / "model"
CSV_PATH = PROJECT_ROOT / "data" / "training_data.csv"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

# Feature names (must match TypeScript FEATURE_NAMES in metaModel.ts)
FEATURE_NAMES = [
    "elo_diff",
    "pythagorean_diff",
    "win_pct_diff",
    "ppg_diff",
    "papg_diff",
    "net_ppg_diff",
    "yds_per_play_diff",
    "def_yds_per_play_diff",
    "turnover_diff_diff",
    "passer_rating_diff",
    "third_down_diff",
    "red_zone_diff",
    "injury_impact_diff",
    "qb_out_home",
    "qb_out_away",
    "rest_days_diff",
    "short_week_home",
    "short_week_away",
    "wind_adj",
    "precip_adj",
    "temp_adj",
    "altitude_adj",
    "divisional_game",
    "is_neutral",
    "vegas_home_prob",
    "mc_win_pct",
]

TARGET = "home_win"


# ─── Load data ─────────────────────────────────────────────────────────────────

def load_data() -> pd.DataFrame:
    if not CSV_PATH.exists():
        print(f"ERROR: {CSV_PATH} not found. Run build_dataset.py first.")
        sys.exit(1)
    df = pd.read_csv(CSV_PATH)
    print(f"Loaded {len(df)} rows from {CSV_PATH}")

    # Validate features
    missing = [f for f in FEATURE_NAMES if f not in df.columns]
    if missing:
        print(f"WARNING: Missing features: {missing}")
        for f in missing:
            df[f] = 0.0

    # Fill NaN
    df[FEATURE_NAMES] = df[FEATURE_NAMES].fillna(0.0)
    df[TARGET] = df[TARGET].astype(int)

    print(f"Seasons: {sorted(df['season'].unique().tolist())}")
    print(f"Home win rate: {df[TARGET].mean():.3f}  (should be ~0.56)")
    return df


# ─── Walk-forward cross-validation ────────────────────────────────────────────

def walk_forward_cv(df: pd.DataFrame) -> dict:
    seasons = sorted(df["season"].unique())
    if len(seasons) < 2:
        print("WARNING: Need at least 2 seasons for walk-forward CV")
        return {"splits": [], "avg_brier": 0.25, "avg_accuracy": 0.5}

    results = []
    print("\n=== Walk-Forward Cross-Validation ===")

    for i in range(1, len(seasons)):
        train_seasons = seasons[:i]
        test_season = seasons[i]

        train_df = df[df["season"].isin(train_seasons)]
        test_df = df[df["season"] == test_season]

        if len(test_df) == 0:
            continue

        X_train = train_df[FEATURE_NAMES].values
        y_train = train_df[TARGET].values
        X_test = test_df[FEATURE_NAMES].values
        y_test = test_df[TARGET].values

        # Scale
        scaler = StandardScaler()
        X_train_s = scaler.fit_transform(X_train)
        X_test_s = scaler.transform(X_test)

        # Train LR with strong L2 regularization (small NFL dataset → overfit risk)
        lr = LogisticRegression(C=0.5, max_iter=1000, solver="lbfgs", random_state=42)
        lr.fit(X_train_s, y_train)

        # Calibrate on test set (in practice we'd use a hold-out, but NFL sample is small)
        probs = lr.predict_proba(X_test_s)[:, 1]
        preds = (probs >= 0.5).astype(int)

        brier = brier_score_loss(y_test, probs)
        acc = accuracy_score(y_test, preds)

        print(f"  Train: {train_seasons} -> Test: {test_season}  |  Brier: {brier:.4f}  Acc: {acc:.3f}  N={len(test_df)}")
        results.append({"train_seasons": train_seasons, "test_season": test_season, "brier": brier, "accuracy": acc})

    avg_brier = np.mean([r["brier"] for r in results]) if results else 0.25
    avg_acc = np.mean([r["accuracy"] for r in results]) if results else 0.5
    print(f"\nAvg Brier: {avg_brier:.4f}  Avg Accuracy: {avg_acc:.3f}")
    return {"splits": results, "avg_brier": avg_brier, "avg_accuracy": avg_acc}


# ─── Train final model on all data ────────────────────────────────────────────

def train_final_model(df: pd.DataFrame):
    X = df[FEATURE_NAMES].values
    y = df[TARGET].values

    print(f"\nTraining final model on {len(df)} games ({df['season'].min()}–{df['season'].max()})...")

    # Fit scaler
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Train logistic regression
    lr = LogisticRegression(C=0.5, max_iter=1000, solver="lbfgs", random_state=42)
    lr.fit(X_scaled, y)

    # Isotonic calibration on the full dataset (reasonable for small datasets)
    probs_train = lr.predict_proba(X_scaled)[:, 1]

    # Use isotonic regression for calibration mapping
    iso = IsotonicRegression(out_of_bounds="clip")
    iso.fit(probs_train, y)

    calibrated_probs = iso.predict(probs_train)
    brier_before = brier_score_loss(y, probs_train)
    brier_after = brier_score_loss(y, calibrated_probs)
    acc = accuracy_score(y, (probs_train >= 0.5).astype(int))

    print(f"  In-sample Brier (before cal): {brier_before:.4f}")
    print(f"  In-sample Brier (after cal):  {brier_after:.4f}")
    print(f"  In-sample Accuracy: {acc:.3f}")
    print(f"  Feature importances (top 10):")

    coef_df = pd.DataFrame({"feature": FEATURE_NAMES, "coef": lr.coef_[0]})
    coef_df["abs"] = coef_df["coef"].abs()
    top = coef_df.nlargest(10, "abs")
    for _, row in top.iterrows():
        print(f"    {row['feature']:30s}  {row['coef']:+.4f}")

    return lr, scaler, iso, probs_train, y


# ─── Export model artifacts ───────────────────────────────────────────────────

def export_artifacts(lr, scaler, iso, cv_results: dict, df: pd.DataFrame):
    probs_sorted = np.sort(lr.predict_proba(scaler.transform(df[FEATURE_NAMES].values))[:, 1])

    # coefficients.json
    coef_data = {
        "intercept": float(lr.intercept_[0]),
        "coefficients": [float(c) for c in lr.coef_[0]],
        "feature_names": FEATURE_NAMES,
    }
    (MODEL_DIR / "coefficients.json").write_text(json.dumps(coef_data, indent=2))

    # scaler.json
    scaler_data = {
        "mean": [float(m) for m in scaler.mean_],
        "scale": [float(s) for s in scaler.scale_],
        "feature_names": FEATURE_NAMES,
    }
    (MODEL_DIR / "scaler.json").write_text(json.dumps(scaler_data, indent=2))

    # calibration.json (isotonic thresholds)
    x_thresh = [float(x) for x in iso.X_thresholds_] if hasattr(iso, "X_thresholds_") else list(np.linspace(0.1, 0.9, 20))
    y_thresh = [float(y) for y in iso.y_thresholds_] if hasattr(iso, "y_thresholds_") else x_thresh
    cal_data = {"x_thresholds": x_thresh, "y_thresholds": y_thresh}
    (MODEL_DIR / "calibration.json").write_text(json.dumps(cal_data, indent=2))

    # metadata.json
    coef_hash = hashlib.md5(json.dumps(coef_data).encode()).hexdigest()[:12]
    meta_data = {
        "version": "4.0.0",
        "train_seasons": f"{df['season'].min()}-{df['season'].max()}",
        "n_games": int(len(df)),
        "n_features": len(FEATURE_NAMES),
        "avg_brier": round(cv_results["avg_brier"], 4),
        "avg_accuracy": round(cv_results["avg_accuracy"], 4),
        "coef_hash": coef_hash,
        "trained_at": datetime.now().isoformat(),
    }
    (MODEL_DIR / "metadata.json").write_text(json.dumps(meta_data, indent=2))

    print(f"\nModel artifacts exported to {MODEL_DIR}/")
    print(f"  coefficients.json  scaler.json  calibration.json  metadata.json")
    print(f"\nModel summary:")
    print(f"  Trained on: {meta_data['train_seasons']} ({meta_data['n_games']} games)")
    print(f"  Walk-forward CV Brier: {meta_data['avg_brier']:.4f}  (0.25 = coin flip)")
    print(f"  Walk-forward CV Acc:   {meta_data['avg_accuracy']:.3f}")
    print(f"\nRun the pipeline to use the new model:")
    print(f"  npm start")


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    df = load_data()
    cv_results = walk_forward_cv(df)
    lr, scaler, iso, _, _ = train_final_model(df)
    export_artifacts(lr, scaler, iso, cv_results, df)


if __name__ == "__main__":
    main()
