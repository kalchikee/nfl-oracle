#!/usr/bin/env python3
"""
NFL Oracle v4.0 — Walk-Forward Backtest
Loads training_data.csv and runs walk-forward cross-validation,
reporting per-season accuracy, Brier score, and feature importance.

Usage:
  python python/build_dataset.py   # if training_data.csv does not exist yet
  python python/backtest.py
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, brier_score_loss
from sklearn.preprocessing import StandardScaler

PROJECT_ROOT = Path(__file__).parent.parent
CSV_PATH  = PROJECT_ROOT / "data" / "training_data.csv"
MODEL_DIR = PROJECT_ROOT / "data" / "model"

FEATURE_NAMES = [
    "elo_diff", "pythagorean_diff", "win_pct_diff",
    "ppg_diff", "papg_diff", "net_ppg_diff",
    "yds_per_play_diff", "def_yds_per_play_diff",
    "turnover_diff_diff", "passer_rating_diff",
    "third_down_diff", "red_zone_diff",
    "injury_impact_diff", "qb_out_home", "qb_out_away",
    "rest_days_diff", "short_week_home", "short_week_away",
    "wind_adj", "precip_adj", "temp_adj",
    "altitude_adj", "divisional_game", "is_neutral",
    "vegas_home_prob", "mc_win_pct",
]
TARGET = "home_win"


# -- Load production model artifacts ------------------------------------------

def load_prod_model():
    coef_path = MODEL_DIR / "coefficients.json"
    scaler_path = MODEL_DIR / "scaler.json"
    meta_path = MODEL_DIR / "metadata.json"
    if not coef_path.exists():
        return None, None, None
    with open(coef_path) as f:
        coef = json.load(f)
    with open(scaler_path) as f:
        scaler_data = json.load(f)
    meta = {}
    if meta_path.exists():
        with open(meta_path) as f:
            meta = json.load(f)
    return coef, scaler_data, meta


def predict_with_prod_model(X, coef, scaler_data):
    mean = np.array(scaler_data["mean"])
    std  = np.array(scaler_data["std"])
    X_s  = (X - mean) / np.maximum(std, 1e-8)
    logits = X_s @ np.array(coef["coefficients"]) + coef["intercept"]
    return 1 / (1 + np.exp(-np.array(logits)))


# -- Simulated ROI ------------------------------------------------------------

def simulate_roi(preds, labels, min_conf=0.55, bet=10.0, bankroll=1000.0):
    w = l = 0
    bank = bankroll
    for p, y in zip(preds, labels):
        if min_conf <= p <= (1 - min_conf):
            continue
        pick = 1 if p >= 0.5 else 0
        edge_p = p if pick == 1 else (1 - p)
        fair_odds = 1.0 / max(edge_p, 0.01)
        offered = fair_odds * 0.95
        if pick == int(y):
            bank += bet * (offered - 1)
            w += 1
        else:
            bank -= bet
            l += 1
    roi = (bank - bankroll) / ((w + l) * bet) * 100 if (w + l) > 0 else 0.0
    return roi, w, l


# -- Walk-forward CV ----------------------------------------------------------

def walk_forward(df):
    seasons = sorted(df["season"].unique())
    if len(seasons) < 2:
        print("Need at least 2 seasons for walk-forward CV.")
        return {}, np.array([]), np.array([])

    all_preds, all_labels = [], []
    season_results = {}

    print("Walk-forward results:")
    print(f"  {'Season':>8}  {'N':>5}  {'Acc':>6}  {'Brier':>7}  {'HC%':>7}  {'HC N':>6}  {'ROI%':>7}")
    print(f"  {'-'*8}  {'-'*5}  {'-'*6}  {'-'*7}  {'-'*7}  {'-'*6}  {'-'*7}")

    for i, test_season in enumerate(seasons[1:], 1):
        train_df = df[df["season"].isin(seasons[:i])]
        test_df  = df[df["season"] == test_season]
        if len(train_df) < 30 or len(test_df) < 10:
            continue

        feats = [f for f in FEATURE_NAMES if f in train_df.columns]
        X_tr = train_df[feats].fillna(0).values
        y_tr = train_df[TARGET].values
        X_te = test_df[feats].fillna(0).values
        y_te = test_df[TARGET].values

        scaler = StandardScaler()
        X_tr_s = scaler.fit_transform(X_tr)
        X_te_s = scaler.transform(X_te)

        model = LogisticRegression(C=1.0, max_iter=1000, solver="lbfgs")
        model.fit(X_tr_s, y_tr)

        preds = np.clip(model.predict_proba(X_te_s)[:, 1], 0.01, 0.99)

        acc   = accuracy_score(y_te, preds >= 0.5)
        brier = brier_score_loss(y_te, preds)
        hc    = (preds >= 0.65) | (preds <= 0.35)
        hc_acc = accuracy_score(y_te[hc], preds[hc] >= 0.5) if hc.sum() > 0 else None
        roi, wins, losses = simulate_roi(preds, y_te)

        season_results[int(test_season)] = {
            "n": len(test_df),
            "accuracy": round(float(acc), 4),
            "brier": round(float(brier), 4),
            "hc_accuracy": round(float(hc_acc), 4) if hc_acc is not None else None,
            "hc_n": int(hc.sum()),
            "roi_pct": round(roi, 2),
        }

        hc_str = f"{hc_acc:.3f}" if hc_acc is not None else "   N/A"
        print(f"  {test_season:>8}  {len(test_df):>5}  {acc:>6.3f}  {brier:>7.4f}  "
              f"{hc_str:>7}  {hc.sum():>6}  {roi:>+7.1f}%")

        all_preds.extend(preds.tolist())
        all_labels.extend(y_te.tolist())

    return season_results, np.array(all_preds), np.array(all_labels)


# -- Feature importance -------------------------------------------------------

def print_feature_importance(df):
    feats = [f for f in FEATURE_NAMES if f in df.columns]
    X = df[feats].fillna(0).values
    y = df[TARGET].values
    scaler = StandardScaler()
    X_s = scaler.fit_transform(X)
    model = LogisticRegression(C=1.0, max_iter=1000, solver="lbfgs")
    model.fit(X_s, y)
    ranked = sorted(zip(feats, model.coef_[0]), key=lambda x: abs(x[1]), reverse=True)
    print("\nFeature Importance (LR coefficient magnitude, full dataset):")
    for name, val in ranked[:15]:
        bar = "#" * int(abs(val) * 20)
        sign = "+" if val > 0 else "-"
        print(f"  {name:<32} {sign}{abs(val):>5.3f}  {bar}")


# -- Main ---------------------------------------------------------------------

def main():
    if not CSV_PATH.exists():
        print(f"ERROR: {CSV_PATH} not found.")
        print("  Run: python python/build_dataset.py")
        sys.exit(1)

    df = pd.read_csv(CSV_PATH)
    print("NFL Oracle v4.0 -- Walk-Forward Backtest")
    print("=" * 50)
    print(f"Loaded {len(df)} games, seasons: {sorted(df['season'].unique().tolist())}")
    print(f"Home win rate: {df[TARGET].mean():.3f}  (expect ~0.56)\n")

    season_results, all_preds, all_labels = walk_forward(df)

    if len(all_preds) == 0:
        print("No predictions generated.")
        return

    print()
    print("Aggregate (all out-of-sample seasons):")
    print(f"  Total games:        {len(all_preds)}")
    print(f"  Accuracy:           {accuracy_score(all_labels, all_preds >= 0.5):.4f}")
    print(f"  Brier score:        {brier_score_loss(all_labels, all_preds):.4f}  (naive baseline ~0.247)")
    hc = (all_preds >= 0.65) | (all_preds <= 0.35)
    if hc.sum() > 0:
        print(f"  HC accuracy (>=65%): {accuracy_score(all_labels[hc], all_preds[hc] >= 0.5):.4f}  n={hc.sum()}")
    roi, w, l = simulate_roi(all_preds, all_labels)
    print(f"  Simulated ROI:      {roi:+.2f}%  (W={w}, L={l})")

    # Production model check
    coef, scaler_data, meta = load_prod_model()
    if coef is not None:
        feats = [f for f in FEATURE_NAMES if f in df.columns]
        X_all = df[feats].fillna(0).values
        y_all = df[TARGET].values
        prod_preds = predict_with_prod_model(X_all, coef, scaler_data)
        print(f"\nProduction model (in-sample check):")
        print(f"  Trained:   {meta.get('trained_at', 'unknown')}")
        print(f"  Accuracy:  {accuracy_score(y_all, prod_preds >= 0.5):.4f}")
        print(f"  Brier:     {brier_score_loss(y_all, prod_preds):.4f}")

    print_feature_importance(df)
    print("\nBacktest complete.")


if __name__ == "__main__":
    main()
