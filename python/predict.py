#!/usr/bin/env python3
"""
NFL Oracle v4.1 — Live Predictions
Fetches today's NFL games from ESPN, rebuilds Elo from history,
and runs the trained model. Auto-switches to the playoff model
from Wild Card weekend (Jan 7) through Super Bowl (Feb 15).

Usage:
  python python/predict.py              # today's games
  python python/predict.py --date 20270112
"""
import argparse, json, math, time, requests
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path
from collections import defaultdict

import sys
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

ROOT      = Path(__file__).parent.parent
DATA_DIR  = ROOT / "data"
MODEL_DIR = DATA_DIR / "model"
HIST_CSV  = DATA_DIR / "training_data.csv"

ESPN_BASE = "https://site.api.espn.com/apis/site/v2/sports/football/nfl"
HEADERS   = {"User-Agent": "NFL-Oracle/4.1"}

INITIAL_ELO = 1500.0
K_FACTOR    = 20.0
HOME_ADV    = 65.0

ESPN_ABBR_FIX = {"WSH": "WAS", "JAC": "JAX"}
def fix(a): return ESPN_ABBR_FIX.get(a, a)


# ── Season detection ───────────────────────────────────────────────────────────

def is_playoff_season(date_str: str) -> bool:
    """NFL playoffs: Jan 7 – Feb 15."""
    try:
        d = datetime.strptime(date_str, "%Y%m%d")
    except ValueError:
        return False
    return (d.month == 1 and d.day >= 7) or (d.month == 2 and d.day <= 15)


# ── ESPN helpers ───────────────────────────────────────────────────────────────

def fetch_json(url: str) -> dict:
    for attempt in range(3):
        try:
            r = requests.get(url, headers=HEADERS, timeout=12)
            r.raise_for_status()
            return r.json()
        except Exception:
            if attempt < 2:
                time.sleep(1.5 * (attempt + 1))
    return {}


def fetch_games(date_str: str) -> list:
    # NFL uses season type 2 (regular) or 3 (postseason)
    data = fetch_json(f"{ESPN_BASE}/scoreboard?dates={date_str}&limit=20")
    games = []
    for ev in data.get("events", []):
        status = ev.get("status", {}).get("type", {}).get("name", "")
        comp   = (ev.get("competitions") or [{}])[0]
        cs     = comp.get("competitors", [])
        home   = next((c for c in cs if c.get("homeAway") == "home"), None)
        away   = next((c for c in cs if c.get("homeAway") == "away"), None)
        if not home or not away:
            continue
        h_abbr = fix(home.get("team", {}).get("abbreviation", "").upper())
        a_abbr = fix(away.get("team", {}).get("abbreviation", "").upper())
        games.append({
            "event_name": ev.get("name", ""),
            "status":     status,
            "neutral":    int(comp.get("neutralSite", False)),
            "home_abbr":  h_abbr,
            "home_id":    home.get("team", {}).get("id", ""),
            "home_name":  home.get("team", {}).get("displayName", ""),
            "away_abbr":  a_abbr,
            "away_id":    away.get("team", {}).get("id", ""),
            "away_name":  away.get("team", {}).get("displayName", ""),
        })
    return games


def fetch_team_record(team_id: str) -> dict:
    data = fetch_json(f"{ESPN_BASE}/teams/{team_id}?enable=record,stats")
    items = data.get("team", {}).get("record", {}).get("items", [])
    result = {"win_pct": 0.5, "games_played": 0}
    for item in items:
        if item.get("type") == "total":
            stats = {s["name"]: s["value"] for s in item.get("stats", [])}
            gp    = stats.get("gamesPlayed", 0) or 0
            wins  = stats.get("wins", 0) or 0
            result["win_pct"] = wins / gp if gp > 0 else 0.5
            result["games_played"] = int(gp)
    return result


# ── Elo reconstruction ─────────────────────────────────────────────────────────

def build_elo_from_history() -> dict:
    if not HIST_CSV.exists():
        return {}
    df = pd.read_csv(HIST_CSV, usecols=["season", "game_date", "home_team",
                                         "away_team", "home_win"])
    df = df.sort_values("game_date")
    elo = defaultdict(lambda: INITIAL_ELO)
    last_season = None

    for _, row in df.iterrows():
        season = row["season"]
        if last_season and season != last_season:
            for t in list(elo.keys()):
                elo[t] = 0.75 * elo[t] + 0.25 * INITIAL_ELO
        last_season = season
        h, a  = str(row["home_team"]).upper(), str(row["away_team"]).upper()
        hw    = int(row["home_win"])
        rh, ra = elo[h], elo[a]
        exp_h  = 1.0 / (1.0 + 10 ** ((ra - (rh + HOME_ADV)) / 400.0))
        delta  = K_FACTOR * (hw - exp_h)
        elo[h] = rh + delta
        elo[a] = ra - delta

    return dict(elo)


# ── Model loading ──────────────────────────────────────────────────────────────

def load_model(date_str: str) -> dict | None:
    try:
        if is_playoff_season(date_str):
            po = MODEL_DIR / "playoff_coefficients.json"
            ps = MODEL_DIR / "playoff_scaler.json"
            pm = MODEL_DIR / "playoff_metadata.json"
            if po.exists() and ps.exists() and pm.exists():
                print("  [PLAYOFFS] Using NFL playoff model")
                return {
                    "coeff":  json.loads(po.read_text()),
                    "scaler": json.loads(ps.read_text()),
                    "calib":  {"x_thresholds": [], "y_thresholds": []},
                    "meta":   json.loads(pm.read_text()),
                }
        coeff  = json.loads((MODEL_DIR / "coefficients.json").read_text())
        scaler = json.loads((MODEL_DIR / "scaler.json").read_text())
        calib  = json.loads((MODEL_DIR / "calibration.json").read_text())
        meta   = json.loads((MODEL_DIR / "metadata.json").read_text())
        return {"coeff": coeff, "scaler": scaler, "calib": calib, "meta": meta}
    except Exception as e:
        print(f"  Model load failed: {e}")
        return None


def predict_proba(model: dict, fv: dict) -> float:
    coeff_data = model["coeff"]
    # Support both list-format {coefficients:[], intercept:float} and dict-format
    if "coefficients" in coeff_data:
        features  = coeff_data.get("feature_names", model["meta"].get("feature_names", []))
        coeff_arr = coeff_data["coefficients"]
        intercept = coeff_data["intercept"]
    else:
        features  = model["meta"].get("feature_names", [])
        coeff_arr = [coeff_data.get(f, 0.0) for f in features]
        intercept = coeff_data.get("_intercept", coeff_data.get("intercept", 0.0))

    mean  = model["scaler"]["mean"]
    scale = model["scaler"]["scale"]
    x     = [(fv.get(f, 0.0) - mean[i]) / (scale[i] if scale[i] != 0 else 1.0)
             for i, f in enumerate(features)]
    logit = sum(c * xi for c, xi in zip(coeff_arr, x)) + intercept
    raw   = 1.0 / (1.0 + math.exp(-logit))

    calib = model["calib"]
    bins  = calib.get("x_thresholds", calib.get("bins", []))
    cals  = calib.get("y_thresholds", calib.get("calibrated", []))
    if not bins or not cals:
        return raw
    if raw <= bins[0]:
        return cals[0]
    if raw >= bins[-1]:
        return cals[-1]
    for i in range(len(bins) - 1):
        if bins[i] <= raw <= bins[i + 1]:
            t = (raw - bins[i]) / (bins[i + 1] - bins[i])
            return cals[i] + t * (cals[i + 1] - cals[i])
    return raw


# ── Feature builder ────────────────────────────────────────────────────────────

def build_features(elo_ratings: dict, h_abbr: str, a_abbr: str,
                   h_rec: dict, a_rec: dict, is_neutral: int = 0) -> dict:
    rh   = elo_ratings.get(h_abbr, INITIAL_ELO)
    ra   = elo_ratings.get(a_abbr, INITIAL_ELO)
    h_wp = h_rec["win_pct"]
    a_wp = a_rec["win_pct"]

    return {
        "elo_diff":              rh - ra,
        "pythagorean_diff":      h_wp - a_wp,
        "win_pct_diff":          h_wp - a_wp,
        "ppg_diff":              0.0,
        "papg_diff":             0.0,
        "net_ppg_diff":          0.0,
        "yds_per_play_diff":     0.0,
        "def_yds_per_play_diff": 0.0,
        "turnover_diff_diff":    0.0,
        "passer_rating_diff":    0.0,
        "third_down_diff":       0.0,
        "red_zone_diff":         0.0,
        "injury_impact_diff":    0.0,
        "qb_out_home":           0.0,
        "qb_out_away":           0.0,
        "rest_days_diff":        0.0,
        "short_week_home":       0.0,
        "short_week_away":       0.0,
        "wind_adj":              0.0,
        "precip_adj":            0.0,
        "temp_adj":              0.0,
        "altitude_adj":          0.0,
        "divisional_game":       0.0,
        "is_neutral":            float(is_neutral),
        "vegas_home_prob":       0.0,
        "mc_win_pct":            h_wp,
    }


# ── Printing ───────────────────────────────────────────────────────────────────

def pad(s: str, w: int) -> str:
    return s[:w].ljust(w)


def print_predictions(results: list, date_str: str) -> None:
    width = 85
    print("\n" + "=" * width)
    print(f"  NFL ORACLE v4.1  |  {date_str}  |  {len(results)} games")
    print("=" * width)
    print("  " + pad("MATCHUP", 30) + pad("HOME WIN%", 11) + pad("AWAY WIN%", 11) + "PICK")
    print("-" * width)
    for r in sorted(results, key=lambda x: -max(x["home_prob"], x["away_prob"])):
        matchup  = f"{r['home_abbr']} vs {r['away_abbr']}"
        home_pct = f"{r['home_prob']*100:.1f}%"
        away_pct = f"{r['away_prob']*100:.1f}%"
        pick     = r["home_abbr"] if r["home_prob"] >= r["away_prob"] else r["away_abbr"]
        star     = " *" if max(r["home_prob"], r["away_prob"]) >= 0.70 else ""
        neutral  = " [N]" if r.get("neutral") else ""
        print(f"  {pad(matchup + neutral, 30)}{pad(home_pct, 11)}{pad(away_pct, 11)}{pick}{star}")
    print("-" * width)
    print("* = high confidence (>= 70%)  |  [N] = neutral site\n")


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--date", default=datetime.now().strftime("%Y%m%d"))
    args = parser.parse_args()
    date_str = args.date

    print(f"=== NFL Oracle v4.1 — Predictions for {date_str} ===\n")

    model = load_model(date_str)
    if not model:
        print("ERROR: No model found. Run: python python/train_model.py")
        return

    print("Loading Elo ratings from history...")
    elo = build_elo_from_history()
    print(f"  {len(elo)} teams rated")

    print(f"\nFetching games for {date_str}...")
    games = fetch_games(date_str)

    if not games:
        # Try ±7 days
        for offset in list(range(1, 15)) + list(range(-1, -8, -1)):
            d = (datetime.strptime(date_str, "%Y%m%d") + timedelta(days=offset)).strftime("%Y%m%d")
            games = fetch_games(d)
            if games:
                label = "next" if offset > 0 else "most recent"
                print(f"  No games today — showing {label} games ({d})")
                date_str = d
                break

    if not games:
        print("No upcoming NFL games found. Season may be between weeks.")
        return

    scheduled = [g for g in games if "SCHEDULED" in g["status"]] or games
    print(f"  Found {len(scheduled)} game(s)\n")

    results = []
    for game in scheduled:
        h_rec = fetch_team_record(game["home_id"])
        a_rec = fetch_team_record(game["away_id"])
        time.sleep(0.1)

        fv     = build_features(elo, game["home_abbr"], game["away_abbr"],
                                h_rec, a_rec, game.get("neutral", 0))
        home_p = predict_proba(model, fv)

        results.append({
            "home_abbr": game["home_abbr"],
            "away_abbr": game["away_abbr"],
            "home_prob": home_p,
            "away_prob": 1.0 - home_p,
            "neutral":   game.get("neutral", 0),
        })

    print_predictions(results, date_str)


if __name__ == "__main__":
    main()
