#!/usr/bin/env python3
"""
NFL Playoff Data Fetcher — Wild Card through Super Bowl, last 5 seasons.
Uses ESPN API postseason schedule. Features from end-of-regular-season stats.
Output: data/playoff_data.csv

Usage: python python/fetch_playoff_data.py
"""
import sys, json, time, urllib.request, urllib.error
import numpy as np
import pandas as pd
from pathlib import Path
from collections import defaultdict

sys.stdout.reconfigure(encoding='utf-8', errors='replace')

PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR  = PROJECT_ROOT / "data"
CACHE_DIR = PROJECT_ROOT / "cache" / "python"
DATA_DIR.mkdir(exist_ok=True)
CACHE_DIR.mkdir(exist_ok=True)

REG_CSV  = DATA_DIR / "training_data.csv"
OUT_CSV  = DATA_DIR / "playoff_data.csv"

ESPN_BASE = "https://site.api.espn.com/apis/site/v2/sports/football/nfl"
# NFL playoff seasons: year = season year (2021 = 2021-22 playoffs in Jan 2022)
PLAYOFF_SEASONS = [2021, 2022, 2023, 2024, 2025]

K_FACTOR   = 20.0
HOME_ADV   = 65.0
LEAGUE_ELO = 1500.0

ESPN_ABBR_FIX = {"WSH": "WAS", "JAC": "JAX"}

def fix(a): return ESPN_ABBR_FIX.get(a, a)


def espn_get(url: str, retries: int = 3) -> dict:
    for i in range(retries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "NFL-Oracle/4.0"})
            with urllib.request.urlopen(req, timeout=20) as r:
                return json.loads(r.read())
        except Exception as e:
            if i == retries - 1:
                print(f"    Failed: {e}")
                return {}
            time.sleep(2 ** i)
    return {}


def fetch_postseason_games(season: int) -> list:
    """Fetch all postseason games for an NFL season via ESPN.
    NFL season 2021 = playoffs in Jan 2022. ESPN uses the ending year.
    """
    cache = CACHE_DIR / f"nfl_playoffs_{season}.json"
    if cache.exists():
        return json.loads(cache.read_text())

    games = []
    # NFL playoffs run Jan-Feb of (season+1). Scan date range.
    from datetime import datetime, timedelta
    start = datetime(season + 1, 1, 7)   # Wild Card weekend earliest
    end   = datetime(season + 1, 2, 15)  # Super Bowl latest
    current = start
    seen_ids = set()

    while current <= end:
        date_str = current.strftime("%Y%m%d")
        url = f"{ESPN_BASE}/scoreboard?dates={date_str}&seasontype=3&limit=20"
        data = espn_get(url)
        events = data.get("events", [])
        for ev in events:
            if ev.get("id") in seen_ids:
                continue
            seen_ids.add(ev.get("id",""))
            comps = ev.get("competitions", [{}])[0]
            competitors = comps.get("competitors", [])
            if len(competitors) < 2:
                continue
            home = next((c for c in competitors if c.get("homeAway") == "home"), None)
            away = next((c for c in competitors if c.get("homeAway") == "away"), None)
            if not home or not away:
                continue
            status = ev.get("status", {}).get("type", {}).get("completed", False)
            if not status:
                continue
            h_score = int(home.get("score", 0) or 0)
            a_score = int(away.get("score", 0) or 0)
            h_abbr  = fix(home.get("team", {}).get("abbreviation", ""))
            a_abbr  = fix(away.get("team", {}).get("abbreviation", ""))
            if not h_abbr or not a_abbr:
                continue
            games.append({
                "game_id":   ev.get("id", ""),
                "game_date": current.strftime("%Y-%m-%d"),
                "week":      0,
                "home_team": h_abbr,
                "away_team": a_abbr,
                "home_score": h_score,
                "away_score": a_score,
                "season":    season,
            })
        current += timedelta(days=1)

    # Remove duplicates
    seen = set(); unique = []
    for g in games:
        if g["game_id"] not in seen:
            seen.add(g["game_id"]); unique.append(g)

    cache.write_text(json.dumps(unique, indent=2))
    return unique


def build_elo_and_stats(reg_df: pd.DataFrame, season: int) -> tuple:
    elo = defaultdict(lambda: LEAGUE_ELO)
    season_stats = defaultdict(lambda: {"wins": 0, "games": 0, "pts_for": [], "pts_agst": []})

    s = reg_df[reg_df["season"] == season].copy()
    s = s.sort_values("game_date", na_position="last")

    for _, row in s.iterrows():
        h, a = row["home_team"], row["away_team"]
        he, ae = elo[h] + HOME_ADV, elo[a]
        exp_h = 1 / (1 + 10 ** ((ae - he) / 400))
        act_h = 1 if row.get("home_win", row.get("label", 0)) == 1 else 0
        elo[h] += K_FACTOR * (act_h - exp_h)
        elo[a] += K_FACTOR * ((1 - act_h) - (1 - exp_h))
        season_stats[h]["wins"]   += act_h
        season_stats[a]["wins"]   += (1 - act_h)
        season_stats[h]["games"]  += 1
        season_stats[a]["games"]  += 1
        season_stats[h]["pts_for"].append(row.get("ppg_diff", 0) + 23)
        season_stats[a]["pts_for"].append(23 - row.get("ppg_diff", 0))

    team_stats = {}
    for team, st in season_stats.items():
        g = st["games"]
        team_stats[team] = {
            "win_pct": st["wins"] / g if g > 0 else 0.5,
            "ppg": np.mean(st["pts_for"]) if st["pts_for"] else 23,
        }

    return dict(elo), team_stats


def main():
    print("NFL Playoff Data Fetcher")
    print("=" * 40)

    if not REG_CSV.exists():
        print(f"No regular season CSV at {REG_CSV}")
        sys.exit(1)

    reg_df = pd.read_csv(REG_CSV)
    reg_df["game_date"] = pd.to_datetime(reg_df["game_date"], errors="coerce")

    # Map column names
    if "home_win" not in reg_df.columns and "label" in reg_df.columns:
        reg_df["home_win"] = reg_df["label"]

    all_rows = []

    for season in PLAYOFF_SEASONS:
        print(f"\nSeason {season}")
        elo, stats = build_elo_and_stats(reg_df, season)
        games = fetch_postseason_games(season)
        print(f"  Fetched {len(games)} playoff games")

        for g in games:
            h, a = g["home_team"], g["away_team"]
            h_elo = elo.get(h, LEAGUE_ELO)
            a_elo = elo.get(a, LEAGUE_ELO)
            hs = stats.get(h, {"win_pct": 0.5, "ppg": 23})
            as_ = stats.get(a, {"win_pct": 0.5, "ppg": 23})
            label = 1 if g["home_score"] > g["away_score"] else 0

            row = {
                "season":        season,
                "game_id":       g["game_id"],
                "game_date":     g["game_date"],
                "home_team":     h,
                "away_team":     a,
                "home_score":    g["home_score"],
                "away_score":    g["away_score"],
                "label":         label,
                "home_win":      label,
                "is_playoff":    1,
                "playoff_week":  g["week"],
                "elo_diff":      h_elo - a_elo,
                "win_pct_diff":  hs["win_pct"] - as_["win_pct"],
                "ppg_diff":      hs["ppg"] - as_["ppg"],
                "pythagorean_diff": (hs["win_pct"] - as_["win_pct"]) * 0.3,
                "rest_days_diff": 0,
                "is_neutral":    1 if g["week"] == 5 else 0,  # Super Bowl
            }
            all_rows.append(row)

            # Update Elo
            exp = 1 / (1 + 10 ** ((a_elo - (h_elo + HOME_ADV)) / 400))
            elo[h] = h_elo + K_FACTOR * (label - exp)
            elo[a] = a_elo + K_FACTOR * ((1 - label) - (1 - exp))

    if not all_rows:
        print("\nNo playoff games fetched.")
        return

    df = pd.DataFrame(all_rows)
    df.to_csv(OUT_CSV, index=False)
    print(f"\nSaved {len(df)} playoff games to {OUT_CSV}")
    print(f"Seasons: {df['season'].unique().tolist()}")
    print(f"Home win rate: {df['label'].mean():.3f}")


if __name__ == "__main__":
    main()
