#!/usr/bin/env python3
"""
NFL Oracle v4.0 — Training Dataset Builder (ESPN API version)
Fetches 5 seasons of historical NFL data directly from ESPN's free API.
No nfl_data_py / C compiler required.

Run:
    python python/build_dataset.py
"""

import json
import sys
import time
import urllib.request
import urllib.error
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).parent.parent
OUTPUT_PATH = PROJECT_ROOT / "data" / "training_data.csv"
OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

TRAIN_SEASONS = [2021, 2022, 2023, 2024, 2025]
ESPN_BASE = "https://site.api.espn.com/apis/site/v2/sports/football/nfl"

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

DIVISIONS = {
    "BUF":"AFC_E","MIA":"AFC_E","NE":"AFC_E","NYJ":"AFC_E",
    "BAL":"AFC_N","CIN":"AFC_N","CLE":"AFC_N","PIT":"AFC_N",
    "HOU":"AFC_S","IND":"AFC_S","JAX":"AFC_S","TEN":"AFC_S",
    "DEN":"AFC_W","KC":"AFC_W","LV":"AFC_W","LAC":"AFC_W",
    "DAL":"NFC_E","NYG":"NFC_E","PHI":"NFC_E","WAS":"NFC_E",
    "CHI":"NFC_N","DET":"NFC_N","GB":"NFC_N","MIN":"NFC_N",
    "ATL":"NFC_S","CAR":"NFC_S","NO":"NFC_S","TB":"NFC_S",
    "ARI":"NFC_W","LAR":"NFC_W","SF":"NFC_W","SEA":"NFC_W",
}
DOME_TEAMS = {"ARI","ATL","DAL","DET","HOU","IND","LV","MIN","NO","LAR","LAC"}
ALTITUDE_TEAMS = {"DEN"}

# ESPN team abbreviation normalization
ESPN_ABBR_FIX = {"WSH": "WAS", "JAC": "JAX"}

def fix_abbr(a: str) -> str:
    return ESPN_ABBR_FIX.get(a, a)

# ─── ESPN fetch helper ────────────────────────────────────────────────────────

def espn_get(url: str, retries=3) -> dict:
    for attempt in range(retries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "NFLOracle/4.0"})
            with urllib.request.urlopen(req, timeout=20) as resp:
                return json.loads(resp.read())
        except Exception as e:
            if attempt < retries - 1:
                time.sleep(2 ** attempt)
            else:
                return {}
    return {}

# ─── Elo engine ───────────────────────────────────────────────────────────────

class EloEngine:
    K = 20; HOME_ADV = 65; MEAN = 1500; REGRESSION = 0.67

    def __init__(self):
        all_teams = list(DIVISIONS.keys())
        self.ratings = {t: 1500.0 for t in all_teams}

    def regress(self):
        for t in self.ratings:
            self.ratings[t] = self.REGRESSION * self.ratings[t] + (1 - self.REGRESSION) * self.MEAN

    def expected(self, home, away):
        h = self.ratings.get(home, 1500) + self.HOME_ADV
        a = self.ratings.get(away, 1500)
        return 1 / (1 + 10 ** ((a - h) / 400))

    def update(self, home, away, hs, as_):
        exp = self.expected(home, away)
        act = 1.0 if hs > as_ else 0.0 if hs < as_ else 0.5
        mov = np.log(1 + min(abs(hs - as_), 21)) / np.log(22)
        delta = self.K * mov * (act - exp)
        self.ratings[home] = self.ratings.get(home, 1500) + delta
        self.ratings[away] = self.ratings.get(away, 1500) - delta

    def diff(self, home, away):
        return self.ratings.get(home, 1500) - self.ratings.get(away, 1500)

def pythagorean(ppg, papg, exp=2.37):
    if papg <= 0: return 0.5
    return ppg**exp / (ppg**exp + papg**exp)

# ─── Fetch full season schedule from ESPN ─────────────────────────────────────

def fetch_season_games(season: int) -> list[dict]:
    """Fetch all regular-season + postseason games for a given season."""
    games = []
    # Regular season: weeks 1–18; Postseason: weeks 1–4 (seasontype=3)
    for season_type, max_weeks in [(2, 18), (3, 4)]:
        for week in range(1, max_weeks + 1):
            url = f"{ESPN_BASE}/scoreboard?seasontype={season_type}&week={week}&dates={season}&limit=50"
            data = espn_get(url)
            events = data.get("events", [])
            for ev in events:
                comp = (ev.get("competitions") or [{}])[0]
                competitors = comp.get("competitors", [])
                if len(competitors) < 2:
                    continue
                home = next((c for c in competitors if c.get("homeAway") == "home"), None)
                away = next((c for c in competitors if c.get("homeAway") == "away"), None)
                if not home or not away:
                    continue
                status = (ev.get("status") or {}).get("type", {}).get("name", "")
                if "FINAL" not in status.upper():
                    continue
                home_score = int(home.get("score", 0) or 0)
                away_score = int(away.get("score", 0) or 0)
                home_abbr = fix_abbr(home.get("team", {}).get("abbreviation", "UNK"))
                away_abbr = fix_abbr(away.get("team", {}).get("abbreviation", "UNK"))
                game_date = ev.get("date", "")[:10]

                # Extract embedded odds
                odds_list = comp.get("odds", [])
                vegas_spread = 0.0
                if odds_list:
                    details = odds_list[0].get("details", "")
                    # e.g. "BAL -3.5" → home favored by 3.5
                    if details:
                        parts = details.split()
                        if len(parts) == 2:
                            try:
                                fav = parts[0]
                                val = float(parts[1])
                                vegas_spread = -val if fav == home_abbr else val
                            except ValueError:
                                pass

                games.append({
                    "game_id": ev.get("id", ""),
                    "season": season,
                    "season_type": season_type,
                    "week": week,
                    "game_date": game_date,
                    "home_team": home_abbr,
                    "away_team": away_abbr,
                    "home_score": home_score,
                    "away_score": away_score,
                    "neutral_site": comp.get("neutralSite", False),
                    "vegas_spread": vegas_spread,
                })
            time.sleep(0.15)  # be polite to ESPN API
    return games

# ─── Rolling stats from prior games ──────────────────────────────────────────

def rolling_stats(past_games: list[dict], team: str) -> dict:
    """Compute team stats from a list of that team's prior games."""
    if not past_games:
        return {"ppg":22.5,"papg":22.5,"win_pct":0.5,"pythagorean":0.5,
                "ypp":5.3,"def_ypp":5.3,"turnover_diff":0.0,
                "passer_rating":90.0,"third_down":0.40,"red_zone":0.60}
    pf = pa = wins = 0
    for g in past_games:
        if g["home_team"] == team:
            pf += g["home_score"]; pa += g["away_score"]
            wins += 1 if g["home_score"] > g["away_score"] else 0
        else:
            pf += g["away_score"]; pa += g["home_score"]
            wins += 1 if g["away_score"] > g["home_score"] else 0
    n = len(past_games)
    ppg = pf / n; papg = pa / n
    ypp = max(3.5, min(7.0, 5.3 + (ppg - 22.5) * 0.04))
    def_ypp = max(3.5, min(7.0, 5.3 + (papg - 22.5) * 0.04))
    return {
        "ppg": ppg, "papg": papg, "win_pct": wins/n,
        "pythagorean": pythagorean(ppg, papg),
        "ypp": ypp, "def_ypp": def_ypp,
        "turnover_diff": 0.0, "passer_rating": 90.0,
        "third_down": 0.40, "red_zone": 0.60,
    }

# ─── Rest days ────────────────────────────────────────────────────────────────

def rest_days(past_games: list[dict], game_date: str) -> int:
    if not past_games:
        return 14
    last = max(g["game_date"] for g in past_games)
    try:
        diff = (pd.to_datetime(game_date) - pd.to_datetime(last)).days
        return max(1, min(21, int(diff)))
    except Exception:
        return 7

# ─── Build features for one game ─────────────────────────────────────────────

def build_features(game: dict, all_games: list[dict], elo: EloEngine) -> dict:
    home = game["home_team"]
    away = game["away_team"]
    gdate = game["game_date"]
    season = game["season"]
    week = game["week"]
    stype = game["season_type"]

    # Prior games for each team (same season, before this game)
    def prior(team):
        return [g for g in all_games
                if g["season"] == season
                and (g["season_type"] < stype or (g["season_type"] == stype and g["week"] < week))
                and (g["home_team"] == team or g["away_team"] == team)]

    home_prior = prior(home)
    away_prior = prior(away)

    # If week 1, use previous season
    if not home_prior:
        home_prior = [g for g in all_games if g["season"] == season-1
                      and (g["home_team"] == home or g["away_team"] == home)]
    if not away_prior:
        away_prior = [g for g in all_games if g["season"] == season-1
                      and (g["home_team"] == away or g["away_team"] == away)]

    hs = rolling_stats(home_prior, home)
    as_ = rolling_stats(away_prior, away)

    elo_diff = elo.diff(home, away)
    pyth_diff = hs["pythagorean"] - as_["pythagorean"]
    win_diff = hs["win_pct"] - as_["win_pct"]
    ppg_diff = hs["ppg"] - as_["ppg"]
    papg_diff = hs["papg"] - as_["papg"]
    net_ppg = (hs["ppg"] - hs["papg"]) - (as_["ppg"] - as_["papg"])
    ypp_diff = hs["ypp"] - as_["ypp"]
    def_ypp_diff = hs["def_ypp"] - as_["def_ypp"]

    hr = rest_days(home_prior, gdate)
    ar = rest_days(away_prior, gdate)
    rest_diff = hr - ar

    div_game = 1 if DIVISIONS.get(home) and DIVISIONS.get(home) == DIVISIONS.get(away) else 0
    altitude = 0.3 if home in ALTITUDE_TEAMS else 0.0

    vs = game["vegas_spread"]  # positive = home favored
    vegas_prob = max(0.10, min(0.90, 0.5 + vs / 28.0)) if vs != 0 else 0.5
    mc_win_pct = max(0.10, min(0.90, 0.5 + pyth_diff * 0.5 + 0.03))

    return {
        "season": season, "week": week, "season_type": stype,
        "game_date": gdate, "home_team": home, "away_team": away,
        "home_score": game["home_score"], "away_score": game["away_score"],
        "home_win": 1 if game["home_score"] > game["away_score"] else 0,
        "elo_diff": round(elo_diff, 2),
        "pythagorean_diff": round(pyth_diff, 4),
        "win_pct_diff": round(win_diff, 4),
        "ppg_diff": round(ppg_diff, 2),
        "papg_diff": round(papg_diff, 2),
        "net_ppg_diff": round(net_ppg, 2),
        "yds_per_play_diff": round(ypp_diff, 3),
        "def_yds_per_play_diff": round(def_ypp_diff, 3),
        "turnover_diff_diff": 0.0,
        "passer_rating_diff": 0.0,
        "third_down_diff": 0.0,
        "red_zone_diff": 0.0,
        "injury_impact_diff": 0.0,
        "qb_out_home": 0, "qb_out_away": 0,
        "rest_days_diff": int(rest_diff),
        "short_week_home": 1 if hr <= 5 else 0,
        "short_week_away": 1 if ar <= 5 else 0,
        "wind_adj": 0.0, "precip_adj": 0.0, "temp_adj": 0.0,
        "altitude_adj": altitude,
        "divisional_game": div_game,
        "is_neutral": 1 if game["neutral_site"] else 0,
        "vegas_home_prob": round(vegas_prob, 4),
        "mc_win_pct": round(mc_win_pct, 4),
    }

# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    elo = EloEngine()
    all_games = []
    rows = []

    for season in TRAIN_SEASONS:
        print(f"Fetching {season} season from ESPN...", flush=True)
        games = fetch_season_games(season)
        print(f"  -> {len(games)} completed games", flush=True)

        # Sort chronologically
        games.sort(key=lambda g: (g["season_type"], g["week"], g["game_date"]))

        elo.regress()  # offseason regression before each new season

        for game in games:
            try:
                feat = build_features(game, all_games + games, elo)
                rows.append(feat)
            except Exception as e:
                pass
            # Update Elo AFTER building features (no lookahead)
            elo.update(game["home_team"], game["away_team"], game["home_score"], game["away_score"])

        all_games.extend(games)

    df = pd.DataFrame(rows)
    df.to_csv(OUTPUT_PATH, index=False)
    print(f"\nDataset: {len(df)} games -> {OUTPUT_PATH}")
    by_season = df.groupby("season")["home_win"].agg(["count","mean"])
    print("\nGames per season:")
    print(by_season.rename(columns={"count":"games","mean":"home_win_rate"}).to_string())
    print(f"\nOverall home win rate: {df['home_win'].mean():.3f}")

if __name__ == "__main__":
    main()
