#!/usr/bin/env python3
"""
NFL Oracle — Enrich Training Data with ESPN Team Stats

Backfills four previously-zero features in training_data.csv:
  - turnover_diff_diff: team turnover differential difference (home - away)
  - passer_rating_diff: team passer rating difference
  - third_down_diff: third down conversion % difference
  - red_zone_diff: red zone TD % difference

Data source: ESPN public team statistics API.
Stats are per-season averages applied to each game in that season.

Usage:
    python python/enrich_features.py
"""

import json
import shutil
import sys
import time
import urllib.request
import urllib.error
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).parent.parent
CSV_PATH = PROJECT_ROOT / "data" / "training_data.csv"
BACKUP_PATH = PROJECT_ROOT / "data" / "training_data_backup.csv"

# ESPN team ID -> abbreviation mapping (our internal abbreviation)
ESPN_TEAM_MAP = {
    22: "ARI", 1: "ATL", 33: "BAL", 2: "BUF", 29: "CAR", 3: "CHI",
    4: "CIN", 5: "CLE", 6: "DAL", 7: "DEN", 8: "DET", 9: "GB",
    34: "HOU", 11: "IND", 30: "JAX", 12: "KC", 13: "LV", 24: "LAC",
    14: "LAR", 15: "MIA", 16: "MIN", 17: "NE", 18: "NO", 19: "NYG",
    20: "NYJ", 21: "PHI", 23: "PIT", 25: "SF", 26: "SEA", 27: "TB",
    10: "TEN", 28: "WAS",
}

ESPN_BASE = "https://site.api.espn.com/apis/site/v2/sports/football/nfl"


def espn_get(url: str, retries: int = 3) -> dict:
    """Fetch JSON from ESPN API with retries."""
    for attempt in range(retries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "NFLOracle/4.0"})
            with urllib.request.urlopen(req, timeout=20) as resp:
                return json.loads(resp.read())
        except Exception as e:
            if attempt < retries - 1:
                time.sleep(2 ** attempt)
            else:
                print(f"  WARN: Failed to fetch {url}: {e}")
                return {}
    return {}


def extract_stat(categories: list, category_name: str, stat_name: str) -> float | None:
    """Extract a named stat from ESPN categories list."""
    for cat in categories:
        if cat.get("name") == category_name:
            for s in cat.get("stats", []):
                if s.get("name") == stat_name:
                    return s.get("value")
    return None


def fetch_team_stats(season: int) -> dict[str, dict]:
    """
    Fetch season-level team stats from ESPN for all 32 NFL teams.
    Returns: { team_abbr: { turnover_diff, passer_rating, third_down_pct, red_zone_td_pct } }
    """
    team_stats = {}
    for team_id, abbr in ESPN_TEAM_MAP.items():
        url = f"{ESPN_BASE}/teams/{team_id}/statistics?season={season}"
        data = espn_get(url)

        # Navigate to categories
        categories = []
        results = data.get("results", {})
        if isinstance(results, dict):
            stats_block = results.get("stats", {})
            if isinstance(stats_block, dict):
                categories = stats_block.get("categories", [])

        if not categories:
            print(f"  WARN: No stats for {abbr} ({team_id}) in {season}")
            team_stats[abbr] = {
                "turnover_diff": 0.0,
                "passer_rating": 90.0,
                "third_down_pct": 40.0,
                "red_zone_td_pct": 55.0,
            }
            continue

        turnover_diff = extract_stat(categories, "miscellaneous", "turnOverDifferential") or 0.0
        passer_rating = extract_stat(categories, "passing", "QBRating") or 90.0
        third_down_pct = extract_stat(categories, "miscellaneous", "thirdDownConvPct") or 40.0
        red_zone_td_pct = extract_stat(categories, "miscellaneous", "redzoneTouchdownPct") or 55.0

        team_stats[abbr] = {
            "turnover_diff": turnover_diff,
            "passer_rating": passer_rating,
            "third_down_pct": third_down_pct,
            "red_zone_td_pct": red_zone_td_pct,
        }
        time.sleep(0.15)  # be polite to ESPN API

    return team_stats


def main():
    if not CSV_PATH.exists():
        print(f"ERROR: {CSV_PATH} not found. Run build_dataset.py first.")
        sys.exit(1)

    # Load and backup
    df = pd.read_csv(CSV_PATH)
    shutil.copy2(CSV_PATH, BACKUP_PATH)
    print(f"Loaded {len(df)} rows from {CSV_PATH}")
    print(f"Backup saved to {BACKUP_PATH}")

    # Show before state
    for col in ["turnover_diff_diff", "passer_rating_diff", "third_down_diff", "red_zone_diff"]:
        print(f"  BEFORE {col}: mean={df[col].mean():.4f}, std={df[col].std():.4f}, non-zero={( df[col] != 0).sum()}")

    # Fetch stats for each season
    seasons = sorted(df["season"].unique())
    all_team_stats: dict[int, dict[str, dict]] = {}
    for season in seasons:
        print(f"\nFetching ESPN team stats for {season}...")
        all_team_stats[season] = fetch_team_stats(season)
        n_teams = len(all_team_stats[season])
        print(f"  Got stats for {n_teams} teams")

    # Enrich each row
    enriched = 0
    for idx, row in df.iterrows():
        season = int(row["season"])
        home = row["home_team"]
        away = row["away_team"]

        stats = all_team_stats.get(season, {})
        home_stats = stats.get(home)
        away_stats = stats.get(away)

        if home_stats and away_stats:
            df.at[idx, "turnover_diff_diff"] = round(
                home_stats["turnover_diff"] - away_stats["turnover_diff"], 2
            )
            df.at[idx, "passer_rating_diff"] = round(
                home_stats["passer_rating"] - away_stats["passer_rating"], 3
            )
            df.at[idx, "third_down_diff"] = round(
                home_stats["third_down_pct"] - away_stats["third_down_pct"], 3
            )
            df.at[idx, "red_zone_diff"] = round(
                home_stats["red_zone_td_pct"] - away_stats["red_zone_td_pct"], 3
            )
            enriched += 1

    print(f"\nEnriched {enriched} / {len(df)} rows")

    # Show after state
    for col in ["turnover_diff_diff", "passer_rating_diff", "third_down_diff", "red_zone_diff"]:
        print(f"  AFTER  {col}: mean={df[col].mean():.4f}, std={df[col].std():.4f}, non-zero={(df[col] != 0).sum()}")

    # Save
    df.to_csv(CSV_PATH, index=False)
    print(f"\nSaved enriched data to {CSV_PATH}")


if __name__ == "__main__":
    main()
