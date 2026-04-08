// NFL Oracle v4.0 — Monte Carlo Simulation Engine
// 10,000 Normal distribution simulations
// NFL scores: ~22 pts avg per team, ~9 pt std (from historical data)

import type { FeatureVector, MonteCarloResult } from '../types.js';

const N_SIMULATIONS = 10_000;
const LEAGUE_AVG_PPG = 22.5;    // points per team per game
const LEAGUE_AVG_SCORE_STD = 9.0;
const HOME_FIELD_ADV_PTS = 2.5; // home field advantage in points

// ─── Normal random (Box-Muller) ───────────────────────────────────────────────

function normalRandom(mean: number, std: number): number {
  const u1 = Math.random();
  const u2 = Math.random();
  const z = Math.sqrt(-2 * Math.log(Math.max(1e-10, u1))) * Math.cos(2 * Math.PI * u2);
  return mean + z * std;
}

// ─── Expected points estimation ───────────────────────────────────────────────
//
// Uses feature diffs to reconstruct team-level expected points.
// exp_pts = league_avg + (net_ppg_diff/2) + adjustments

export function estimateExpectedPoints(features: FeatureVector): { homeExpPts: number; awayExpPts: number; homeStd: number; awayStd: number } {
  // Reconstruct from net PPG diff
  const homeNetPPG = LEAGUE_AVG_PPG + features.net_ppg_diff / 2;
  const awayNetPPG = LEAGUE_AVG_PPG - features.net_ppg_diff / 2;

  // Further refine using individual scoring stats (if both agree, higher confidence)
  // Offensive scoring advantage: home ppg diff / 2
  const homeOffBonus = features.ppg_diff / 2;
  const awayOffBonus = -features.ppg_diff / 2;

  // Defensive advantage: home papg diff is negative if home defense is better
  const homeDefBonus = -features.papg_diff / 2;  // lower papg = better
  const awayDefBonus = features.papg_diff / 2;

  let homeExpPts = LEAGUE_AVG_PPG + (homeOffBonus + homeDefBonus) / 2;
  let awayExpPts = LEAGUE_AVG_PPG + (awayOffBonus + awayDefBonus) / 2;

  // Blend with net PPG approach (equal weight)
  homeExpPts = (homeExpPts + homeNetPPG) / 2;
  awayExpPts = (awayExpPts + awayNetPPG) / 2;

  // Home field advantage
  if (!features.is_neutral) {
    homeExpPts += HOME_FIELD_ADV_PTS / 2;
    awayExpPts -= HOME_FIELD_ADV_PTS / 2;
  }

  // Rest adjustment: each extra day of rest beyond 7 ~ +0.3 pts; short week ~ -1.5 pts
  if (features.short_week_home) homeExpPts -= 1.5;
  if (features.short_week_away) awayExpPts -= 1.5;
  const restBonus = Math.max(-2.0, Math.min(2.0, features.rest_days_diff * 0.25));
  homeExpPts += restBonus / 2;
  awayExpPts -= restBonus / 2;

  // QB impact: losing a starter (doubtful/out) = -4 to -8 pts
  if (features.qb_out_home) homeExpPts -= 6.0;
  if (features.qb_out_away) awayExpPts -= 6.0;

  // General injury adjustment (normalized, ~2 pts max each side)
  homeExpPts -= features.injury_impact_diff > 0 ? 0 : Math.abs(features.injury_impact_diff) * 1.5;
  awayExpPts -= features.injury_impact_diff < 0 ? 0 : features.injury_impact_diff * 1.5;

  // Weather adjustments (apply symmetrically — hurts both teams, but passing more)
  const weatherPenalty = features.wind_adj * 3.5 + features.precip_adj * 2.0 + features.temp_adj * 1.5;
  homeExpPts -= weatherPenalty / 2;
  awayExpPts -= weatherPenalty / 2;

  // Altitude: away team penalty at Denver
  if (features.altitude_adj > 0) {
    awayExpPts -= features.altitude_adj * 2.5;
  }

  // Floor at 7 (safety only game) and cap at 50 (historic high)
  homeExpPts = Math.max(7, Math.min(50, homeExpPts));
  awayExpPts = Math.max(7, Math.min(50, awayExpPts));

  // Adjust score std for weather (windy/cold games have more variance due to field goals)
  const weatherStdAdj = 1 + (features.wind_adj * 0.5 + features.precip_adj * 0.3);
  const homeStd = LEAGUE_AVG_SCORE_STD * weatherStdAdj;
  const awayStd = LEAGUE_AVG_SCORE_STD * weatherStdAdj;

  return { homeExpPts, awayExpPts, homeStd, awayStd };
}

// ─── Monte Carlo simulation ───────────────────────────────────────────────────

export function runMonteCarlo(features: FeatureVector): MonteCarloResult {
  const { homeExpPts, awayExpPts, homeStd, awayStd } = estimateExpectedPoints(features);

  let homeWins = 0;
  let totalHome = 0;
  let totalAway = 0;
  let blowouts = 0;
  let ties = 0;

  for (let i = 0; i < N_SIMULATIONS; i++) {
    let homeScore = Math.round(Math.max(0, normalRandom(homeExpPts, homeStd)));
    let awayScore = Math.round(Math.max(0, normalRandom(awayExpPts, awayStd)));

    // NFL ties go to OT. Current rules: each team gets a possession unless opening TD
    if (homeScore === awayScore) {
      ties++;
      // Simple OT: ~60% chance the first possession team scores (either TD or FG ends it)
      const otWinner = Math.random() < 0.55 ? 'home' : 'away';
      if (otWinner === 'home') {
        homeScore += Math.random() < 0.5 ? 7 : 3;
      } else {
        awayScore += Math.random() < 0.5 ? 7 : 3;
      }
    }

    if (homeScore > awayScore) homeWins++;
    totalHome += homeScore;
    totalAway += awayScore;
    if (Math.abs(homeScore - awayScore) >= 14) blowouts++;
  }

  const winProbability = homeWins / N_SIMULATIONS;
  const avgHome = totalHome / N_SIMULATIONS;
  const avgAway = totalAway / N_SIMULATIONS;

  const isHomeFavored = features.elo_diff > 0;
  const upsetProb = isHomeFavored ? 1 - winProbability : winProbability;

  return {
    win_probability: winProbability,
    away_win_probability: 1 - winProbability,
    spread: avgHome - avgAway,
    total_points: avgHome + avgAway,
    most_likely_score: [Math.round(homeExpPts), Math.round(awayExpPts)],
    upset_probability: upsetProb,
    blowout_probability: blowouts / N_SIMULATIONS,
    home_exp_pts: homeExpPts,
    away_exp_pts: awayExpPts,
    simulations: N_SIMULATIONS,
  };
}
