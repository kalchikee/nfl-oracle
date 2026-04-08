// NFL Oracle v4.0 — Feature Engineering
// Computes 20+ features as home-vs-away differences (no lookahead bias).
// Sources: ESPN team stats, injury reports, weather, Elo, schedule rest.

import { logger } from '../logger.js';
import type { NFLGame, NFLTeamStats, FeatureVector } from '../types.js';
import {
  fetchAllTeamStats, fetchInjuries, fetchWeatherAtKickoff, computeWeatherAdj,
  isIndoorStadium, hasAltitude, isDivisionalGame, TEAM_VENUE_COORDS,
} from '../api/nflClient.js';
import { getEloDiff } from './eloEngine.js';

// Position impact weights for injury model (0–1 scale)
const POSITION_IMPACT: Record<string, number> = {
  QB:  1.00,  // highest impact — can shift spread by 5–10 pts
  LT:  0.30,  // blindside protector
  WR1: 0.25,  // top receiver
  TE:  0.20,
  RB:  0.18,
  DE:  0.15,  // pass rusher
  CB:  0.15,  // lockdown corner
  LB:  0.12,
  DT:  0.12,
  S:   0.10,
  RT:  0.10,
  G:   0.08,
  C:   0.08,
  WR:  0.15,
  OT:  0.20,
  K:   0.05,
};

const STATUS_PROB: Record<string, number> = {
  'Out': 1.0,
  'Doubtful': 0.75,
  'Questionable': 0.45,
  'Probable': 0.15,
};

// ─── Compute injury impact for a team ────────────────────────────────────────

function computeInjuryImpact(teamAbbr: string, injuryReports: Awaited<ReturnType<typeof fetchInjuries>>): { impact: number; qbOut: boolean } {
  const report = injuryReports.find(r => r.teamAbbr === teamAbbr);
  if (!report) return { impact: 0, qbOut: false };

  let totalImpact = 0;
  let qbOut = false;

  for (const player of report.players) {
    const pos = player.position.toUpperCase();
    const posWeight = POSITION_IMPACT[pos] ?? 0.05;
    const statusProb = STATUS_PROB[player.status] ?? 0.3;
    totalImpact += posWeight * statusProb;

    if (pos === 'QB' && (player.status === 'Out' || player.status === 'Doubtful')) {
      qbOut = true;
    }
  }

  return { impact: Math.min(totalImpact, 2.0), qbOut };
}

// ─── Rest days calculation ────────────────────────────────────────────────────

function computeRestDays(gameDate: string, lastGameDate: string | null): number {
  if (!lastGameDate) return 7; // assume full week rest
  const game = new Date(gameDate);
  const last = new Date(lastGameDate);
  const diffMs = game.getTime() - last.getTime();
  return Math.round(diffMs / (1000 * 60 * 60 * 24));
}

// For Thursday games after Sunday games: 4-day turnaround
// For regular Sunday games: 7 days
// For bye weeks: 14 days
function isShortWeek(restDays: number): boolean {
  return restDays <= 5;
}

// ─── Main feature computation ─────────────────────────────────────────────────

export async function computeFeatures(
  game: NFLGame,
  allTeamStats: Map<string, NFLTeamStats>,
  injuryReports: Awaited<ReturnType<typeof fetchInjuries>>,
  homeLastGameDate: string | null,
  awayLastGameDate: string | null,
): Promise<FeatureVector> {
  const homeAbbr = game.homeTeam.teamAbbr;
  const awayAbbr = game.awayTeam.teamAbbr;

  logger.debug({ home: homeAbbr, away: awayAbbr }, 'Computing features');

  const homeStats = allTeamStats.get(homeAbbr);
  const awayStats = allTeamStats.get(awayAbbr);

  if (!homeStats || !awayStats) {
    logger.warn({ home: homeAbbr, away: awayAbbr }, 'Missing team stats — using league averages');
  }

  const home = homeStats ?? defaultStats(homeAbbr);
  const away = awayStats ?? defaultStats(awayAbbr);

  // ── Elo ───────────────────────────────────────────────────────────────────
  const eloDiff = getEloDiff(homeAbbr, awayAbbr);

  // ── Team strength diffs ───────────────────────────────────────────────────
  const pythagoreanDiff = home.pythagoreanWinPct - away.pythagoreanWinPct;
  const winPctDiff = home.winPct - away.winPct;

  // ── Scoring ───────────────────────────────────────────────────────────────
  const ppgDiff = home.pointsPerGame - away.pointsPerGame;
  const papgDiff = home.pointsAllowedPerGame - away.pointsAllowedPerGame;
  const netPpgDiff = (home.pointsPerGame - home.pointsAllowedPerGame) - (away.pointsPerGame - away.pointsAllowedPerGame);

  // ── Yardage ───────────────────────────────────────────────────────────────
  const ydsPerPlayDiff = home.yardsPerPlay - away.yardsPerPlay;
  const defYdsPerPlayDiff = home.defYardsPerPlay - away.defYardsPerPlay;

  // ── Turnovers ─────────────────────────────────────────────────────────────
  const turnoverDiffDiff = home.turnoverDiff - away.turnoverDiff;

  // ── QB / Passing ──────────────────────────────────────────────────────────
  const passerRatingDiff = home.passerRating - away.passerRating;

  // ── Situational ───────────────────────────────────────────────────────────
  const thirdDownDiff = home.thirdDownConvPct - away.thirdDownConvPct;
  const redZoneDiff = home.redZoneTdPct - away.redZoneTdPct;

  // ── Injuries ──────────────────────────────────────────────────────────────
  const homeInj = computeInjuryImpact(homeAbbr, injuryReports);
  const awayInj = computeInjuryImpact(awayAbbr, injuryReports);
  const injuryImpactDiff = awayInj.impact - homeInj.impact; // positive = away more injured (home advantage)

  // ── Rest ──────────────────────────────────────────────────────────────────
  const homeRestDays = computeRestDays(game.gameDate, homeLastGameDate);
  const awayRestDays = computeRestDays(game.gameDate, awayLastGameDate);
  const restDaysDiff = homeRestDays - awayRestDays;
  const shortWeekHome = isShortWeek(homeRestDays) ? 1 : 0;
  const shortWeekAway = isShortWeek(awayRestDays) ? 1 : 0;

  // ── Weather ───────────────────────────────────────────────────────────────
  const indoor = game.venueIndoor || isIndoorStadium(homeAbbr);
  let windAdj = 0, precipAdj = 0, tempAdj = 0;

  if (!indoor) {
    const coords = TEAM_VENUE_COORDS[homeAbbr];
    if (coords) {
      const kickoffHour = parseInt(game.gameTime.split('T')[1]?.split(':')[0] ?? '18', 10);
      try {
        const weather = await fetchWeatherAtKickoff(coords.lat, coords.lon, game.gameDate, kickoffHour);
        const adj = computeWeatherAdj(weather, false);
        windAdj = adj.wind;
        precipAdj = adj.precip;
        tempAdj = adj.temp;
      } catch {
        // weather not critical — continue with defaults
      }
    }
  }

  // ── Altitude ──────────────────────────────────────────────────────────────
  const altitudeAdj = hasAltitude(homeAbbr) ? 0.3 : 0;

  // ── Game context ──────────────────────────────────────────────────────────
  const divisionalGame = isDivisionalGame(homeAbbr, awayAbbr) ? 1 : 0;
  const isNeutral = game.neutralSite ? 1 : 0;

  return {
    elo_diff: eloDiff,
    pythagorean_diff: pythagoreanDiff,
    win_pct_diff: winPctDiff,
    ppg_diff: ppgDiff,
    papg_diff: papgDiff,
    net_ppg_diff: netPpgDiff,
    yds_per_play_diff: ydsPerPlayDiff,
    def_yds_per_play_diff: defYdsPerPlayDiff,
    turnover_diff_diff: turnoverDiffDiff,
    passer_rating_diff: passerRatingDiff,
    third_down_diff: thirdDownDiff,
    red_zone_diff: redZoneDiff,
    injury_impact_diff: injuryImpactDiff,
    qb_out_home: homeInj.qbOut ? 1 : 0,
    qb_out_away: awayInj.qbOut ? 1 : 0,
    rest_days_diff: restDaysDiff,
    short_week_home: shortWeekHome,
    short_week_away: shortWeekAway,
    wind_adj: windAdj,
    precip_adj: precipAdj,
    temp_adj: tempAdj,
    is_dome: indoor ? 1 : 0,
    altitude_adj: altitudeAdj,
    divisional_game: divisionalGame,
    is_neutral: isNeutral,
    vegas_home_prob: 0, // filled later after odds lookup
    mc_win_pct: 0,      // filled later after Monte Carlo
  };
}

function defaultStats(abbr: string): NFLTeamStats {
  return {
    teamId: abbr,
    teamAbbr: abbr,
    teamName: abbr,
    wins: 8, losses: 9, ties: 0, winPct: 0.47,
    pointsPerGame: 22.5, pointsAllowedPerGame: 22.5,
    offPassYdsPerGame: 230, offRushYdsPerGame: 115, offTotalYdsPerGame: 345,
    offTurnoversPG: 1.3,
    thirdDownConvPct: 0.40, redZoneTdPct: 0.60, passerRating: 90.0,
    defPassYdsPerGame: 230, defRushYdsPerGame: 115, defTotalYdsPerGame: 345,
    defTurnoversPG: 1.3, sacksPerGame: 1.5,
    pythagoreanWinPct: 0.50,
    turnoverDiff: 0,
    yardsPerPlay: 5.3, defYardsPerPlay: 5.3,
  };
}

// ─── Re-export fetch helpers for pipeline use ─────────────────────────────────
export { fetchAllTeamStats, fetchInjuries };
