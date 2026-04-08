// NFL Oracle v4.0 — Elo Rating Engine
// NFL Elo: K=20, margin-of-victory cap, offseason regression to mean.
// Seeded from 2025 final ratings (carry into 2026 season).

import { getElo, upsertElo, getAllElos } from '../db/database.js';

const LEAGUE_MEAN_ELO = 1500;
const K_FACTOR = 20;
const OFFSEASON_REGRESSION = 0.67; // how much prior rating carries over

// ─── 2025 final season Elo seeds (approximate) ───────────────────────────────
// Based on 2024 season performance
const ELO_SEEDS_2025: Record<string, number> = {
  KC:  1620,  // 2x Super Bowl champs
  PHI: 1590,
  BAL: 1575,
  BUF: 1565,
  DET: 1555,
  SF:  1545,
  GB:  1535,
  HOU: 1525,
  MIN: 1520,
  WAS: 1510,
  LAC: 1505,
  LAR: 1500,
  TB:  1495,
  ATL: 1490,
  PIT: 1490,
  MIA: 1485,
  SEA: 1480,
  DEN: 1480,
  CIN: 1475,
  ARI: 1470,
  CHI: 1465,
  IND: 1460,
  NO:  1455,
  DAL: 1455,
  NE:  1445,
  LV:  1440,
  TEN: 1435,
  NYJ: 1430,
  JAX: 1425,
  CLE: 1420,
  NYG: 1415,
  CAR: 1405,
};

// ─── Seed Elos into DB (idempotent) ──────────────────────────────────────────

export function seedElos(): void {
  const existing = getAllElos();
  if (existing.length >= 32) return; // already seeded

  const now = new Date().toISOString();
  for (const [abbr, rating] of Object.entries(ELO_SEEDS_2025)) {
    // Apply offseason regression to mean for 2026
    const regressedRating = OFFSEASON_REGRESSION * rating + (1 - OFFSEASON_REGRESSION) * LEAGUE_MEAN_ELO;
    upsertElo({ teamAbbr: abbr, rating: regressedRating, updatedAt: now });
  }
}

// ─── Elo difference ───────────────────────────────────────────────────────────

export function getEloDiff(homeAbbr: string, awayAbbr: string): number {
  return getElo(homeAbbr) - getElo(awayAbbr);
}

// ─── Expected win probability from Elo ───────────────────────────────────────

export function eloWinProb(homeAbbr: string, awayAbbr: string, homeFieldAdv = 65): number {
  const homeElo = getElo(homeAbbr) + homeFieldAdv; // home field ~ 65 Elo points
  const awayElo = getElo(awayAbbr);
  return 1 / (1 + Math.pow(10, (awayElo - homeElo) / 400));
}

// ─── Elo update after game ───────────────────────────────────────────────────

export function updateElo(
  homeAbbr: string,
  awayAbbr: string,
  homeScore: number,
  awayScore: number
): void {
  const homeElo = getElo(homeAbbr);
  const awayElo = getElo(awayAbbr);

  const homeExpected = 1 / (1 + Math.pow(10, (awayElo - (homeElo + 65)) / 400));
  const homeActual = homeScore > awayScore ? 1 : homeScore < awayScore ? 0 : 0.5;

  // Margin of victory multiplier: log(1 + margin) capped at log(22)
  const margin = Math.abs(homeScore - awayScore);
  const movMultiplier = Math.log(1 + Math.min(margin, 21)) / Math.log(22);

  const homeChange = K_FACTOR * movMultiplier * (homeActual - homeExpected);

  const now = new Date().toISOString();
  upsertElo({ teamAbbr: homeAbbr, rating: homeElo + homeChange, updatedAt: now });
  upsertElo({ teamAbbr: awayAbbr, rating: awayElo - homeChange, updatedAt: now });
}

// ─── Log5 win probability from win% ──────────────────────────────────────────

export function log5Prob(homeWinPct: number, awayWinPct: number): number {
  const h = Math.max(0.01, Math.min(0.99, homeWinPct));
  const a = Math.max(0.01, Math.min(0.99, awayWinPct));
  return (h - h * a) / (h + a - 2 * h * a);
}
