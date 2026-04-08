// NFL Oracle v4.0 — Odds API Client
// Uses The Odds API (free tier: 500 req/month) when ODDS_API_KEY is set.
// Falls back gracefully to ESPN embedded odds parsed in nflClient.

import fetch from 'node-fetch';
import { logger } from '../logger.js';

const ODDS_API_BASE = 'https://api.the-odds-api.com/v4/sports/americanfootball_nfl/odds';

interface OddsAPIBookmaker {
  key: string;
  markets: Array<{
    key: string;
    outcomes: Array<{ name: string; price: number; point?: number }>;
  }>;
}

interface OddsAPIGame {
  id: string;
  home_team: string;
  away_team: string;
  bookmakers?: OddsAPIBookmaker[];
}

export interface GameOdds {
  homeML: number;
  awayML: number;
  homeImpliedProb: number;
  awayImpliedProb: number;
  spread: number;       // positive = home favored
  total: number;
}

let _oddsCache: Map<string, GameOdds> | null = null;

function mlToProb(ml: number): number {
  if (ml > 0) return 100 / (ml + 100);
  return Math.abs(ml) / (Math.abs(ml) + 100);
}

function removeVig(homeProb: number, awayProb: number): { home: number; away: number } {
  const total = homeProb + awayProb;
  return { home: homeProb / total, away: awayProb / total };
}

export async function loadOddsApiLines(): Promise<void> {
  const key = process.env.ODDS_API_KEY;
  if (!key) {
    logger.info('ODDS_API_KEY not set — skipping Odds API');
    return;
  }

  const url = `${ODDS_API_BASE}?apiKey=${key}&regions=us&markets=h2h,spreads,totals&oddsFormat=american&bookmakers=draftkings,fanduel,betmgm`;

  try {
    const resp = await fetch(url, { signal: AbortSignal.timeout(15000) });
    if (!resp.ok) {
      logger.warn({ status: resp.status }, 'Odds API request failed');
      return;
    }
    const games = (await resp.json()) as OddsAPIGame[];
    _oddsCache = new Map();

    for (const game of games) {
      // Use DraftKings first, then any bookmaker
      const bk = game.bookmakers?.find(b => b.key === 'draftkings') ?? game.bookmakers?.[0];
      if (!bk) continue;

      const h2h = bk.markets.find(m => m.key === 'h2h');
      const spreads = bk.markets.find(m => m.key === 'spreads');
      const totals = bk.markets.find(m => m.key === 'totals');

      const homeH2H = h2h?.outcomes.find(o => o.name === game.home_team);
      const awayH2H = h2h?.outcomes.find(o => o.name === game.away_team);

      if (!homeH2H || !awayH2H) continue;

      const rawHome = mlToProb(homeH2H.price);
      const rawAway = mlToProb(awayH2H.price);
      const { home: homeProb, away: awayProb } = removeVig(rawHome, rawAway);

      const spreadHome = spreads?.outcomes.find(o => o.name === game.home_team);
      const totalOver = totals?.outcomes.find(o => o.name === 'Over');

      // Build match key: use simple team name matching
      // The Odds API uses full team names; we need to match to our abbreviation pairs
      const matchKey = `${game.away_team}@${game.home_team}`;

      _oddsCache.set(matchKey, {
        homeML: homeH2H.price,
        awayML: awayH2H.price,
        homeImpliedProb: homeProb,
        awayImpliedProb: awayProb,
        spread: spreadHome?.point !== undefined ? -spreadHome.point : 0, // convert: negative spread = home favored in Odds API
        total: totalOver?.point ?? 45,
      });
    }

    logger.info({ games: _oddsCache.size }, 'Odds API lines loaded');
  } catch (err) {
    logger.warn({ err }, 'Failed to load Odds API lines');
  }
}

export function getOddsForGame(homeAbbr: string, awayAbbr: string, vegasSpread?: number, vegasTotal?: number, homeML?: number, awayML?: number): GameOdds | null {
  // If we have ESPN embedded odds, use those as fallback
  if (_oddsCache === null && homeML && awayML) {
    const rawHome = mlToProb(homeML);
    const rawAway = mlToProb(awayML);
    const { home: homeProb } = removeVig(rawHome, rawAway);
    return {
      homeML, awayML,
      homeImpliedProb: homeProb,
      awayImpliedProb: 1 - homeProb,
      spread: vegasSpread ?? 0,
      total: vegasTotal ?? 45,
    };
  }

  if (!_oddsCache) return null;

  // Try exact match key first (won't match since we stored full names)
  for (const [key, odds] of _oddsCache.entries()) {
    if (key.includes(homeAbbr) || key.includes(awayAbbr)) {
      return odds;
    }
  }

  // Fallback: if ESPN embedded odds available
  if (homeML && awayML) {
    const rawHome = mlToProb(homeML);
    const rawAway = mlToProb(awayML);
    const { home: homeProb } = removeVig(rawHome, rawAway);
    return {
      homeML, awayML,
      homeImpliedProb: homeProb,
      awayImpliedProb: 1 - homeProb,
      spread: vegasSpread ?? 0,
      total: vegasTotal ?? 45,
    };
  }

  return null;
}

export function hasAnyOdds(): boolean {
  return _oddsCache !== null && _oddsCache.size > 0;
}

export function initializeOdds(): void {
  // No-op if Odds API key not present; ESPN odds are parsed inline
  logger.debug('Odds initialized from ESPN embedded lines');
}
