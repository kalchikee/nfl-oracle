// NFL Oracle v4.0 — ESPN NFL API Client
// Free API — no key required.
// Covers: schedule, team stats, injuries, completed results.
// JSON file caching with TTL + exponential backoff retry.

import { mkdirSync, readFileSync, writeFileSync, existsSync, statSync } from 'fs';
import { resolve, dirname } from 'path';
import { fileURLToPath } from 'url';
import fetch from 'node-fetch';
import { logger } from '../logger.js';
import type { NFLGame, NFLGameTeam, NFLTeamStats, InjuryReport, InjuredPlayer, GameResult } from '../types.js';

const __filename = fileURLToPath(import.meta.url);
const __dirname = dirname(__filename);

const CACHE_DIR = process.env.CACHE_DIR ?? resolve(__dirname, '../../cache');
const CACHE_TTL_MS = (Number(process.env.CACHE_TTL_HOURS ?? 4)) * 60 * 60 * 1000;

mkdirSync(CACHE_DIR, { recursive: true });

const ESPN_BASE = 'https://site.api.espn.com/apis/site/v2/sports/football/nfl';

// ─── NFL team abbreviation → ESPN ID mapping ──────────────────────────────────

export const ABBR_TO_ESPN_ID: Record<string, string> = {
  ARI: '22', ATL: '1',  BAL: '33', BUF: '2',  CAR: '29', CHI: '3',
  CIN: '4',  CLE: '5',  DAL: '6',  DEN: '7',  DET: '8',  GB: '9',
  HOU: '34', IND: '11', JAX: '30', KC: '12',  LAC: '24', LAR: '14',
  LV: '13',  MIA: '15', MIN: '16', NE: '17',  NO: '18',  NYG: '19',
  NYJ: '20', PHI: '21', PIT: '23', SEA: '26', SF: '25',  TB: '27',
  TEN: '10', WAS: '28',
};

export const ESPN_ID_TO_ABBR: Record<string, string> = Object.fromEntries(
  Object.entries(ABBR_TO_ESPN_ID).map(([abbr, id]) => [id, abbr])
);

// Division lookup (for divisional game detection)
const DIVISIONS: Record<string, string> = {
  BUF: 'AFC_EAST', MIA: 'AFC_EAST', NE: 'AFC_EAST', NYJ: 'AFC_EAST',
  BAL: 'AFC_NORTH', CIN: 'AFC_NORTH', CLE: 'AFC_NORTH', PIT: 'AFC_NORTH',
  HOU: 'AFC_SOUTH', IND: 'AFC_SOUTH', JAX: 'AFC_SOUTH', TEN: 'AFC_SOUTH',
  DEN: 'AFC_WEST', KC: 'AFC_WEST', LV: 'AFC_WEST', LAC: 'AFC_WEST',
  DAL: 'NFC_EAST', NYG: 'NFC_EAST', PHI: 'NFC_EAST', WAS: 'NFC_EAST',
  CHI: 'NFC_NORTH', DET: 'NFC_NORTH', GB: 'NFC_NORTH', MIN: 'NFC_NORTH',
  ATL: 'NFC_SOUTH', CAR: 'NFC_SOUTH', NO: 'NFC_SOUTH', TB: 'NFC_SOUTH',
  ARI: 'NFC_WEST', LAR: 'NFC_WEST', SF: 'NFC_WEST', SEA: 'NFC_WEST',
};

export function isDivisionalGame(teamA: string, teamB: string): boolean {
  return !!DIVISIONS[teamA] && DIVISIONS[teamA] === DIVISIONS[teamB];
}

// Indoor / covered stadiums (weather doesn't apply)
const DOME_TEAMS = new Set(['ARI', 'ATL', 'DAL', 'DET', 'HOU', 'IND', 'LV', 'MIN', 'NO', 'LAR', 'LAC']);

export function isIndoorStadium(homeTeamAbbr: string): boolean {
  return DOME_TEAMS.has(homeTeamAbbr);
}

// Denver altitude disadvantage for away team
const ALTITUDE_TEAMS = new Set(['DEN']);
export function hasAltitude(homeTeamAbbr: string): boolean {
  return ALTITUDE_TEAMS.has(homeTeamAbbr);
}

// ─── Venue geocoordinates for weather lookup ──────────────────────────────────

export const TEAM_VENUE_COORDS: Record<string, { lat: number; lon: number }> = {
  ARI: { lat: 33.53,  lon: -112.26 }, ATL: { lat: 33.76,  lon: -84.40  },
  BAL: { lat: 39.28,  lon: -76.62  }, BUF: { lat: 42.77,  lon: -78.79  },
  CAR: { lat: 35.23,  lon: -80.85  }, CHI: { lat: 41.86,  lon: -87.62  },
  CIN: { lat: 39.09,  lon: -84.52  }, CLE: { lat: 41.50,  lon: -81.70  },
  DAL: { lat: 32.75,  lon: -97.09  }, DEN: { lat: 39.74,  lon: -105.02 },
  DET: { lat: 42.34,  lon: -83.05  }, GB:  { lat: 44.50,  lon: -88.06  },
  HOU: { lat: 29.68,  lon: -95.41  }, IND: { lat: 39.76,  lon: -86.16  },
  JAX: { lat: 30.32,  lon: -81.64  }, KC:  { lat: 39.05,  lon: -94.48  },
  LAC: { lat: 33.95,  lon: -118.34 }, LAR: { lat: 33.95,  lon: -118.34 },
  LV:  { lat: 36.09,  lon: -115.18 }, MIA: { lat: 25.96,  lon: -80.24  },
  MIN: { lat: 44.97,  lon: -93.26  }, NE:  { lat: 42.09,  lon: -71.26  },
  NO:  { lat: 29.95,  lon: -90.08  }, NYG: { lat: 40.81,  lon: -74.07  },
  NYJ: { lat: 40.81,  lon: -74.07  }, PHI: { lat: 39.90,  lon: -75.17  },
  PIT: { lat: 40.45,  lon: -80.02  }, SEA: { lat: 47.60,  lon: -122.33 },
  SF:  { lat: 37.40,  lon: -121.97 }, TB:  { lat: 27.97,  lon: -82.50  },
  TEN: { lat: 36.17,  lon: -86.77  }, WAS: { lat: 38.91,  lon: -76.86  },
};

// ─── Cache helpers ─────────────────────────────────────────────────────────────

function cacheKey(url: string): string {
  return url.replace(/[^a-zA-Z0-9]/g, '_').slice(0, 200) + '.json';
}

function readCache<T>(key: string): T | null {
  const path = resolve(CACHE_DIR, key);
  if (!existsSync(path)) return null;
  const stat = statSync(path);
  if (Date.now() - stat.mtimeMs > CACHE_TTL_MS) return null;
  try {
    return JSON.parse(readFileSync(path, 'utf-8')) as T;
  } catch {
    return null;
  }
}

function writeCache(key: string, data: unknown): void {
  try {
    writeFileSync(resolve(CACHE_DIR, key), JSON.stringify(data), 'utf-8');
  } catch (err) {
    logger.warn({ err }, 'Failed to write cache');
  }
}

// ─── Retry fetch ──────────────────────────────────────────────────────────────

async function fetchWithRetry<T>(url: string, attempts = 3, bypassCache = false): Promise<T> {
  const key = cacheKey(url);
  if (!bypassCache) {
    const cached = readCache<T>(key);
    if (cached !== null) {
      logger.debug({ url }, 'Cache HIT');
      return cached;
    }
  }

  let lastError: Error | null = null;
  for (let attempt = 0; attempt < attempts; attempt++) {
    try {
      logger.debug({ url, attempt }, 'Fetching');
      const resp = await fetch(url, {
        headers: { 'User-Agent': 'NFLOracle/4.0 (educational)' },
        signal: AbortSignal.timeout(20000),
      });
      if (!resp.ok) throw new Error(`HTTP ${resp.status} for ${url}`);
      const data = (await resp.json()) as T;
      writeCache(key, data);
      return data;
    } catch (err) {
      lastError = err instanceof Error ? err : new Error(String(err));
      if (attempt < attempts - 1) {
        const delay = Math.pow(2, attempt) * 1000;
        logger.warn({ url, attempt, delay, err: lastError.message }, 'Retrying');
        await new Promise(r => setTimeout(r, delay));
      }
    }
  }
  throw lastError ?? new Error(`Failed to fetch ${url}`);
}

// ─── Season / week helpers ────────────────────────────────────────────────────

export function getCurrentNFLSeason(): number {
  const now = new Date();
  const month = now.getMonth() + 1;
  const year = now.getFullYear();
  // NFL season spans Aug-Feb. Season labeled by start year (e.g., "2026" season = Sept 2026 - Feb 2027)
  return month >= 8 ? year : year - 1;
}

export function isNFLSeason(date: Date = new Date()): boolean {
  const month = date.getMonth() + 1;
  if (month >= 9) return true;   // September - December
  if (month === 1) return true;   // January (playoffs)
  if (month === 2 && date.getDate() <= 15) return true; // Super Bowl (first/second Sunday in Feb)
  return false;
}

// ─── Scoreboard / Schedule ────────────────────────────────────────────────────

interface ESPNScoreboardResponse {
  week?: { number: number };
  season?: { year: number; type: number };
  events?: ESPNEvent[];
}

interface ESPNEvent {
  id: string;
  date: string;
  week?: { number: number };
  season?: { year: number; type: number };
  status?: { type?: { name?: string; description?: string } };
  competitions?: ESPNCompetition[];
}

interface ESPNCompetition {
  id: string;
  competitors?: ESPNCompetitor[];
  venue?: { fullName?: string; address?: { city?: string }; indoor?: boolean };
  neutralSite?: boolean;
  odds?: Array<{
    details?: string;      // e.g. "BAL -3.5"
    overUnder?: number;
    homeTeamOdds?: { moneyLine?: number };
    awayTeamOdds?: { moneyLine?: number };
    spread?: number;
  }>;
}

interface ESPNCompetitor {
  id: string;
  homeAway: string;
  team: { id?: string; abbreviation?: string; displayName?: string };
  score?: string;
}

function parseESPNEvent(event: ESPNEvent, defaultSeason: number): NFLGame | null {
  const comp = event.competitions?.[0];
  if (!comp?.competitors || comp.competitors.length < 2) return null;

  const home = comp.competitors.find(c => c.homeAway === 'home');
  const away = comp.competitors.find(c => c.homeAway === 'away');
  if (!home || !away) return null;

  const homeAbbr = ESPN_ID_TO_ABBR[home.team.id ?? ''] ?? home.team.abbreviation ?? 'UNK';
  const awayAbbr = ESPN_ID_TO_ABBR[away.team.id ?? ''] ?? away.team.abbreviation ?? 'UNK';

  const homeTeam: NFLGameTeam = {
    teamId: home.team.id ?? '',
    teamAbbr: homeAbbr,
    teamName: home.team.displayName ?? homeAbbr,
    score: home.score !== undefined ? Number(home.score) : undefined,
  };
  const awayTeam: NFLGameTeam = {
    teamId: away.team.id ?? '',
    teamAbbr: awayAbbr,
    teamName: away.team.displayName ?? awayAbbr,
    score: away.score !== undefined ? Number(away.score) : undefined,
  };

  // Parse Vegas odds
  const odds = comp.odds?.[0];
  let vegasSpread: number | undefined;
  let vegasTotal: number | undefined;
  let homeML: number | undefined;
  let awayML: number | undefined;

  if (odds) {
    vegasTotal = odds.overUnder ?? undefined;
    homeML = odds.homeTeamOdds?.moneyLine ?? undefined;
    awayML = odds.awayTeamOdds?.moneyLine ?? undefined;
    // Parse spread from details string, e.g. "BAL -3.5" means BAL favored by 3.5
    if (odds.details) {
      const parts = odds.details.split(' ');
      if (parts.length >= 2) {
        const favAbbr = parts[0];
        const spreadVal = parseFloat(parts[1]);
        if (!isNaN(spreadVal)) {
          vegasSpread = favAbbr === homeAbbr ? -spreadVal : spreadVal; // positive = home favored
        }
      }
    }
  }

  const gameDate = event.date.split('T')[0];

  return {
    gameId: event.id,
    gameDate,
    gameTime: event.date,
    week: event.week?.number ?? 0,
    season: event.season?.year ?? defaultSeason,
    seasonType: event.season?.type ?? 2,
    status: event.status?.type?.name ?? 'STATUS_SCHEDULED',
    homeTeam,
    awayTeam,
    venueName: comp.venue?.fullName ?? '',
    venueCity: comp.venue?.address?.city ?? '',
    venueIndoor: comp.venue?.indoor ?? isIndoorStadium(homeAbbr),
    neutralSite: comp.neutralSite ?? false,
    vegasSpread,
    vegasTotal,
    homeMoneyLine: homeML,
    awayMoneyLine: awayML,
  };
}

// Fetch games for a specific week
export async function fetchWeekSchedule(week: number, season: number, seasonType = 2): Promise<NFLGame[]> {
  const url = `${ESPN_BASE}/scoreboard?seasontype=${seasonType}&week=${week}&dates=${season}&limit=50`;
  try {
    const data = await fetchWithRetry<ESPNScoreboardResponse>(url);
    if (!data.events) return [];
    const games: NFLGame[] = [];
    for (const event of data.events) {
      const game = parseESPNEvent(event, season);
      if (game) games.push(game);
    }
    logger.info({ week, season, games: games.length }, 'Week schedule fetched');
    return games;
  } catch (err) {
    logger.warn({ err, week, season }, 'Failed to fetch week schedule');
    return [];
  }
}

// Fetch games for a date range (used for recap — look at last 7 days)
export async function fetchScheduleForDates(startDate: string, endDate: string): Promise<NFLGame[]> {
  const season = getCurrentNFLSeason();
  // Try fetching each date in the range via the scoreboard
  const games: NFLGame[] = [];
  const seen = new Set<string>();
  let current = new Date(startDate);
  const end = new Date(endDate);

  while (current <= end) {
    const dateStr = current.toISOString().split('T')[0].replace(/-/g, '');
    const url = `${ESPN_BASE}/scoreboard?dates=${dateStr}&limit=50`;
    try {
      const data = await fetchWithRetry<ESPNScoreboardResponse>(url);
      if (data.events) {
        for (const event of data.events) {
          if (!seen.has(event.id)) {
            seen.add(event.id);
            const game = parseESPNEvent(event, season);
            if (game) games.push(game);
          }
        }
      }
    } catch (err) {
      logger.debug({ err, date: dateStr }, 'Failed to fetch scoreboard for date');
    }
    current.setDate(current.getDate() + 1);
  }

  return games;
}

// Determine current and upcoming week
export async function getCurrentWeekInfo(): Promise<{ week: number; season: number }> {
  const season = getCurrentNFLSeason();
  const url = `${ESPN_BASE}/scoreboard?limit=5`;
  try {
    const data = await fetchWithRetry<ESPNScoreboardResponse>(url, 3, true); // bypass cache for real-time week
    const week = data.week?.number ?? 1;
    return { week, season };
  } catch {
    return { week: 1, season };
  }
}

// ─── Team stats ───────────────────────────────────────────────────────────────

interface ESPNTeamStatsResponse {
  team?: { id?: string; abbreviation?: string; displayName?: string };
  results?: { splits?: { categories?: ESPNStatCategory[] } };
}

interface ESPNStatCategory {
  name?: string;
  stats?: Array<{ name?: string; displayValue?: string; value?: number }>;
}

let _teamStatsCache: Map<string, NFLTeamStats> | null = null;
let _teamStatsCacheTime = 0;

export async function fetchAllTeamStats(season?: number): Promise<Map<string, NFLTeamStats>> {
  const now = Date.now();
  if (_teamStatsCache && now - _teamStatsCacheTime < CACHE_TTL_MS) {
    return _teamStatsCache;
  }

  const s = season ?? getCurrentNFLSeason();
  const teamMap = new Map<string, NFLTeamStats>();

  // Fetch all 32 teams
  const allAbbrs = Object.keys(ABBR_TO_ESPN_ID);

  await Promise.allSettled(
    allAbbrs.map(async abbr => {
      const id = ABBR_TO_ESPN_ID[abbr];
      const url = `${ESPN_BASE}/teams/${id}/statistics?season=${s}`;
      try {
        const data = await fetchWithRetry<ESPNTeamStatsResponse>(url);
        const stats = parseTeamStats(abbr, data);
        teamMap.set(abbr, stats);
      } catch (err) {
        logger.debug({ err, abbr }, 'Failed to fetch team stats — using defaults');
        teamMap.set(abbr, defaultTeamStats(abbr));
      }
    })
  );

  _teamStatsCache = teamMap;
  _teamStatsCacheTime = now;
  logger.info({ teams: teamMap.size, season: s }, 'NFL team stats loaded');
  return teamMap;
}

function getStat(categories: ESPNStatCategory[], categoryName: string, statName: string): number {
  const cat = categories.find(c => c.name?.toLowerCase() === categoryName.toLowerCase());
  if (!cat) return 0;
  const stat = cat.stats?.find(s => s.name?.toLowerCase() === statName.toLowerCase());
  return stat?.value ?? 0;
}

function parseTeamStats(abbr: string, data: ESPNTeamStatsResponse): NFLTeamStats {
  const categories = data.results?.splits?.categories ?? [];

  const wins = getStat(categories, 'record', 'wins');
  const losses = getStat(categories, 'record', 'losses');
  const ties = getStat(categories, 'record', 'ties');
  const gp = wins + losses + ties || 1;
  const winPct = (wins + ties * 0.5) / gp;

  const ppg = getStat(categories, 'scoring', 'totalPointsPerGame') ||
              getStat(categories, 'scoring', 'avgPoints') || 22;
  const papg = getStat(categories, 'scoring', 'opponentTotalPointsPerGame') ||
               getStat(categories, 'scoring', 'opponentAvgPoints') || 22;

  const offPassYds = getStat(categories, 'passing', 'passingYardsPerGame') ||
                     getStat(categories, 'passing', 'avgPassingYards') || 230;
  const offRushYds = getStat(categories, 'rushing', 'rushingYardsPerGame') ||
                     getStat(categories, 'rushing', 'avgRushingYards') || 115;
  const defPassYds = getStat(categories, 'defensivePassing', 'passingYardsPerGame') || 230;
  const defRushYds = getStat(categories, 'defensiveRushing', 'rushingYardsPerGame') || 115;

  const offTotalYds = offPassYds + offRushYds || 350;
  const defTotalYds = defPassYds + defRushYds || 350;
  const yardsPerPlay = offTotalYds / 65; // approx 65 plays/game
  const defYardsPerPlay = defTotalYds / 65;

  const offTurnovers = getStat(categories, 'turnovers', 'lostFumblesPerGame') ||
                       getStat(categories, 'turnovers', 'turnoversPerGame') || 1.3;
  const defTurnovers = getStat(categories, 'defensiveTurnovers', 'turnoversPerGame') ||
                       getStat(categories, 'turnovers', 'oppTurnoversPerGame') || 1.3;

  const thirdDown = getStat(categories, 'thirdDownConversions', 'thirdDownConversionPct') ||
                    getStat(categories, 'efficiency', 'thirdDownConversionPct') || 0.40;
  const redZone = getStat(categories, 'redZoneConversions', 'redZonePct') ||
                  getStat(categories, 'redZone', 'redZonePct') || 0.60;
  const passerRating = getStat(categories, 'passing', 'ratingPerGame') ||
                       getStat(categories, 'passing', 'QBRating') ||
                       getStat(categories, 'passerRating', 'rating') || 90;

  const sacks = getStat(categories, 'defensiveSacks', 'sacks') ||
                getStat(categories, 'defense', 'sacks') || 1.5;

  // Pythagorean win expectation (NFL exponent ≈ 2.37)
  const exp = 2.37;
  const pythagorean = ppg > 0 && papg > 0
    ? Math.pow(ppg, exp) / (Math.pow(ppg, exp) + Math.pow(papg, exp))
    : 0.5;

  return {
    teamId: ABBR_TO_ESPN_ID[abbr] ?? '',
    teamAbbr: abbr,
    teamName: abbr,
    wins, losses, ties, winPct,
    pointsPerGame: ppg,
    pointsAllowedPerGame: papg,
    offPassYdsPerGame: offPassYds,
    offRushYdsPerGame: offRushYds,
    offTotalYdsPerGame: offTotalYds,
    offTurnoversPG: offTurnovers,
    thirdDownConvPct: thirdDown > 1 ? thirdDown / 100 : thirdDown,
    redZoneTdPct: redZone > 1 ? redZone / 100 : redZone,
    passerRating,
    defPassYdsPerGame: defPassYds,
    defRushYdsPerGame: defRushYds,
    defTotalYdsPerGame: defTotalYds,
    defTurnoversPG: defTurnovers,
    sacksPerGame: sacks,
    pythagoreanWinPct: pythagorean,
    turnoverDiff: defTurnovers - offTurnovers,
    yardsPerPlay,
    defYardsPerPlay,
  };
}

function defaultTeamStats(abbr: string): NFLTeamStats {
  return {
    teamId: ABBR_TO_ESPN_ID[abbr] ?? '',
    teamAbbr: abbr, teamName: abbr,
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

// ─── Injury reports ───────────────────────────────────────────────────────────

interface ESPNInjuryRoot {
  items?: Array<{
    team?: { abbreviation?: string };
    injuries?: Array<{
      athlete?: { id?: string; displayName?: string; position?: { abbreviation?: string } };
      status?: string;
      shortComment?: string;
    }>;
  }>;
}

export async function fetchInjuries(): Promise<InjuryReport[]> {
  const url = `${ESPN_BASE}/injuries`;
  try {
    const data = await fetchWithRetry<ESPNInjuryRoot>(url);
    const reportMap = new Map<string, InjuredPlayer[]>();

    if (data.items) {
      for (const item of data.items) {
        const abbr = item.team?.abbreviation ?? 'UNK';
        if (!item.injuries) continue;
        for (const inj of item.injuries) {
          if (!inj.athlete) continue;
          const status = inj.status ?? 'Questionable';
          if (!['Out', 'Doubtful', 'Questionable', 'Probable'].includes(status)) continue;
          const pos = inj.athlete.position?.abbreviation ?? '';
          if (!reportMap.has(abbr)) reportMap.set(abbr, []);
          reportMap.get(abbr)!.push({
            playerId: inj.athlete.id ?? '',
            playerName: inj.athlete.displayName ?? 'Unknown',
            position: pos,
            status,
            description: inj.shortComment ?? '',
          });
        }
      }
    }

    const reports: InjuryReport[] = [];
    for (const [abbr, players] of reportMap.entries()) {
      reports.push({ teamAbbr: abbr, players });
    }
    logger.info({ teams: reports.length }, 'Injury reports fetched');
    return reports;
  } catch (err) {
    logger.warn({ err }, 'Failed to fetch injuries — continuing without');
    return [];
  }
}

// ─── Completed results (for recap) ───────────────────────────────────────────

export async function fetchCompletedResults(startDate: string, endDate: string): Promise<GameResult[]> {
  const games = await fetchScheduleForDates(startDate, endDate);
  const results: GameResult[] = [];

  for (const game of games) {
    const status = game.status;
    if (!status.includes('FINAL') && !status.includes('final') && !status.includes('STATUS_FINAL')) continue;
    const homeScore = game.homeTeam.score;
    const awayScore = game.awayTeam.score;
    if (homeScore === undefined || awayScore === undefined) continue;

    results.push({
      game_id: game.gameId,
      date: game.gameDate,
      week: game.week,
      season: game.season,
      home_team: game.homeTeam.teamAbbr,
      away_team: game.awayTeam.teamAbbr,
      home_score: homeScore,
      away_score: awayScore,
      venue: game.venueName,
    });
  }

  return results;
}

// ─── Weather (Open-Meteo — free, no key) ──────────────────────────────────────

interface OpenMeteoResponse {
  hourly?: {
    time?: string[];
    windspeed_10m?: number[];
    precipitation?: number[];
    temperature_2m?: number[];
  };
}

export interface WeatherData {
  windSpeedMph: number;
  precipInches: number;
  tempF: number;
}

export async function fetchWeatherAtKickoff(
  lat: number,
  lon: number,
  date: string,
  kickoffHourUtc: number
): Promise<WeatherData> {
  const url = `https://api.open-meteo.com/v1/forecast?latitude=${lat}&longitude=${lon}&hourly=windspeed_10m,precipitation,temperature_2m&windspeed_unit=mph&temperature_unit=fahrenheit&timezone=UTC&start_date=${date}&end_date=${date}`;
  try {
    const data = await fetchWithRetry<OpenMeteoResponse>(url);
    const times = data.hourly?.time ?? [];
    const winds = data.hourly?.windspeed_10m ?? [];
    const precips = data.hourly?.precipitation ?? [];
    const temps = data.hourly?.temperature_2m ?? [];

    const idx = Math.min(kickoffHourUtc, times.length - 1);
    return {
      windSpeedMph: winds[idx] ?? 5,
      precipInches: precips[idx] ?? 0,
      tempF: temps[idx] ?? 65,
    };
  } catch {
    return { windSpeedMph: 5, precipInches: 0, tempF: 65 };
  }
}

// Compute weather adjustments (0 = no impact, positive = harsh conditions hurt scoring)
export function computeWeatherAdj(weather: WeatherData, indoor: boolean): { wind: number; precip: number; temp: number } {
  if (indoor) return { wind: 0, precip: 0, temp: 0 };

  // Wind: > 15 mph starts hurting passing; > 25 mph significant
  const wind = Math.max(0, (weather.windSpeedMph - 10) * 0.04);

  // Precipitation: rain/snow hurts ball control
  const precip = Math.min(weather.precipInches * 2.0, 1.5);

  // Temperature: < 32°F hurts scoring, > 90°F hurts performance too
  const temp = weather.tempF < 32
    ? (32 - weather.tempF) * 0.04
    : weather.tempF > 90 ? (weather.tempF - 90) * 0.02 : 0;

  return { wind, precip, temp };
}

export { defaultTeamStats };
