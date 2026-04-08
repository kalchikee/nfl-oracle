// NFL Oracle v4.0 — Weekly Pipeline
// Orchestrates: Fetch → Features → Monte Carlo → ML Model → Edge → Store → Print

import { logger } from './logger.js';
import { fetchWeekSchedule, fetchInjuries, fetchAllTeamStats, getCurrentWeekInfo } from './api/nflClient.js';
import { computeFeatures } from './features/featureEngine.js';
import { runMonteCarlo } from './models/monteCarlo.js';
import { loadModel, predict as mlPredict, isModelLoaded, getModelInfo } from './models/metaModel.js';
import { upsertPrediction, initDb, getPredictionsByWeek } from './db/database.js';
import { getOddsForGame, loadOddsApiLines } from './api/oddsClient.js';
import { seedElos } from './features/eloEngine.js';
import { computeEdge, getConfidenceTier } from './features/marketEdge.js';
import type { NFLGame, Prediction, PipelineOptions } from './types.js';

const MODEL_VERSION = '4.0.0';

// ─── Last-game-date lookup (from schedule — used for rest days) ───────────────
// We track the previous week's game date per team by fetching the prior week's schedule.
// Simple approach: assume 7 days rest unless it's a Thursday night game (4 days).

async function getLastGameDates(
  week: number,
  season: number,
  homeAbbr: string,
  awayAbbr: string,
  gameDate: string,
): Promise<{ homeLastGame: string | null; awayLastGame: string | null }> {
  if (week <= 1) {
    return { homeLastGame: null, awayLastGame: null };
  }
  try {
    const prevWeekGames = await fetchWeekSchedule(week - 1, season);
    const homeGame = prevWeekGames.find(g =>
      g.homeTeam.teamAbbr === homeAbbr || g.awayTeam.teamAbbr === homeAbbr
    );
    const awayGame = prevWeekGames.find(g =>
      g.homeTeam.teamAbbr === awayAbbr || g.awayTeam.teamAbbr === awayAbbr
    );
    return {
      homeLastGame: homeGame?.gameDate ?? null,
      awayLastGame: awayGame?.gameDate ?? null,
    };
  } catch {
    return { homeLastGame: null, awayLastGame: null };
  }
}

// ─── Main pipeline ────────────────────────────────────────────────────────────

export async function runPipeline(options: PipelineOptions = {}): Promise<Prediction[]> {
  // Determine week / season
  let { week, season } = options;
  if (!week || !season) {
    const current = await getCurrentWeekInfo();
    week = options.week ?? current.week;
    season = options.season ?? current.season;
  }

  logger.info({ week, season, version: MODEL_VERSION }, '=== NFL Oracle v4.0 Pipeline Start ===');

  // 1. Init DB
  await initDb();

  // 2. Seed Elo ratings (idempotent)
  seedElos();

  // 3. Load ML meta-model (if available)
  const modelLoaded = loadModel();
  if (modelLoaded) {
    const info = getModelInfo();
    logger.info({ version: info?.version, brier: info?.avg_brier, seasons: info?.train_seasons }, 'ML model active');
  } else {
    logger.info('ML model not found — using Monte Carlo only. Run: npm run train');
  }

  // 4. Load Vegas odds (Odds API, if key present)
  await loadOddsApiLines();

  // 5. Fetch team stats and injuries (shared across all games this week)
  const [teamStats, injuries] = await Promise.all([
    fetchAllTeamStats(season),
    fetchInjuries(),
  ]);

  // 6. Fetch this week's schedule
  const games = await fetchWeekSchedule(week, season);
  if (games.length === 0) {
    logger.warn({ week, season }, 'No games found for this week');
    return [];
  }

  logger.info({ week, season, games: games.length }, 'Schedule fetched');

  // 7. Process each game
  const predictions: Prediction[] = [];

  for (const game of games) {
    // Skip completed games
    if (game.status.includes('FINAL') || game.status.includes('final')) {
      logger.debug({ gameId: game.gameId, status: game.status }, 'Skipping completed game');
      continue;
    }

    try {
      const pred = await processGame(game, week, season, teamStats, injuries, modelLoaded);
      if (pred) predictions.push(pred);
    } catch (err) {
      logger.error({ err, gameId: game.gameId, home: game.homeTeam.teamAbbr, away: game.awayTeam.teamAbbr }, 'Failed to process game');
    }
  }

  logger.info({ processed: predictions.length, week, season }, 'Pipeline complete');

  if (options.verbose !== false) {
    printPredictions(predictions, week, season, modelLoaded);
  }

  return predictions;
}

// ─── Single game processing ───────────────────────────────────────────────────

async function processGame(
  game: NFLGame,
  week: number,
  season: number,
  teamStats: Awaited<ReturnType<typeof fetchAllTeamStats>>,
  injuries: Awaited<ReturnType<typeof fetchInjuries>>,
  modelLoaded: boolean,
): Promise<Prediction | null> {
  const homeAbbr = game.homeTeam.teamAbbr;
  const awayAbbr = game.awayTeam.teamAbbr;

  logger.info({ gameId: game.gameId, matchup: `${awayAbbr} @ ${homeAbbr}` }, 'Processing game');

  // ── Rest days ────────────────────────────────────────────────────────────
  const { homeLastGame, awayLastGame } = await getLastGameDates(week, season, homeAbbr, awayAbbr, game.gameDate);

  // ── Feature vector ───────────────────────────────────────────────────────
  const features = await computeFeatures(game, teamStats, injuries, homeLastGame, awayLastGame);

  // ── Monte Carlo ──────────────────────────────────────────────────────────
  const mc = runMonteCarlo(features);
  features.mc_win_pct = mc.win_probability;

  // ── Vegas odds ───────────────────────────────────────────────────────────
  let vegas_prob: number | undefined;
  let edge: number | undefined;

  const gameOdds = getOddsForGame(
    homeAbbr, awayAbbr,
    game.vegasSpread, game.vegasTotal, game.homeMoneyLine, game.awayMoneyLine
  );

  if (gameOdds) {
    vegas_prob = gameOdds.homeImpliedProb;
    features.vegas_home_prob = vegas_prob;
  }

  // ── ML calibration ───────────────────────────────────────────────────────
  let calibrated_prob: number;
  if (modelLoaded && isModelLoaded()) {
    calibrated_prob = mlPredict(features, mc.win_probability);
  } else {
    calibrated_prob = mc.win_probability;
  }

  if (vegas_prob !== undefined) {
    const homeML = game.homeMoneyLine ?? -110;
    const awayML = game.awayMoneyLine ?? -110;
    const edgeResult = computeEdge(calibrated_prob, homeML, awayML);
    edge = edgeResult.edge;
    logger.debug({ matchup: `${awayAbbr} @ ${homeAbbr}` }, edgeResult.edgeCategory);
  }

  // ── Build prediction ─────────────────────────────────────────────────────
  const prediction: Prediction = {
    game_date: game.gameDate,
    game_id: game.gameId,
    week,
    season,
    home_team: homeAbbr,
    away_team: awayAbbr,
    venue: game.venueName,
    feature_vector: features,
    mc_win_pct: mc.win_probability,
    calibrated_prob,
    vegas_prob,
    edge,
    model_version: MODEL_VERSION,
    home_exp_pts: mc.home_exp_pts,
    away_exp_pts: mc.away_exp_pts,
    total_points: mc.total_points,
    spread: mc.spread,
    most_likely_score: `${mc.most_likely_score[0]}-${mc.most_likely_score[1]}`,
    upset_probability: mc.upset_probability,
    blowout_probability: mc.blowout_probability,
    created_at: new Date().toISOString(),
  };

  upsertPrediction(prediction);
  return prediction;
}

// ─── Console output ───────────────────────────────────────────────────────────

function printPredictions(
  predictions: Prediction[],
  week: number,
  season: number,
  mlActive = false,
): void {
  if (predictions.length === 0) {
    console.log(`\nNo predictions for Week ${week}, ${season}\n`);
    return;
  }

  const label = mlActive ? 'ML+Isotonic' : 'Monte Carlo';
  const width = 105;

  console.log('\n' + '═'.repeat(width));
  console.log(`  NFL ORACLE v4.0  ·  Week ${week}, ${season}  ·  ${predictions.length} games  ·  [${label}]`);
  console.log('═'.repeat(width));

  const sorted = [...predictions].sort(
    (a, b) => Math.abs(b.calibrated_prob - 0.5) - Math.abs(a.calibrated_prob - 0.5)
  );

  console.log('\n' + [
    pad('MATCHUP', 22), pad('WIN%', 8), pad('MC%', 8), pad('SPREAD', 12),
    pad('TOTAL', 7), pad('PROJ', 10), pad('EDGE', 8), 'TIER',
  ].join('  '));
  console.log('─'.repeat(width));

  for (const p of sorted) {
    const matchup = `${p.away_team} @ ${p.home_team}`;
    const calPct = (p.calibrated_prob * 100).toFixed(1) + '%';
    const mcPct = (p.mc_win_pct * 100).toFixed(1) + '%';
    const sp = p.spread >= 0
      ? `${p.home_team} -${Math.abs(p.spread).toFixed(1)}`
      : `${p.away_team} -${Math.abs(p.spread).toFixed(1)}`;
    const edgeStr = p.edge !== undefined
      ? (p.edge >= 0 ? '+' : '') + (p.edge * 100).toFixed(1) + '%'
      : '—';
    const tier = getConfidenceTier(p.calibrated_prob).replace('_', ' ').toUpperCase();
    const marker = Math.abs(p.calibrated_prob - 0.5) >= 0.17 ? ' ★' : '';

    console.log([
      pad(matchup, 22), pad(calPct, 8), pad(mcPct, 8), pad(sp, 12),
      pad(p.total_points.toFixed(0), 7), pad(p.most_likely_score, 10),
      pad(edgeStr, 8), tier + marker,
    ].join('  '));
  }

  console.log('─'.repeat(width));
  console.log('★ = high conviction (≥67%)  |  EDGE = model vs vig-removed Vegas\n');
}

function pad(s: string, w: number): string {
  return s.length >= w ? s.slice(0, w) : s + ' '.repeat(w - s.length);
}
