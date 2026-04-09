// NFL Oracle v4.0 — Result Processing
// Fetches completed games, matches to stored predictions, computes accuracy.

import { logger } from '../logger.js';
import { fetchCompletedResults } from '../api/nflClient.js';
import {
  getPredictionByGameId, updatePredictionResult, upsertGameResult, upsertAccuracyLog,
  getPredictionsByWeek, getSeasonTotals,
} from '../db/database.js';
import type { SeasonTotals } from '../db/database.js';
import { updateElo } from '../features/eloEngine.js';
import type { Prediction } from '../types.js';

export interface GameWithResult {
  prediction: Prediction;
  homeScore: number;
  awayScore: number;
}

export interface RecapMetrics {
  accuracy: number;
  brier: number;
  highConvAccuracy: number | null;
  seasonTotals?: SeasonTotals;
}

// ─── Process a week's results ─────────────────────────────────────────────────

export async function processWeekResults(
  week: number,
  season: number,
  startDate: string,
  endDate: string,
): Promise<{ games: GameWithResult[]; metrics: RecapMetrics }> {
  // 1. Fetch completed games from ESPN
  const results = await fetchCompletedResults(startDate, endDate);
  logger.info({ count: results.length, week, season }, 'Completed results fetched');

  const games: GameWithResult[] = [];

  for (const result of results) {
    // Store the result
    upsertGameResult(result);

    // Look up stored prediction
    const pred = getPredictionByGameId(result.game_id);
    if (!pred) {
      logger.debug({ gameId: result.game_id }, 'No prediction found for game');
      continue;
    }

    // Determine winner and correctness
    const homeWon = result.home_score > result.away_score;
    const actualWinner = homeWon ? pred.home_team : pred.away_team;
    const pickedHome = pred.calibrated_prob >= 0.5;
    const correct = pickedHome === homeWon;

    updatePredictionResult(result.game_id, actualWinner, correct);

    // Update Elo after game
    try {
      updateElo(result.home_team, result.away_team, result.home_score, result.away_score);
    } catch (err) {
      logger.debug({ err }, 'Elo update failed');
    }

    games.push({
      prediction: { ...pred, actual_winner: actualWinner, correct },
      homeScore: result.home_score,
      awayScore: result.away_score,
    });
  }

  // 2. Also check predictions stored from this week that we may have already processed
  if (games.length === 0) {
    const weekPreds = getPredictionsByWeek(week, season).filter(p => p.correct !== undefined);
    if (weekPreds.length > 0) {
      logger.info({ count: weekPreds.length }, 'Using already-processed predictions from DB');
      // Can't reconstruct scores, but we can compute metrics
      const metrics = computeMetrics(weekPreds, week, season, []);
      return { games: [], metrics };
    }
  }

  const metrics = computeMetrics(games.map(g => g.prediction), week, season, games);
  metrics.seasonTotals = getSeasonTotals(season);

  return { games, metrics };
}

// ─── Compute accuracy metrics ─────────────────────────────────────────────────

function computeMetrics(
  predictions: Prediction[],
  week: number,
  season: number,
  _games: GameWithResult[],
): RecapMetrics {
  const evaluated = predictions.filter(p => p.correct !== undefined);
  if (evaluated.length === 0) {
    return { accuracy: 0, brier: 0.25, highConvAccuracy: null };
  }

  const correct = evaluated.filter(p => p.correct).length;
  const accuracy = correct / evaluated.length;

  // Brier score: mean((prob - outcome)^2)
  let brierSum = 0;
  for (const pred of evaluated) {
    const outcome = pred.correct ? 1 : 0;
    const prob = Math.max(pred.calibrated_prob, 1 - pred.calibrated_prob);
    brierSum += Math.pow(prob - outcome, 2);
  }
  const brier = brierSum / evaluated.length;

  // High-conviction accuracy (>= 67% confidence)
  const hc = evaluated.filter(p => Math.max(p.calibrated_prob, 1 - p.calibrated_prob) >= 0.67);
  const highConvAccuracy = hc.length > 0 ? hc.filter(p => p.correct).length / hc.length : null;

  // Log to DB
  const logLoss = evaluated.reduce((s, p) => {
    const prob = p.correct ? p.calibrated_prob : 1 - p.calibrated_prob;
    return s + -Math.log(Math.max(0.001, prob));
  }, 0) / evaluated.length;

  upsertAccuracyLog({
    week,
    season,
    brier_score: brier,
    log_loss: logLoss,
    accuracy,
    high_conv_accuracy: highConvAccuracy ?? 0,
    games_evaluated: evaluated.length,
  });

  return { accuracy, brier, highConvAccuracy };
}
