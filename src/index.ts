// NFL Oracle v4.0 — CLI Entry Point
// Usage:
//   npm start                             → predictions for current NFL week
//   npm start -- --week 5                → specific week (current season)
//   npm start -- --alert picks           → send weekly picks to Discord
//   npm start -- --alert recap           → send last week's recap to Discord
//   npm start -- --help                  → show help

import 'dotenv/config';
import { logger } from './logger.js';
import { runPipeline } from './pipeline.js';
import { closeDb, initDb, getPredictionsByWeek } from './db/database.js';
import { getCurrentWeekInfo, isNFLSeason, getCurrentNFLSeason } from './api/nflClient.js';
import type { PipelineOptions } from './types.js';

type AlertMode = 'picks' | 'recap' | null;

function parseArgs(): PipelineOptions & { help: boolean; alertMode: AlertMode } {
  const args = process.argv.slice(2);
  const opts: PipelineOptions & { help: boolean; alertMode: AlertMode } = {
    help: false,
    verbose: true,
    forceRefresh: false,
    alertMode: null,
  };

  for (let i = 0; i < args.length; i++) {
    const arg = args[i];
    switch (arg) {
      case '--help': case '-h': opts.help = true; break;
      case '--week': case '-w': opts.week = Number(args[++i]); break;
      case '--season': case '-s': opts.season = Number(args[++i]); break;
      case '--force-refresh': case '-f': opts.forceRefresh = true; break;
      case '--quiet': case '-q': opts.verbose = false; break;
      case '--alert': case '-a': {
        const mode = args[++i];
        if (mode === 'picks' || mode === 'recap') opts.alertMode = mode;
        else { console.error(`Unknown alert mode: "${mode}". Use "picks" or "recap".`); process.exit(1); }
        break;
      }
    }
  }
  return opts;
}

function printHelp(): void {
  console.log(`
NFL Oracle v4.0 — ML Prediction Engine
=======================================

USAGE:
  npm start [options]

OPTIONS:
  --week, -w N            Run predictions for specific week (default: current)
  --season, -s YYYY       NFL season year (default: current)
  --force-refresh, -f     Bypass cache and re-fetch all data
  --quiet, -q             Suppress table output
  --alert, -a picks|recap Send Discord alert
  --help, -h              Show this help

EXAMPLES:
  npm start                              # Current week's picks
  npm start -- --week 3                 # Week 3 predictions
  npm run alerts:picks                   # Send weekly picks to Discord
  npm run alerts:recap                   # Send last week's recap to Discord
  npm run train                          # Train the ML model (Python)
  npm run build-dataset                  # Build training CSV (Python)

ENVIRONMENT (.env):
  DISCORD_WEBHOOK_URL    Discord webhook URL (required for alerts)
  ODDS_API_KEY           The Odds API key (optional — better Vegas lines)
  LOG_LEVEL              Logging level (default: info)

ARCHITECTURE:
  ESPN API → Feature Engineering (26 features) → Monte Carlo (10k sims)
  → ML Meta-model (Logistic Regression) → Isotonic Calibration
  → Edge Detection → SQLite → Discord
`);
}

// ─── Week label (e.g. "2026 Week 5" or "2026 Wild Card") ─────────────────────

function weekLabel(week: number, season: number): string {
  if (week <= 18) return `${season} Week ${week}`;
  const postseason: Record<number, string> = { 19: 'Wild Card', 20: 'Divisional', 21: 'Conference Championship', 22: 'Super Bowl' };
  return `${season} ${postseason[week] ?? `Postseason Week ${week}`}`;
}

// ─── Date range for a given NFL week (approximate) ───────────────────────────

function getWeekDateRange(week: number, season: number): { startDate: string; endDate: string } {
  // NFL regular season starts first Thursday of September
  // Approximate: Week 1 = early September, each subsequent week +7 days
  // For simplicity, use a reference date of Sept 4 as Week 1 start (close enough)
  const week1Start = new Date(`${season}-09-04`);
  const weekOffset = (week - 1) * 7;
  const start = new Date(week1Start);
  start.setDate(start.getDate() + weekOffset - 2); // -2 for Thursday start
  const end = new Date(start);
  end.setDate(end.getDate() + 6);
  return {
    startDate: start.toISOString().split('T')[0],
    endDate: end.toISOString().split('T')[0],
  };
}

// ─── Alert handlers ───────────────────────────────────────────────────────────

async function runPicksAlert(week: number, season: number): Promise<void> {
  const { sendWeeklyPicks } = await import('./alerts/discord.js');
  await initDb();

  let predictions = getPredictionsByWeek(week, season);

  if (predictions.length === 0) {
    logger.info({ week, season }, 'No predictions in DB — running pipeline first');
    predictions = await runPipeline({ week, season, verbose: false });
  }

  if (predictions.length === 0) {
    logger.warn({ week, season }, 'No games found — nothing to send');
    return;
  }

  const label = weekLabel(week, season);
  await sendWeeklyPicks(week, season, predictions, label);
}

async function runRecapAlert(week: number, season: number): Promise<void> {
  await initDb();  // Was missing — caused "Database not initialized" crash
  const { sendWeeklyRecap } = await import('./alerts/discord.js');
  const { processWeekResults } = await import('./alerts/results.js');

  // Recap the PREVIOUS week's results
  const recapWeek = week > 1 ? week - 1 : 1;
  const { startDate, endDate } = getWeekDateRange(recapWeek, season);
  const label = weekLabel(recapWeek, season);

  const { games, metrics } = await processWeekResults(recapWeek, season, startDate, endDate);
  await sendWeeklyRecap(recapWeek, season, label, games, metrics, metrics.seasonTotals);

  // After Super Bowl (week 22), send season summary
  if (week > 22) {
    const { sendSeasonSummary } = await import('./alerts/discord.js');
    const allPreds = getPredictionsByWeek(0, season); // rough — would need aggregate query
    logger.info('Season complete — sending summary');
    await sendSeasonSummary(season, 0, 0, 0, 0); // DB aggregate would fill these
  }
}

// ─── Entry point ──────────────────────────────────────────────────────────────

async function main(): Promise<void> {
  const opts = parseArgs();

  if (opts.help) { printHelp(); process.exit(0); }

  if (!isNFLSeason() && !opts.week) {
    logger.info('NFL is currently off-season. Use --week to force a specific week.');
    // Still allow running with explicit week (for testing)
    if (!opts.alertMode) { process.exit(0); }
  }

  const currentInfo = await getCurrentWeekInfo();
  const week = opts.week ?? currentInfo.week;
  const season = opts.season ?? currentInfo.season ?? getCurrentNFLSeason();

  // Guard against null/0/NaN week — would silently post empty Discord alerts
  if (!week || !Number.isFinite(week) || week < 1 || week > 22) {
    logger.warn({ week, currentInfo }, 'Invalid week — skipping alert (off-season or API failure)');
    process.exit(0);
  }

  logger.info({ week, season, alert: opts.alertMode ?? 'pipeline' }, 'NFL Oracle starting');

  try {
    if (opts.alertMode === 'picks') {
      await runPicksAlert(week, season);
      return;
    }

    if (opts.alertMode === 'recap') {
      await runRecapAlert(week, season);
      return;
    }

    // Clear cache if force refresh
    if (opts.forceRefresh) {
      const { readdirSync, unlinkSync } = await import('fs');
      const cacheDir = process.env.CACHE_DIR ?? './cache';
      try {
        for (const file of readdirSync(cacheDir)) {
          if (file.endsWith('.json')) unlinkSync(`${cacheDir}/${file}`);
        }
        logger.info('Cache cleared');
      } catch { /* may not exist */ }
    }

    const predictions = await runPipeline({ week, season, verbose: opts.verbose });

    if (predictions.length === 0) {
      console.log(`\nNo upcoming games for Week ${week}, ${season}.\n`);
    } else {
      logger.info({ predictions: predictions.length }, 'Done');
    }

  } catch (err) {
    logger.error({ err }, 'Fatal error');
    process.exit(1);
  } finally {
    closeDb();
  }
}

process.on('unhandledRejection', reason => { logger.error({ reason }, 'Unhandled rejection'); process.exit(1); });
process.on('uncaughtException', err => { logger.error({ err }, 'Uncaught exception'); closeDb(); process.exit(1); });

main();
