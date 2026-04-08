// NFL Oracle v4.0 — SQLite Database Layer (sql.js — pure JS, no native build)

import initSqlJs, { type Database as SqlJsDatabase } from 'sql.js';
import { readFileSync, writeFileSync, existsSync, mkdirSync } from 'fs';
import { resolve, dirname } from 'path';
import { fileURLToPath } from 'url';
import type { Prediction, GameResult, AccuracyLog, EloRating } from '../types.js';

const __filename = fileURLToPath(import.meta.url);
const __dirname = dirname(__filename);

const DB_PATH = resolve(
  process.env.DB_PATH
    ? process.env.DB_PATH.startsWith('.')
      ? resolve(__dirname, '../../', process.env.DB_PATH)
      : process.env.DB_PATH
    : resolve(__dirname, '../../data/nfl_oracle.db')
);

mkdirSync(dirname(DB_PATH), { recursive: true });

let _db: SqlJsDatabase | null = null;
let _SQL: Awaited<ReturnType<typeof initSqlJs>> | null = null;

// ─── Initialization ───────────────────────────────────────────────────────────

export async function initDb(): Promise<SqlJsDatabase> {
  if (_db) return _db;
  _SQL = await initSqlJs();
  if (existsSync(DB_PATH)) {
    const fileBuffer = readFileSync(DB_PATH);
    _db = new _SQL.Database(fileBuffer);
  } else {
    _db = new _SQL.Database();
  }
  initializeSchema(_db);
  persistDb();
  return _db;
}

export function getDb(): SqlJsDatabase {
  if (!_db) throw new Error('Database not initialized. Call initDb() first.');
  return _db;
}

export function persistDb(): void {
  if (!_db) return;
  const data = _db.export();
  writeFileSync(DB_PATH, Buffer.from(data));
}

function run(sql: string, params: (string | number | null | undefined)[] = []): void {
  const db = getDb();
  const stmt = db.prepare(sql);
  stmt.run(params.map(p => (p === undefined ? null : p)));
  stmt.free();
  persistDb();
}

function queryAll<T = Record<string, unknown>>(sql: string, params: (string | number | null)[] = []): T[] {
  const db = getDb();
  const stmt = db.prepare(sql);
  stmt.bind(params);
  const results: T[] = [];
  while (stmt.step()) {
    results.push(stmt.getAsObject() as T);
  }
  stmt.free();
  return results;
}

function queryOne<T = Record<string, unknown>>(sql: string, params: (string | number | null)[] = []): T | undefined {
  return queryAll<T>(sql, params)[0];
}

// ─── Schema ───────────────────────────────────────────────────────────────────

function initializeSchema(db: SqlJsDatabase): void {
  db.run(`
    CREATE TABLE IF NOT EXISTS elo_ratings (
      team_abbr TEXT PRIMARY KEY,
      rating REAL NOT NULL DEFAULT 1500,
      games_played INTEGER NOT NULL DEFAULT 0,
      season INTEGER NOT NULL DEFAULT 2025,
      updated_at TEXT NOT NULL DEFAULT (datetime('now'))
    );

    CREATE TABLE IF NOT EXISTS predictions (
      id INTEGER PRIMARY KEY AUTOINCREMENT,
      game_id TEXT NOT NULL,
      game_date TEXT NOT NULL,
      week INTEGER NOT NULL DEFAULT 0,
      season INTEGER NOT NULL DEFAULT 2026,
      home_team TEXT NOT NULL,
      away_team TEXT NOT NULL,
      venue TEXT NOT NULL DEFAULT '',
      feature_vector TEXT NOT NULL,
      home_exp_pts REAL NOT NULL DEFAULT 0,
      away_exp_pts REAL NOT NULL DEFAULT 0,
      mc_win_pct REAL NOT NULL,
      calibrated_prob REAL NOT NULL,
      vegas_prob REAL,
      edge REAL,
      total_points REAL NOT NULL DEFAULT 0,
      spread REAL NOT NULL DEFAULT 0,
      most_likely_score TEXT NOT NULL DEFAULT '',
      upset_probability REAL NOT NULL DEFAULT 0,
      blowout_probability REAL NOT NULL DEFAULT 0,
      model_version TEXT NOT NULL DEFAULT '4.0.0',
      actual_winner TEXT,
      correct INTEGER,
      created_at TEXT NOT NULL DEFAULT (datetime('now'))
    );

    CREATE TABLE IF NOT EXISTS accuracy_log (
      id INTEGER PRIMARY KEY AUTOINCREMENT,
      week INTEGER NOT NULL,
      season INTEGER NOT NULL,
      brier_score REAL NOT NULL DEFAULT 0,
      log_loss REAL NOT NULL DEFAULT 0,
      accuracy REAL NOT NULL DEFAULT 0,
      high_conv_accuracy REAL NOT NULL DEFAULT 0,
      games_evaluated INTEGER NOT NULL DEFAULT 0,
      created_at TEXT NOT NULL DEFAULT (datetime('now')),
      UNIQUE(week, season)
    );

    CREATE TABLE IF NOT EXISTS game_results (
      game_id TEXT PRIMARY KEY,
      date TEXT NOT NULL,
      week INTEGER NOT NULL DEFAULT 0,
      season INTEGER NOT NULL DEFAULT 2026,
      home_team TEXT NOT NULL,
      away_team TEXT NOT NULL,
      home_score INTEGER NOT NULL,
      away_score INTEGER NOT NULL,
      venue TEXT NOT NULL DEFAULT '',
      created_at TEXT NOT NULL DEFAULT (datetime('now'))
    );

    CREATE TABLE IF NOT EXISTS model_registry (
      id INTEGER PRIMARY KEY AUTOINCREMENT,
      version TEXT NOT NULL UNIQUE,
      train_seasons TEXT NOT NULL DEFAULT '',
      test_brier REAL NOT NULL DEFAULT 0,
      test_accuracy REAL NOT NULL DEFAULT 0,
      created_at TEXT NOT NULL DEFAULT (datetime('now'))
    );
  `);

  const cnt = queryOne<{ cnt: number }>('SELECT COUNT(*) as cnt FROM model_registry');
  if (!cnt || cnt.cnt === 0) {
    db.run(
      `INSERT OR IGNORE INTO model_registry (version, train_seasons, test_brier, test_accuracy)
       VALUES (?, ?, ?, ?)`,
      ['4.0.0', '2021-2025', 0, 0]
    );
  }
}

// ─── Elo helpers ──────────────────────────────────────────────────────────────

export function upsertElo(rating: EloRating): void {
  run(
    `INSERT INTO elo_ratings (team_abbr, rating, updated_at)
     VALUES (?, ?, ?)
     ON CONFLICT(team_abbr) DO UPDATE SET
       rating = excluded.rating,
       updated_at = excluded.updated_at`,
    [rating.teamAbbr, rating.rating, rating.updatedAt]
  );
}

export function getElo(teamAbbr: string): number {
  const row = queryOne<{ rating: number }>(
    'SELECT rating FROM elo_ratings WHERE team_abbr = ?',
    [teamAbbr]
  );
  return row?.rating ?? 1500;
}

export function getAllElos(): EloRating[] {
  return queryAll<{ team_abbr: string; rating: number; updated_at: string }>(
    'SELECT team_abbr, rating, updated_at FROM elo_ratings ORDER BY rating DESC'
  ).map(r => ({ teamAbbr: r.team_abbr, rating: r.rating, updatedAt: r.updated_at }));
}

// ─── Prediction helpers ───────────────────────────────────────────────────────

export function upsertPrediction(pred: Prediction): void {
  run(`DELETE FROM predictions WHERE game_id = ? AND model_version = ?`, [pred.game_id, pred.model_version]);
  run(
    `INSERT INTO predictions (
       game_id, game_date, week, season, home_team, away_team, venue,
       feature_vector, home_exp_pts, away_exp_pts,
       mc_win_pct, calibrated_prob, vegas_prob, edge,
       total_points, spread, most_likely_score,
       upset_probability, blowout_probability,
       model_version, created_at
     ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)`,
    [
      pred.game_id, pred.game_date, pred.week, pred.season,
      pred.home_team, pred.away_team, pred.venue,
      JSON.stringify(pred.feature_vector),
      pred.home_exp_pts, pred.away_exp_pts,
      pred.mc_win_pct, pred.calibrated_prob,
      pred.vegas_prob ?? null, pred.edge ?? null,
      pred.total_points, pred.spread, pred.most_likely_score,
      pred.upset_probability, pred.blowout_probability,
      pred.model_version, pred.created_at,
    ]
  );
}

export function getPredictionsByWeek(week: number, season: number): Prediction[] {
  const rows = queryAll<Record<string, unknown>>(
    'SELECT * FROM predictions WHERE week = ? AND season = ? ORDER BY calibrated_prob DESC',
    [week, season]
  );
  return rows.map(row => ({
    ...row,
    feature_vector: JSON.parse(row.feature_vector as string),
  })) as Prediction[];
}

export function getPredictionByGameId(gameId: string): Prediction | undefined {
  const row = queryOne<Record<string, unknown>>(
    'SELECT * FROM predictions WHERE game_id = ? ORDER BY created_at DESC LIMIT 1',
    [gameId]
  );
  if (!row) return undefined;
  return { ...row, feature_vector: JSON.parse(row.feature_vector as string) } as Prediction;
}

export function updatePredictionResult(gameId: string, winner: string, correct: boolean): void {
  run(
    `UPDATE predictions SET actual_winner = ?, correct = ? WHERE game_id = ?`,
    [winner, correct ? 1 : 0, gameId]
  );
}

// ─── Game result helpers ───────────────────────────────────────────────────────

export function upsertGameResult(result: GameResult): void {
  run(
    `INSERT INTO game_results (game_id, date, week, season, home_team, away_team, home_score, away_score, venue)
     VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
     ON CONFLICT(game_id) DO UPDATE SET
       home_score = excluded.home_score,
       away_score = excluded.away_score`,
    [
      result.game_id, result.date, result.week, result.season,
      result.home_team, result.away_team,
      result.home_score, result.away_score, result.venue,
    ]
  );
}

// ─── Accuracy helpers ─────────────────────────────────────────────────────────

export function upsertAccuracyLog(log: AccuracyLog): void {
  run(
    `INSERT INTO accuracy_log (week, season, brier_score, log_loss, accuracy, high_conv_accuracy, games_evaluated)
     VALUES (?, ?, ?, ?, ?, ?, ?)
     ON CONFLICT(week, season) DO UPDATE SET
       brier_score = excluded.brier_score,
       accuracy = excluded.accuracy,
       high_conv_accuracy = excluded.high_conv_accuracy,
       games_evaluated = excluded.games_evaluated`,
    [log.week, log.season, log.brier_score, log.log_loss, log.accuracy, log.high_conv_accuracy, log.games_evaluated]
  );
}

export function getRecentAccuracy(lastN = 5): AccuracyLog[] {
  return queryAll<AccuracyLog>(
    'SELECT * FROM accuracy_log ORDER BY season DESC, week DESC LIMIT ?',
    [lastN]
  );
}

export function closeDb(): void {
  if (_db) {
    persistDb();
    _db.close();
    _db = null;
  }
}
