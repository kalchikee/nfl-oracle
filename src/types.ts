// NFL Oracle v4.0 — Core Type Definitions

// ─── NFL team / game types ────────────────────────────────────────────────────

export interface NFLTeamStats {
  teamId: string;
  teamAbbr: string;
  teamName: string;
  // Record
  wins: number;
  losses: number;
  ties: number;
  winPct: number;
  // Scoring (per game)
  pointsPerGame: number;
  pointsAllowedPerGame: number;
  // Offense
  offPassYdsPerGame: number;
  offRushYdsPerGame: number;
  offTotalYdsPerGame: number;
  offTurnoversPG: number;        // giveaways per game
  thirdDownConvPct: number;      // 0–1
  redZoneTdPct: number;          // 0–1
  passerRating: number;          // QB passer rating (0–158.3)
  // Defense
  defPassYdsPerGame: number;
  defRushYdsPerGame: number;
  defTotalYdsPerGame: number;
  defTurnoversPG: number;        // takeaways per game
  sacksPerGame: number;
  // Derived
  pythagoreanWinPct: number;
  turnoverDiff: number;          // defTurnoversPG - offTurnoversPG (positive = good)
  yardsPerPlay: number;          // off yards / off plays estimate
  defYardsPerPlay: number;
}

export interface NFLGame {
  gameId: string;
  gameDate: string;    // YYYY-MM-DD
  gameTime: string;    // ISO datetime UTC
  week: number;
  season: number;
  seasonType: number;  // 1=preseason, 2=regular, 3=postseason
  status: string;      // 'STATUS_SCHEDULED' | 'STATUS_IN_PROGRESS' | 'STATUS_FINAL'
  homeTeam: NFLGameTeam;
  awayTeam: NFLGameTeam;
  venueName: string;
  venueCity: string;
  venueIndoor: boolean;
  neutralSite: boolean;
  // Embedded odds (if available from ESPN)
  vegasSpread?: number;        // positive = home favored
  vegasTotal?: number;
  homeMoneyLine?: number;
  awayMoneyLine?: number;
}

export interface NFLGameTeam {
  teamId: string;
  teamAbbr: string;
  teamName: string;
  score?: number;
}

export interface InjuredPlayer {
  playerId: string;
  playerName: string;
  position: string;
  status: string;    // 'Out' | 'Doubtful' | 'Questionable' | 'Probable'
  description: string;
}

export interface InjuryReport {
  teamAbbr: string;
  players: InjuredPlayer[];
}

// ─── Feature vector (all features are home - away diffs) ─────────────────────

export interface FeatureVector {
  // Team strength
  elo_diff: number;              // home Elo - away Elo
  pythagorean_diff: number;      // home pythagorean win% - away
  win_pct_diff: number;          // home win% - away win%

  // Scoring efficiency
  ppg_diff: number;              // home points/game - away points/game (offense)
  papg_diff: number;             // home points-allowed/game - away (negative = home better def)
  net_ppg_diff: number;          // (ppg - papg) home - away

  // Yardage efficiency
  yds_per_play_diff: number;     // home off YPP - away off YPP
  def_yds_per_play_diff: number; // home def YPP - away def YPP

  // Turnover
  turnover_diff_diff: number;    // home turnover_diff - away turnover_diff

  // Passing
  passer_rating_diff: number;    // home passer rating - away

  // Situational
  third_down_diff: number;       // home 3rd-down% - away
  red_zone_diff: number;         // home RZ TD% - away

  // Injuries
  injury_impact_diff: number;    // injury-adjusted efficiency loss (home - away)
  qb_out_home: number;           // 1 if home QB is out/doubtful
  qb_out_away: number;           // 1 if away QB is out/doubtful

  // Rest / schedule
  rest_days_diff: number;        // home rest days - away rest days
  short_week_home: number;       // 1 if home on short rest (<= 5 days since last game)
  short_week_away: number;       // 1 if away on short rest

  // Venue / weather
  wind_adj: number;              // wind penalty (positive = harsh wind, hurts both teams)
  precip_adj: number;            // precipitation penalty
  temp_adj: number;              // cold temperature penalty
  is_dome: number;               // 1 if indoor stadium (nullifies weather)
  altitude_adj: number;          // altitude disadvantage for away team (Denver ~0.3)

  // Game context
  divisional_game: number;       // 1 if same division matchup
  is_neutral: number;            // 1 if neutral site (Super Bowl / London)

  // Vegas and model inputs
  vegas_home_prob: number;       // vig-removed Vegas implied probability (0 if unavailable)
  mc_win_pct: number;            // Monte Carlo win probability (set at prediction time)
}

// ─── Model outputs ────────────────────────────────────────────────────────────

export interface ExpectedScoreEstimate {
  homeExpPts: number;
  awayExpPts: number;
  homeStd: number;
  awayStd: number;
}

export interface MonteCarloResult {
  win_probability: number;        // home win probability
  away_win_probability: number;
  spread: number;                 // home expected spread (positive = home favored)
  total_points: number;           // expected total
  most_likely_score: [number, number]; // [home, away]
  upset_probability: number;
  blowout_probability: number;    // margin >= 14
  home_exp_pts: number;
  away_exp_pts: number;
  simulations: number;
}

export interface Prediction {
  game_date: string;
  game_id: string;
  week: number;
  season: number;
  home_team: string;
  away_team: string;
  venue: string;
  feature_vector: FeatureVector;
  mc_win_pct: number;
  calibrated_prob: number;
  vegas_prob?: number;
  edge?: number;
  model_version: string;
  home_exp_pts: number;
  away_exp_pts: number;
  total_points: number;
  spread: number;
  most_likely_score: string;
  upset_probability: number;
  blowout_probability: number;
  actual_winner?: string;
  correct?: boolean;
  created_at: string;
}

export interface EloRating {
  teamAbbr: string;
  rating: number;
  updatedAt: string;
}

export interface AccuracyLog {
  week: number;
  season: number;
  brier_score: number;
  log_loss: number;
  accuracy: number;
  high_conv_accuracy: number;
  games_evaluated: number;
}

export interface GameResult {
  game_id: string;
  date: string;
  week: number;
  season: number;
  home_team: string;
  away_team: string;
  home_score: number;
  away_score: number;
  venue: string;
}

export interface PipelineOptions {
  week?: number;
  season?: number;
  forceRefresh?: boolean;
  verbose?: boolean;
}

// ─── Edge detection ───────────────────────────────────────────────────────────

export type EdgeCategory = 'none' | 'small' | 'meaningful' | 'large' | 'extreme';

export interface EdgeResult {
  modelProb: number;
  vegasProb: number;
  edge: number;
  edgeCategory: EdgeCategory;
  homeFavorite: boolean;
}

// ─── Season helpers ───────────────────────────────────────────────────────────

export interface NFLWeekInfo {
  season: number;
  week: number;
  seasonType: number; // 2=regular, 3=postseason
  startDate: string;  // YYYY-MM-DD
  endDate: string;
  label: string;      // e.g. "Week 1" or "Wild Card"
}
