// NFL Oracle v4.0 — Market Edge Detection
// Compares calibrated model probability to Vegas implied probability.
// NFL lines are extremely sharp — edges ≥ 6% are high-value.

import type { EdgeResult, EdgeCategory } from '../types.js';

export function computeEdge(modelProb: number, homeML: number, awayML: number): EdgeResult {
  const rawHome = mlToImplied(homeML);
  const rawAway = mlToImplied(awayML);
  const total = rawHome + rawAway;
  const vegasProb = rawHome / total; // vig-removed home probability

  const edge = modelProb - vegasProb;
  const absEdge = Math.abs(edge);

  let edgeCategory: EdgeCategory;
  if (absEdge < 0.03) edgeCategory = 'none';
  else if (absEdge < 0.06) edgeCategory = 'small';
  else if (absEdge < 0.10) edgeCategory = 'meaningful';
  else if (absEdge < 0.15) edgeCategory = 'large';
  else edgeCategory = 'extreme';

  return {
    modelProb,
    vegasProb,
    edge,
    edgeCategory,
    homeFavorite: modelProb >= 0.5,
  };
}

function mlToImplied(ml: number): number {
  if (ml < 0) return Math.abs(ml) / (Math.abs(ml) + 100);
  return 100 / (ml + 100);
}

export function getConfidenceTier(calibratedProb: number): 'coin_flip' | 'lean' | 'strong' | 'high_conviction' | 'extreme' {
  const p = Math.max(calibratedProb, 1 - calibratedProb);
  if (p >= 0.72) return 'extreme';
  if (p >= 0.67) return 'high_conviction';
  if (p >= 0.62) return 'strong';
  if (p >= 0.57) return 'lean';
  return 'coin_flip';
}

export function formatEdge(result: EdgeResult): string {
  const sign = result.edge >= 0 ? '+' : '';
  return `Model: ${(result.modelProb * 100).toFixed(1)}% | Vegas: ${(result.vegasProb * 100).toFixed(1)}% | Edge: ${sign}${(result.edge * 100).toFixed(1)}% [${result.edgeCategory.toUpperCase()}]`;
}

// ─── Signal agreement ─────────────────────────────────────────────────────────
// Counts how many model signals agree with the pick direction.
// NFL signals: Elo, Pythagorean expectation, net PPG, rest advantage, passer rating.
// More agreeing signals = pick is backed by multiple independent factors.

export type SignalAgreement = {
  agreeing: number;
  total: number;
  label: 'CONTRARIAN' | 'SPLIT' | 'MAJORITY' | 'CONSENSUS' | 'LOCK';
};

export function getSignalAgreement(
  features: Record<string, number>,
  pickIsHome: boolean,
): SignalAgreement {
  const dir = pickIsHome ? 1 : -1;
  const candidates: Array<number | undefined> = [
    features['elo_diff'],
    features['pythagorean_diff'],
    features['net_ppg_diff'],
    features['rest_days_diff'],
    features['passer_rating_diff'],
  ];
  const valid = candidates.filter((v): v is number => v != null);
  const agreeing = valid.filter(v => v * dir > 0).length;
  const total = valid.length;

  let label: SignalAgreement['label'];
  if (agreeing === total)  label = 'LOCK';
  else if (agreeing >= 4)  label = 'CONSENSUS';
  else if (agreeing >= 3)  label = 'MAJORITY';
  else if (agreeing >= 2)  label = 'SPLIT';
  else                     label = 'CONTRARIAN';

  return { agreeing, total, label };
}
