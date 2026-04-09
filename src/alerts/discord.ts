// NFL Oracle v4.0 — Discord Webhook Alerts
// Message 1: Weekly picks briefing (Tuesday — games for the upcoming week)
// Message 2: Weekly recap (Tuesday — results from the prior week)
// Concise, embed-based format.

import fetch from 'node-fetch';
import { logger } from '../logger.js';
import { getConfidenceTier } from '../features/marketEdge.js';
import type { Prediction } from '../types.js';
import type { SeasonTotals } from '../db/database.js';

// ─── Colors ───────────────────────────────────────────────────────────────────

const COLORS = {
  picks: 0x003d7a,       // deep blue
  edge: 0x27ae60,        // green — strong edge picks
  recap_good: 0x2ecc71,  // green
  recap_bad: 0xe74c3c,   // red
  recap_neutral: 0x95a5a6,
} as const;

// ─── Discord types ────────────────────────────────────────────────────────────

interface DiscordField { name: string; value: string; inline?: boolean; }
interface DiscordEmbed {
  title?: string; description?: string; color?: number;
  fields?: DiscordField[]; footer?: { text: string }; timestamp?: string;
}
interface DiscordPayload { content?: string; embeds: DiscordEmbed[]; }

// ─── Webhook sender ───────────────────────────────────────────────────────────

async function sendWebhook(payload: DiscordPayload): Promise<boolean> {
  const webhookUrl = process.env.DISCORD_WEBHOOK_URL;
  if (!webhookUrl) {
    logger.warn('DISCORD_WEBHOOK_URL not set — skipping Discord alert');
    return false;
  }
  try {
    const resp = await fetch(webhookUrl, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload),
      signal: AbortSignal.timeout(10000),
    });
    if (!resp.ok) {
      const text = await resp.text();
      logger.error({ status: resp.status, body: text }, 'Discord webhook error');
      return false;
    }
    logger.info('Discord alert sent');
    return true;
  } catch (err) {
    logger.error({ err }, 'Failed to send Discord webhook');
    return false;
  }
}

// ─── Helpers ──────────────────────────────────────────────────────────────────

function pct(prob: number): string {
  return (prob * 100).toFixed(1) + '%';
}

function confidenceBar(prob: number): string {
  const p = Math.max(prob, 1 - prob);
  if (p >= 0.72) return '🔥🔥🔥';
  if (p >= 0.67) return '🔥🔥';
  if (p >= 0.62) return '🔥';
  if (p >= 0.57) return '✅';
  return '🪙';
}

function pickTeam(pred: Prediction): { team: string; winPct: number } {
  if (pred.calibrated_prob >= 0.5) return { team: pred.home_team, winPct: pred.calibrated_prob };
  return { team: pred.away_team, winPct: 1 - pred.calibrated_prob };
}

function isHighConviction(pred: Prediction): boolean {
  const tier = getConfidenceTier(pred.calibrated_prob);
  return tier === 'extreme' || tier === 'high_conviction';
}

function spreadLabel(pred: Prediction): string {
  const sp = pred.spread;
  if (Math.abs(sp) < 0.5) return 'PK (Pick\'em)';
  if (sp > 0) return `${pred.home_team} -${Math.abs(sp).toFixed(1)}`;
  return `${pred.away_team} -${Math.abs(sp).toFixed(1)}`;
}

// ─── MESSAGE 1: Weekly Picks Briefing ────────────────────────────────────────

export async function sendWeeklyPicks(
  week: number,
  season: number,
  predictions: Prediction[],
  seasonLabel: string,
): Promise<boolean> {
  if (predictions.length === 0) {
    logger.warn({ week, season }, 'No predictions — skipping picks alert');
    return false;
  }

  // Sort by confidence descending
  const sorted = [...predictions].sort(
    (a, b) => Math.abs(b.calibrated_prob - 0.5) - Math.abs(a.calibrated_prob - 0.5)
  );

  const highConv = sorted.filter(p => isHighConviction(p));
  const hasEdge = sorted.some(p => p.edge !== undefined && Math.abs(p.edge ?? 0) >= 0.06);

  // ── Embed 1: All Picks ────────────────────────────────────────────────────
  const picksFields: DiscordField[] = sorted.map(pred => {
    const { team, winPct } = pickTeam(pred);
    const conf = confidenceBar(pred.calibrated_prob);
    const edgeLine = pred.edge !== undefined && Math.abs(pred.edge) >= 0.03
      ? `  Edge: ${pred.edge >= 0 ? '+' : ''}${pct(pred.edge)}`
      : '';
    return {
      name: `${conf} ${pred.away_team} @ ${pred.home_team}  ·  ${pred.game_date ?? ''}`,
      value: [
        `**Pick:** ${team}  (${pct(winPct)})  |  **Spread:** ${spreadLabel(pred)}  |  **O/U:** ${pred.total_points.toFixed(0)}`,
        `Proj: ${pred.away_team} ${pred.most_likely_score.split('-')[1]} – ${pred.home_team} ${pred.most_likely_score.split('-')[0]}${edgeLine}`,
      ].join('\n'),
      inline: false,
    };
  });

  const picksEmbed: DiscordEmbed = {
    title: `🏈 NFL Oracle — ${seasonLabel} Picks`,
    description: `${predictions.length} games this week  ·  ${highConv.length} high-conviction pick${highConv.length !== 1 ? 's' : ''}${hasEdge ? '  ·  ⚡ edge games flagged' : ''}`,
    color: COLORS.picks,
    fields: picksFields.slice(0, 20), // Discord max 25 fields
    footer: { text: '🔥🔥🔥 Extreme  🔥🔥 High  🔥 Strong  ✅ Lean  🪙 Coin Flip  ·  NFL Oracle v4.0' },
    timestamp: new Date().toISOString(),
  };

  // ── Embed 2: High-Conviction Picks (only if any) ──────────────────────────
  const embeds: DiscordEmbed[] = [picksEmbed];

  if (highConv.length > 0) {
    const hcFields: DiscordField[] = highConv.map(pred => {
      const { team, winPct } = pickTeam(pred);
      const tier = getConfidenceTier(pred.calibrated_prob);
      const tierLabel = tier === 'extreme' ? '🔥🔥🔥 EXTREME' : '🔥🔥 HIGH CONVICTION';
      const reasons: string[] = [`Model win%: ${pct(winPct)}`];
      if (pred.edge !== undefined && Math.abs(pred.edge) >= 0.06) {
        reasons.push(`${pred.edge >= 0 ? '+' : ''}${pct(pred.edge)} vs Vegas`);
      }
      if (pred.feature_vector.qb_out_away) reasons.push('Away QB out/doubtful');
      if (pred.feature_vector.short_week_away) reasons.push('Away team on short week');
      if (pred.feature_vector.rest_days_diff >= 4) reasons.push(`Home has ${pred.feature_vector.rest_days_diff}+ more rest days`);

      return {
        name: `${tierLabel}: ${pred.away_team} @ ${pred.home_team}`,
        value: [
          `**Pick:** ${team}  |  **Spread:** ${spreadLabel(pred)}`,
          `**Why:** ${reasons.join(' · ')}`,
        ].join('\n'),
        inline: false,
      };
    });

    embeds.push({
      title: `⭐ High-Conviction Picks — ${highConv.length} bet${highConv.length !== 1 ? 's' : ''}`,
      description: 'Games where model confidence ≥ 67%. Historical accuracy on these: **68–72%**.',
      color: COLORS.edge,
      fields: hcFields,
      footer: { text: 'Bet responsibly. Model picks are for entertainment only.' },
    });
  }

  return sendWebhook({ embeds });
}

// ─── MESSAGE 2: Weekly Recap ──────────────────────────────────────────────────

export async function sendWeeklyRecap(
  week: number,
  season: number,
  seasonLabel: string,
  games: Array<{
    prediction: Prediction;
    homeScore: number;
    awayScore: number;
  }>,
  metrics: { accuracy: number; brier: number; highConvAccuracy: number | null },
  seasonTotals?: SeasonTotals,
): Promise<boolean> {
  if (games.length === 0) {
    return sendWebhook({
      embeds: [{
        title: `📊 NFL Oracle — ${seasonLabel} Recap`,
        description: 'No completed games found. Results may still be processing.',
        color: COLORS.recap_neutral,
        timestamp: new Date().toISOString(),
      }],
    });
  }

  const correct = games.filter(g => g.prediction.correct).length;
  const total = games.length;
  const accPct = (correct / total) * 100;
  const color = accPct >= 65 ? COLORS.recap_good : accPct >= 50 ? COLORS.recap_neutral : COLORS.recap_bad;
  const accEmoji = accPct >= 65 ? '🟢' : accPct >= 50 ? '🟡' : '🔴';

  const hcGames = games.filter(g => isHighConviction(g.prediction));
  const hcCorrect = hcGames.filter(g => g.prediction.correct).length;

  // Game-by-game lines (concise)
  const gameLines = games.map(({ prediction: pred, homeScore, awayScore }) => {
    const { team: picked } = pickTeam(pred);
    const ok = pred.correct ? '✅' : '❌';
    const bet = isHighConviction(pred) ? ' ⭐' : '';
    return `${ok}${bet} **${pred.away_team}** ${awayScore}–${homeScore} **${pred.home_team}** *(picked ${picked})*`;
  }).join('\n');

  // Season running totals
  let seasonLine = '';
  if (seasonTotals && seasonTotals.totalGames > 0) {
    const sAcc = (seasonTotals.accuracy * 100).toFixed(0);
    const sHcAcc = seasonTotals.hcGames > 0
      ? `  ·  ⭐ ${seasonTotals.hcCorrect}/${seasonTotals.hcGames} (${(seasonTotals.hcAccuracy * 100).toFixed(0)}%) high-conv`
      : '';
    seasonLine = `**📅 ${season} Season: ${seasonTotals.totalCorrect}/${seasonTotals.totalGames} (${sAcc}%)${sHcAcc}**`;
  }

  const summaryLines = [
    `**${accEmoji} This week: ${correct}/${total} correct  (${accPct.toFixed(0)}%)**`,
    hcGames.length > 0
      ? `**⭐ High-conviction: ${hcCorrect}/${hcGames.length}  (${((hcCorrect / hcGames.length) * 100).toFixed(0)}%)**`
      : '**⭐ No high-conviction picks this week**',
    seasonLine,
    `Brier score: ${metrics.brier.toFixed(4)} *(lower = better, 0.25 = coin flip)*`,
  ].filter(Boolean).join('\n');

  const embed: DiscordEmbed = {
    title: `📊 NFL Oracle — ${seasonLabel} Results`,
    color,
    fields: [
      { name: '📈 Summary', value: summaryLines, inline: false },
      { name: '🎯 Game-by-Game', value: gameLines.slice(0, 1000) || 'No results.', inline: false },
    ],
    footer: { text: '⭐ = high-conviction pick  ·  NFL Oracle v4.0' },
    timestamp: new Date().toISOString(),
  };

  return sendWebhook({ embeds: [embed] });
}

// ─── Season-end message ───────────────────────────────────────────────────────

export async function sendSeasonSummary(
  season: number,
  totalCorrect: number,
  totalGames: number,
  hcCorrect: number,
  hcGames: number,
): Promise<boolean> {
  const acc = totalGames > 0 ? (totalCorrect / totalGames * 100).toFixed(1) : '0';
  const hcAcc = hcGames > 0 ? (hcCorrect / hcGames * 100).toFixed(1) : 'N/A';

  return sendWebhook({
    embeds: [{
      title: `🏆 NFL Oracle — ${season} Season Complete`,
      description: [
        `**Overall:** ${totalCorrect}/${totalGames} (${acc}%)`,
        `**High-conviction:** ${hcCorrect}/${hcGames} (${hcAcc}%)`,
        '',
        'The Super Bowl is over. Oracle goes dormant until next September. 🏈',
      ].join('\n'),
      color: 0xf1c40f,
      timestamp: new Date().toISOString(),
      footer: { text: 'NFL Oracle v4.0 · See you in September' },
    }],
  });
}
