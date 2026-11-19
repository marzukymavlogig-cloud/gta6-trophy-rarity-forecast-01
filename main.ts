// gta6-trophy-rarity-forecast-01 — dilution-aware trophy-rarity forecaster.
//
// Forecast, for every GTA VI PS5 trophy, five quantiles of its Exophase earn
// rate at the case target. Node 22, standard library only, offline,
// deterministic, one case file in -> one forecast file out.
//
// What the data actually is (read this before touching the model):
//   * A read shows the rates as of Exophase's LAST refresh, not live play.
//     On launch night the page sat unchanged for hours (638 players tracked,
//     every rate identical) and then refreshed once: 3,868 players tracked,
//     trophies-earned 943 -> 4,276, every rate but the 100% one lower.
//   * That refresh is pure dilution arithmetic. The new cohort (3,230
//     players) all hold the one 100% trophy, so the other 54 trophies gained
//     only 103 earners between them: a new tracked player earns a given
//     trophy at roughly 7% of the launch cohort's rate. A naive rate-trend
//     model (persistence, or a line through flat history) misses this by a
//     mile: it is the whole practice game, and it matters at the final too,
//     where every refresh dilutes the rates the same way.
//   * So the model works in COUNTS, not rates: earners E_i(t) and players
//     tracked N(t) are fitted separately against log time-since-release,
//     and the rate is E/N. A refresh that adds players who have barely
//     played moves N a lot and E a little, which is exactly what the
//     E/N split captures, while a rate-only fit smears the two together.
//
// Two regimes, chosen from the case itself (never from the case id):
//   1. FLAT history (every read identical, checkpoint within a day of
//      release, short horizon): no trend exists to fit. The page is stale
//      and a refresh is overdue, so the forecast assumes the launch-night
//      refresh pattern: N multiplies ~6x, new-cohort earn fraction ~0.067
//      of the launch rate, trophies at 100% stay at 100%.
//   2. TREND history (distinct reads, the final's two days of them): fit
//      log E_i and log N against u = ln(1 + hours since release) by
//      recency-weighted least squares, shrink each trophy's slope toward
//      its tier's pooled slope and the global slope, extrapolate E and N
//      to the target, and cross-check with a second estimator (logit rate
//      against u, anchored on the last read). The median is the mean of
//      the two estimators in logit space; the spread adds fit uncertainty,
//      estimator disagreement, count noise and refresh-lag uncertainty.
//
// Spread is honest, not decorative: the pinball loss on the logit scale
// punishes a too-narrow band on every trophy outside it and a too-wide
// band on every trophy inside it, so sigma is built from the actual
// sources of error and floored/capped, never tuned to a single cut.
//
// Input:  every *.in in $MAKEFUN_INPUT_DIR (else /input), one case each.
// Output: <id>.out in $MAKEFUN_OUTPUT_DIR (else /output):
//   {"trophies":[{"id":n,"q":[q05,q25,q50,q75,q95]}, ...]}

import { mkdirSync, readdirSync, readFileSync, writeFileSync } from 'node:fs';
import { join } from 'node:path';

const INPUT_DIR = process.env.MAKEFUN_INPUT_DIR ?? '/input';
const OUTPUT_DIR = process.env.MAKEFUN_OUTPUT_DIR ?? '/output';

/** The quantile levels the contest scores, in the order each row lists them. */
const LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95];

interface Trophy {
  id: number;
  name: string;
  tier: string;
}
interface Read {
  at: string;
  players_tracked: number;
  trophies_earned: number;
  rates: Array<number | null>;
}
interface Case {
  id: string;
  at: string;
  horizon_minutes: number;
  target: string;
  levels: number[];
  game: { release_date: string };
  trophies: Trophy[];
  history: Read[];
}

const clamp = (x: number, lo: number, hi: number): number =>
  Math.min(hi, Math.max(lo, x));
/** logit of a percent, clamped to [0.01, 99.99] exactly as the scorer clamps. */
const logit = (pct: number): number => {
  const p = clamp(pct, 0.01, 99.99) / 100;
  return Math.log(p / (1 - p));
};
const invLogit = (l: number): number => 100 / (1 + Math.exp(-l));
const round4 = (x: number): number => Math.round(x * 10_000) / 10_000;
const median = (xs: number[]): number => {
  const s = [...xs].sort((a, b) => a - b);
  const m = s.length >> 1;
  return s.length % 2 ? s[m] : (s[m - 1] + s[m]) / 2;
};

/**
 * Standard-normal quantile (Acklam's rational approximation, |err| < 1.2e-9),
 * so any level list works, not just the contest's five.
 */
function normQuantile(p: number): number {
  const a = [-3.969683028665376e+01, 2.209460984245205e+02, -2.759285984469534e+02,
    1.383577518672690e+02, -3.066479806614716e+01, 2.506628277459239e+00];
  const b = [-5.447609879822406e+01, 1.615858368580409e+02, -1.556989798598866e+02,
    6.680131188771972e+01, -1.328068155288572e+01];
  const c = [-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e+00,
    -2.549732539343734e+00, 4.374664141464968e+00, 2.938163982698783e+00];
  const d = [7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e+00,
    3.754408661907416e+00];
  const plow = 0.02425;
  const phigh = 1 - plow;
  let q: number;
  let r: number;
  if (p < plow) {
    q = Math.sqrt(-2 * Math.log(p));
    return (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) /
      ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1);
  }
  if (p > phigh) {
    q = Math.sqrt(-2 * Math.log(1 - p));
    return -(((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) /
      ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1);
  }
  q = p - 0.5;
  r = q * q;
  return (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * q /
    (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1);
}

/** Recency-weighted least squares of y on x; returns slope, intercept, slope SE. */
function wls(xs: number[], ys: number[], ws: number[]): { b: number; a: number; seB: number } {
  const n = xs.length;
  const sw = ws.reduce((s, w) => s + w, 0);
  const mx = xs.reduce((s, x, i) => s + ws[i] * x, 0) / sw;
  const my = ys.reduce((s, y, i) => s + ws[i] * y, 0) / sw;
  let sxx = 0;
  let sxy = 0;
  for (let i = 0; i < n; i++) {
    sxx += ws[i] * (xs[i] - mx) * (xs[i] - mx);
    sxy += ws[i] * (xs[i] - mx) * (ys[i] - my);
  }
  if (sxx <= 1e-12) return { b: 0, a: my, seB: 0.5 };
  const b = sxy / sxx;
  const a = my - b * mx;
  let rss = 0;
  for (let i = 0; i < n; i++) {
    const e = ys[i] - (a + b * xs[i]);
    rss += ws[i] * e * e;
  }
  const dof = Math.max(1, n - 2);
  const s2 = rss / dof / (sw / n); // unweighted-scale residual variance
  return { b, a, seB: Math.sqrt(Math.max(s2, 1e-6) / sxx) };
}

/** Hours from the release-day midnight to an ISO timestamp (may be < 0 pre-release). */
function hoursSinceRelease(releaseDate: string, iso: string): number {
  const t0 = Date.parse(`${releaseDate}T00:00:00Z`);
  const base = Number.isNaN(t0) ? Date.parse('2026-11-19T00:00:00Z') : t0;
  return (Date.parse(iso) - base) / 3_600_000;
}
/** Log-time coordinate: ln(1 + hours since release), floored at 0. */
const uOf = (hours: number): number => Math.log(1 + Math.max(0, hours));

/** One row of quantiles from a logit median and a logit sigma. */
function quantiles(medianLogit: number, sigma: number, levels: number[]): number[] {
  const qs = levels.map((tau) => {
    const v = round4(clamp(invLogit(medianLogit + normQuantile(tau) * sigma), 0.01, 100));
    return v;
  });
  // Rounding can only tie neighbours, never invert them; enforce it anyway.
  for (let i = 1; i < qs.length; i++) if (qs[i] < qs[i - 1]) qs[i] = qs[i - 1];
  return qs;
}

// ---------------------------------------------------------------------------
// The forecast
// ---------------------------------------------------------------------------

/**
 * Launch-night refresh, as the kit's own record states it: first probe
 * 638 players tracked / 943 trophies earned, first refresh 3,868 / 4,276,
 * every rate but the 100% one lower. The new cohort (3,230 players) all
 * hold the 100% trophy, so the other trophies gained 103 earners between
 * them: a new tracked player earns a given trophy at about 6.7% of the
 * launch cohort's rate. Hence, across one refresh, a non-100% rate is
 * multiplied by (638 + 0.067 * 3230) / 3868 = 0.221.
 */
const FIRST_REFRESH_N = 3868;
const FIRST_REFRESH_N0 = 638;
const NEW_COHORT_EARN_FRAC = 0.067;
const DILUTION_FACTOR = (FIRST_REFRESH_N0 + NEW_COHORT_EARN_FRAC * (FIRST_REFRESH_N - FIRST_REFRESH_N0)) / FIRST_REFRESH_N;

/** Fallback log-time slopes for earners when a trophy has no fittable trend. */
const PRIOR_EARNER_SLOPE: Record<string, number> = {
  platinum: 0.25, gold: 0.5, silver: 0.7, bronze: 0.9,
};

interface State {
  at: string;
  u: number;
  n: number;
  rates: Array<number | null>;
}

/** Collapse history to its distinct states (a state changes only at a refresh). */
function distinctStates(c: Case): State[] {
  const out: State[] = [];
  let prevKey = '';
  for (const h of c.history) {
    const key = `${h.players_tracked}|${h.rates.map((r) => (r === null ? 'x' : r)).join(',')}`;
    if (key === prevKey) continue;
    prevKey = key;
    out.push({
      at: h.at,
      u: uOf(hoursSinceRelease(c.game.release_date, h.at)),
      n: h.players_tracked,
      rates: h.rates,
    });
  }
  return out;
}

function forecastCase(c: Case): { trophies: Array<{ id: number; q: number[] }> } {
  const levels = c.levels.length === LEVELS.length ? c.levels : LEVELS;
  const last = c.history[c.history.length - 1];
  const uC = uOf(hoursSinceRelease(c.game.release_date, c.at));
  const uT = uOf(hoursSinceRelease(c.game.release_date, c.target));
  const du = Math.max(0, uT - uC);
  const horizonDays = c.horizon_minutes / 1440;
  const states = distinctStates(c);
  const checkpointAgeH = hoursSinceRelease(c.game.release_date, c.at);

  // -- Regime 1: flat history -------------------------------------------
  // No read ever changed, checkpoint still in the first day, short horizon:
  // there is no trend to fit, the page is stale, and a refresh is overdue.
  // Expect the launch-night refresh pattern (see DILUTION_FACTOR above).
  const flat = states.length <= 1 && checkpointAgeH < 24 && c.horizon_minutes <= 720;
  if (flat) {
    const w = clamp(c.horizon_minutes / 90, 0, 1); // a 90-minute horizon spans the refresh
    const factor = 1 + w * (DILUTION_FACTOR - 1);
    return {
      trophies: c.trophies.map((t, i) => {
        const r = last.rates[i];
        if (r !== null && r >= 99.99) {
          // Everyone tracked holds it, including every newcomer: it stays 100%.
          return { id: t.id, q: quantiles(logit(100), 0.03, levels) };
        }
        const base = r === null || r <= 0 ? 0.01 : r;
        return { id: t.id, q: quantiles(logit(base * factor), 0.55, levels) };
      }),
    };
  }

  // -- Regime 2: trend history ------------------------------------------
  // Recency weights: the most recent distinct state counts double.
  const m = states.length;
  const ws = states.map((_, j) => 1 + (m <= 1 ? 0 : j / (m - 1)));
  const us = states.map((s) => s.u);

  // Players tracked: log N against u. Trophy earn curves saturate — the
  // slope a trophy shows lately is a better guide to the target than its
  // whole-history slope — so fit both the full window and the recent half
  // (in u) and lean on the recent one, for N and for every trophy below.
  const recentFit = (xs: number[], ys: number[], ws: number[], uMid: number) => {
    const full = wls(xs, ys, ws);
    const idx = xs.map((x, i) => i).filter((i) => xs[i] >= uMid);
    if (idx.length < 3) return full;
    const rec = wls(idx.map((i) => xs[i]), idx.map((i) => ys[i]), idx.map((i) => ws[i]));
    return { b: 0.65 * rec.b + 0.35 * full.b, a: full.a, seB: Math.max(full.seB, rec.seB) };
  };
  const uMid = m >= 2 ? (us[0] + us[m - 1]) / 2 : us[0];
  let gN = 0.5;
  let seGN = 0.3;
  if (m >= 3) {
    const fitN = recentFit(us, states.map((s) => Math.log(Math.max(1, s.n))), ws, uMid);
    gN = clamp(fitN.b, 0, 1.2);
    seGN = fitN.seB;
  }
  const nLast = Math.max(1, last.players_tracked);
  const nGrowth = clamp(Math.exp(gN * du), 1, 6);
  const nTarget = nLast * nGrowth;

  // Per-trophy earners: log(E + 0.5) against u, and logit(rate) against u.
  interface TrophyFit { bE: number; seE: number; bR: number; seR: number; nPts: number; ok: boolean }
  const fits: TrophyFit[] = c.trophies.map((t, i) => {
    const xs: number[] = [];
    const yE: number[] = [];
    const yR: number[] = [];
    const w: number[] = [];
    states.forEach((s, j) => {
      const r = s.rates[i];
      if (r === null) return;
      xs.push(s.u);
      yE.push(Math.log((r / 100) * s.n + 0.5));
      yR.push(logit(r));
      w.push(ws[j]);
    });
    // Distinct earn-count values: a fit needs at least 3 distinct points.
    const distinctE = new Set(yE.map((y) => y.toFixed(3))).size;
    if (xs.length < 3 || distinctE < 3) {
      return { bE: PRIOR_EARNER_SLOPE[t.tier] ?? 0.8, seE: 0.5, bR: 0, seR: 0.5, nPts: xs.length, ok: false };
    }
    const fE = recentFit(xs, yE, w, (xs[0] + xs[xs.length - 1]) / 2);
    const fR = recentFit(xs, yR, w, (xs[0] + xs[xs.length - 1]) / 2);
    return { bE: clamp(fE.b, 0, 3), seE: fE.seB, bR: clamp(fR.b, -1, 2), seR: fR.seB, nPts: xs.length, ok: true };
  });

  // Tier-pooled and global slopes (medians over fittable trophies only).
  const poolSlope = (key: 'bE' | 'bR'): number => {
    const vals = fits.filter((f) => f.ok).map((f) => f[key]);
    return vals.length ? median(vals) : 0.6;
  };
  const globalE = poolSlope('bE');
  const globalR = poolSlope('bR');
  const tierSlope = (tier: string, key: 'bE' | 'bR'): number => {
    const vals = fits.filter((f, i) => f.ok && c.trophies[i].tier === tier).map((f) => f[key]);
    return vals.length >= 2 ? median(vals) : (key === 'bE' ? globalE : globalR);
  };

  const rows = c.trophies.map((t, i) => {
    const rLast = last.rates[i];
    if (rLast !== null && rLast >= 99.99) {
      return { id: t.id, q: quantiles(logit(100), 0.05, levels) };
    }
    const f = fits[i];
    // Shrink the trophy's slope toward its tier pool and the global pool.
    // Weight grows with the number of distinct points behind the fit.
    const wData = clamp(f.nPts - 1, 0, 10);
    const bE = (wData * f.bE + 3 * tierSlope(t.tier, 'bE') + 2 * globalE) / (wData + 5);
    const bR = (wData * f.bR + 3 * tierSlope(t.tier, 'bR') + 2 * globalR) / (wData + 5);

    // Estimator A (counts): E and N separately, rate = E / N.
    const eLast = rLast === null || rLast <= 0 ? 0.5 : (rLast / 100) * nLast;
    const eTarget = Math.min(0.98 * nTarget, Math.max(eLast, eLast * Math.exp(bE * du)));
    const rateA = clamp((100 * eTarget) / nTarget, 0.01, 99.99);
    // Estimator B (rate curve): logit rate against u, anchored on the last read.
    const baseRate = rLast === null || rLast <= 0 ? 0.01 : rLast;
    const rateB = clamp(invLogit(logit(baseRate) + bR * du), 0.01, 99.99);
    const medLogitRaw = (logit(rateA) + logit(rateB)) / 2;
    // Saturation damping: a trophy already at a high rate cannot keep
    // climbing on its log-time line — its ceiling sits below 100% and
    // growth in logit space slows as it approaches it. Damp the
    // extrapolated logit move by a factor that falls with the last rate
    // (no damping for rare trophies, ~half at 55%+). Rare trophies, where
    // the growth is, are barely touched.
    const damp = 1 - 0.45 * Math.sqrt(clamp(baseRate, 0, 100) / 100);
    const medLogit = logit(baseRate) + (medLogitRaw - logit(baseRate)) * damp;

    // Spread: random-walk base + slope uncertainty + estimator
    // disagreement + count noise + refresh-lag uncertainty.
    const baseRW = 0.35 * Math.sqrt(Math.max(horizonDays, 0.05));
    const slopeUnc = Math.max(f.seE, f.seR) * du;
    const disagree = Math.abs(logit(rateA) - logit(rateB)) * 0.5;
    const countNoise = clamp(1 / Math.sqrt(Math.max(eTarget, 1)), 0, 0.5);
    const refreshLag = 0.2;
    const sigma = clamp(
      0.85 * Math.sqrt(baseRW * baseRW + slopeUnc * slopeUnc + disagree * disagree +
        countNoise * countNoise + refreshLag * refreshLag),
      0.4, 1.3,
    );
    return { id: t.id, q: quantiles(medLogit, sigma, levels) };
  });
  return { trophies: rows };
}

// ---------------------------------------------------------------------------
// Entry point
// ---------------------------------------------------------------------------

mkdirSync(OUTPUT_DIR, { recursive: true });
const inputs = readdirSync(INPUT_DIR).filter((f) => f.endsWith('.in')).sort();
for (const file of inputs) {
  const id = file.slice(0, -'.in'.length);
  const c = JSON.parse(readFileSync(join(INPUT_DIR, file), 'utf8')) as Case;
  if (!c.history.length) throw new Error(`${file} has no history`);
  const out = forecastCase(c);
  writeFileSync(join(OUTPUT_DIR, `${id}.out`), `${JSON.stringify(out)}\n`);
}
