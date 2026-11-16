// Model-agnostic word attributions: leave-one-out occlusion, LIME and KernelSHAP.
// Features are the words of a sentence; a coalition z ∈ {0,1}^M keeps the words with z_j = 1 (removed words are
// deleted from the text). The explained quantity is the model's log-odds of POSITIVE, f(z).
import { rebuild, rng as makeRng } from './text.js';

/** Wraps a batch scorer with word masking, caching and query counting. */
export function objective(parts, scoreTexts, { signal, onProgress } = {}) {
  const W = []; parts.forEach((p, i) => { if (p.word) W.push(i); });
  const pos = new Map(W.map((i, j) => [i, j]));
  const cache = new Map();
  let queries = 0;
  const text = (z) => rebuild(parts, (i) => z[pos.get(i)] === 1);
  async function f(masks) {
    const keys = masks.map((m) => m.join(''));
    const todo = new Map();
    keys.forEach((k, i) => { if (!cache.has(k) && !todo.has(k)) todo.set(k, masks[i]); });
    if (todo.size) {
      const ks = [...todo.keys()];
      const s = await scoreTexts(ks.map((k) => text(todo.get(k))), { signal, onProgress });
      ks.forEach((k, i) => cache.set(k, s[i]));
      queries += ks.length;
    }
    return keys.map((k) => cache.get(k));
  }
  return { M: W.length, W, f, text, get queries() { return queries; }, get cached() { return cache.size; } };
}

const ones = (M) => new Uint8Array(M).fill(1);

export async function occlusion(obj) {
  const { M, f } = obj;
  const masks = [ones(M)];
  for (let j = 0; j < M; j++) { const z = ones(M); z[j] = 0; masks.push(z); }
  const s = await f(masks);
  return Array.from({ length: M }, (_, j) => s[0] - s[j + 1]);
}

/** Weighted least squares with optional ridge (no penalty on column `free`). Solves (XᵀWX + λD)β = XᵀWy. */
function wls(X, y, w, lambda = 0, free = -1) {
  const n = X.length, d = X[0].length;
  const A = Array.from({ length: d }, () => new Float64Array(d)), b = new Float64Array(d);
  for (let i = 0; i < n; i++) {
    const xi = X[i], wi = w[i];
    for (let a = 0; a < d; a++) {
      if (!xi[a]) continue;
      const v = wi * xi[a];
      b[a] += v * y[i];
      for (let c = 0; c < d; c++) if (xi[c]) A[a][c] += v * xi[c];
    }
  }
  for (let a = 0; a < d; a++) A[a][a] += a === free ? 1e-10 : lambda + 1e-10;
  // Gaussian elimination with partial pivoting
  for (let c = 0; c < d; c++) {
    let p = c; for (let r = c + 1; r < d; r++) if (Math.abs(A[r][c]) > Math.abs(A[p][c])) p = r;
    [A[c], A[p]] = [A[p], A[c]]; [b[c], b[p]] = [b[p], b[c]];
    const piv = A[c][c] || 1e-12;
    for (let r = c + 1; r < d; r++) { const k = A[r][c] / piv; if (!k) continue; for (let q = c; q < d; q++) A[r][q] -= k * A[c][q]; b[r] -= k * b[c]; }
  }
  const x = new Float64Array(d);
  for (let r = d - 1; r >= 0; r--) { let s = b[r]; for (let q = r + 1; q < d; q++) s -= A[r][q] * x[q]; x[r] = s / (A[r][r] || 1e-12); }
  return x;
}

/** LIME for text (Ribeiro et al. 2016): random word removals, exponential cosine kernel, ridge regression. */
export async function lime(obj, { samples = 256, seed = 1, width = 25, alpha = 1 } = {}) {
  const { M, f } = obj; const rnd = makeRng(seed);
  if (M === 1) { const [a, b] = await f([ones(1), new Uint8Array(1)]); return [a - b]; }
  const masks = [ones(M)];
  for (let s = 1; s < samples; s++) {
    const k = 1 + Math.floor(rnd() * (M - 1)); // number of words to remove, 1..M-1 (as in the reference implementation)
    const z = ones(M), idx = [...Array(M).keys()];
    for (let i = 0; i < k; i++) { const j = i + Math.floor(rnd() * (M - i)); [idx[i], idx[j]] = [idx[j], idx[i]]; z[idx[i]] = 0; }
    masks.push(z);
  }
  const y = await f(masks);
  const X = [], w = [];
  for (const z of masks) {
    const kept = z.reduce((a, b) => a + b, 0);
    const d = (1 - Math.sqrt(kept / M)) * 100; // cosine distance to the all-ones vector, ×100
    w.push(Math.sqrt(Math.exp(-(d * d) / (width * width))));
    X.push([...z, 1]); // last column = intercept (not penalized)
  }
  const beta = wls(X, y, w, alpha, M);
  return Array.from(beta.slice(0, M));
}

function binom(n, k) { let r = 1; for (let i = 1; i <= k; i++) r = (r * (n - k + i)) / i; return r; }

/** KernelSHAP (Lundberg & Lee 2017) with the efficiency constraint Σφ = f(x) − f(∅) enforced exactly.
 *  Enumerates all coalitions (exact Shapley values) when 2^M − 2 ≤ budget, otherwise paired sampling from the Shapley kernel. */
export async function kernelShap(obj, { samples = 256, seed = 1 } = {}) {
  const { M, f } = obj;
  const [vFull, v0] = await f([ones(M), new Uint8Array(M)]);
  if (M === 1) return { phi: [vFull - v0], exact: true };
  const masks = [], w = [];
  let exact = false;
  if (M <= 20 && 2 ** M - 2 <= samples) {
    exact = true;
    for (let c = 1; c < 2 ** M - 1; c++) {
      const z = new Uint8Array(M); let k = 0;
      for (let j = 0; j < M; j++) if (c & (1 << j)) { z[j] = 1; k++; }
      masks.push(z); w.push((M - 1) / (binom(M, k) * k * (M - k)));
    }
  } else {
    const rnd = makeRng(seed);
    const pk = []; let tot = 0;
    for (let k = 1; k < M; k++) { const v = (M - 1) / (k * (M - k)); pk.push(v); tot += v; }
    for (let s = 0; s + 1 < samples; s += 2) {
      let u = rnd() * tot, k = 1; for (; k < M - 1 && u > pk[k - 1]; k++) u -= pk[k - 1];
      const z = new Uint8Array(M), idx = [...Array(M).keys()];
      for (let i = 0; i < k; i++) { const j = i + Math.floor(rnd() * (M - i)); [idx[i], idx[j]] = [idx[j], idx[i]]; z[idx[i]] = 1; }
      masks.push(z, z.map((v) => 1 - v)); w.push(1, 1); // paired (antithetic) sampling
    }
  }
  const y = await f(masks);
  const D = vFull - v0;
  const X = masks.map((z) => Array.from({ length: M - 1 }, (_, j) => z[j] - z[M - 1]));
  const yy = masks.map((z, i) => y[i] - v0 - z[M - 1] * D);
  const phi = Array.from(wls(X, yy, w, 0));
  phi.push(D - phi.reduce((a, b) => a + b, 0));
  return { phi, exact, vFull, v0 };
}

/** Deletion (most-important-first) and insertion curves for the predicted class; AOPC per Samek et al. (2017). */
export async function faithfulness(obj, order, toProb) {
  const { M, f } = obj;
  const del = [], ins = [];
  for (let k = 0; k <= M; k++) {
    const zd = ones(M), zi = new Uint8Array(M);
    for (let t = 0; t < k; t++) { zd[order[t]] = 0; zi[order[t]] = 1; }
    del.push(zd); ins.push(zi);
  }
  const sd = (await f(del)).map(toProb), si = (await f(ins)).map(toProb);
  const aopc = sd.slice(1).reduce((a, p) => a + (sd[0] - p), 0) / Math.max(1, M);
  const auc = (c) => c.reduce((a, p, i) => (i ? a + (p + c[i - 1]) / 2 : a), 0) / Math.max(1, M);
  return { del: sd, ins: si, aopc, delAUC: auc(sd), insAUC: auc(si) };
}

/** Spearman rank correlation (average ranks for ties). */
export function spearman(a, b) {
  const rank = (v) => { const o = v.map((x, i) => [x, i]).sort((p, q) => p[0] - q[0]); const r = new Array(v.length); for (let i = 0; i < o.length;) { let j = i; while (j + 1 < o.length && o[j + 1][0] === o[i][0]) j++; for (let k = i; k <= j; k++) r[o[k][1]] = (i + j) / 2; i = j + 1; } return r; };
  const ra = rank(a), rb = rank(b), n = a.length;
  if (n < 2) return 1;
  const ma = ra.reduce((x, y) => x + y, 0) / n, mb = rb.reduce((x, y) => x + y, 0) / n;
  let s = 0, sa = 0, sb = 0; for (let i = 0; i < n; i++) { s += (ra[i] - ma) * (rb[i] - mb); sa += (ra[i] - ma) ** 2; sb += (rb[i] - mb) ** 2; }
  return sa && sb ? s / Math.sqrt(sa * sb) : (sa === sb ? 1 : 0);
}

/** Overlap of the top-k features by |attribution|. */
export function topkOverlap(a, b, k) {
  const top = (v) => new Set(v.map((x, i) => [Math.abs(x), i]).sort((p, q) => q[0] - p[0]).slice(0, k).map((p) => p[1]));
  const A = top(a), B = top(b); let n = 0; for (const i of A) if (B.has(i)) n++;
  return n / Math.max(1, Math.min(k, a.length));
}
