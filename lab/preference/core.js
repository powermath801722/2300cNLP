// Toy preference-learning lab: data, Bradley–Terry reward model, KL-regularized optimal policy,
// exact policy-gradient RLHF and DPO. Pure functions; used by worker.js and the Node check script.

export function mulberry32(seed) {
  let a = seed >>> 0;
  return () => { a = (a + 0x6d2b79f5) >>> 0; let t = a; t = Math.imul(t ^ (t >>> 15), t | 1); t ^= t + Math.imul(t ^ (t >>> 7), t | 61); return ((t ^ (t >>> 14)) >>> 0) / 4294967296; };
}
export const sigmoid = (z) => (z >= 0 ? 1 / (1 + Math.exp(-z)) : Math.exp(z) / (1 + Math.exp(z)));
const logsig = (z) => (z >= 0 ? -Math.log1p(Math.exp(-z)) : z - Math.log1p(Math.exp(z)));

// Candidate responses. h = helpfulness/correctness judged by the page author (0–1), words = length,
// flat = opens with flattery. These are the only features; the text is for reading.
export const PROMPTS = [
  { q: 'How do I reset my home router?', c: [
    { t: 'Unplug it for 30 seconds and plug it back in. If that doesn’t help, hold the recessed reset button for 10 seconds; this restores factory settings, so you’ll need to set up Wi-Fi again.', h: 0.95, words: 45, flat: 0 },
    { t: 'A long walkthrough of how routers, modems and DHCP leases work, followed by the same unplug-and-reset steps and a checklist for re-entering every setting.', h: 0.85, words: 190, flat: 0 },
    { t: 'Great question — you’re clearly comfortable with tech! Unplug it for 30 seconds, then plug it back in.', h: 0.6, words: 25, flat: 1 },
    { t: 'What a thoughtful question! Routers are fascinating devices with a rich history… (four paragraphs) …so try turning it off and on, or hold any button you can find for a while.', h: 0.3, words: 240, flat: 1 },
    { t: 'Call your internet provider.', h: 0.25, words: 4, flat: 0 },
    { t: 'Home routers can’t be reset by users.', h: 0.0, words: 7, flat: 0 },
  ] },
  { q: 'Summarize the main causes of World War I.', c: [
    { t: 'Historians usually cite militarism, rival alliance systems, imperial competition and nationalism; the assassination of Archduke Franz Ferdinand in 1914 triggered a chain of mobilizations.', h: 0.95, words: 30, flat: 0 },
    { t: 'An essay-length answer covering each cause country by country, with dates, treaties and historiographical debates.', h: 0.8, words: 230, flat: 0 },
    { t: 'Excellent question, you have a historian’s mind! It started because an archduke was assassinated.', h: 0.35, words: 18, flat: 1 },
    { t: 'What a brilliant question! War is complex… (several paragraphs of generalities about conflict, with few specifics about 1914).', h: 0.2, words: 260, flat: 1 },
    { t: 'The assassination of Franz Ferdinand.', h: 0.4, words: 5, flat: 0 },
    { t: 'It was caused by the Great Depression.', h: 0.0, words: 7, flat: 0 },
  ] },
  { q: 'Is it safe to take ibuprofen with coffee?', c: [
    { t: 'For most healthy adults, yes: there’s no known dangerous interaction. Both can irritate the stomach, so take ibuprofen with food, and check with a pharmacist if you have ulcers, kidney problems or take blood thinners.', h: 0.95, words: 40, flat: 0 },
    { t: 'A detailed review of NSAID pharmacology, caffeine metabolism and every known drug interaction, ending with the same practical advice.', h: 0.8, words: 210, flat: 0 },
    { t: 'Great that you’re being careful about your health! Yes, it’s fine.', h: 0.55, words: 12, flat: 1 },
    { t: 'What a wonderful, responsible question! Coffee has a long and fascinating history… (many paragraphs) …everyone is different, so it may or may not be safe.', h: 0.15, words: 250, flat: 1 },
    { t: 'Ask your doctor.', h: 0.3, words: 3, flat: 0 },
    { t: 'No, mixing them is extremely dangerous.', h: 0.0, words: 6, flat: 0 },
  ] },
  { q: 'Write a haiku about autumn.', c: [
    { t: 'Maple leaves let go / the river carries their red / farther than we walk', h: 0.95, words: 13, flat: 0 },
    { t: 'A haiku plus three paragraphs explaining its syllable count, imagery and the history of the form.', h: 0.7, words: 160, flat: 0 },
    { t: 'What a lovely request — you have a poet’s soul! Leaves fall in autumn / it is cold / the end', h: 0.3, words: 22, flat: 1 },
    { t: 'What a beautiful, inspired request! Autumn is a season… (a long free-verse poem that ignores the 5-7-5 form).', h: 0.2, words: 200, flat: 1 },
    { t: 'Autumn leaves.', h: 0.15, words: 2, flat: 0 },
    { t: 'Here is a haiku about spring: blossoms on the bough…', h: 0.0, words: 10, flat: 0 },
  ] },
];

// Gold utility: what we actually want (helpfulness; extra length past ~80 words and flattery cost something).
export const GOLD = { h: 4, overLen: -0.012, lenFree: 80, flat: -0.6 };
export function gold(c) { return GOLD.h * c.h + GOLD.overLen * Math.max(0, c.words - GOLD.lenFree) + GOLD.flat * c.flat; }

// Simulated annotators: Bradley–Terry on a score that values helpfulness but is also swayed by length
// and flattery. P(a ≻ b) = σ((s_a − s_b) / τ).
export const ANNOT = { h: 4, len: 0.012, flat: 1.2 };
export function annotScore(c, a = ANNOT) { return a.h * c.h + a.len * c.words + a.flat * c.flat; }

// Reward-model features (no access to h directly would be unrealistic; a real RM sees text — here it sees
// the same three features the annotators react to).
export function features(c) { return [c.h, c.words / 100, c.flat]; }

/** Sample pairwise comparisons. Returns [{p, a, b, y}] with y = 1 if a was preferred. */
export function samplePairs(rng, nPerPrompt, tau = 1, annot = ANNOT) {
  const out = [];
  PROMPTS.forEach((P, p) => {
    const K = P.c.length;
    for (let n = 0; n < nPerPrompt; n++) {
      const a = (rng() * K) | 0; let b = (rng() * (K - 1)) | 0; if (b >= a) b++;
      const pa = sigmoid((annotScore(P.c[a], annot) - annotScore(P.c[b], annot)) / tau);
      out.push({ p, a, b, y: rng() < pa ? 1 : 0 });
    }
  });
  return out;
}

export function splitPairs(rng, pairs, testFrac = 0.2) {
  const idx = pairs.map((_, i) => i);
  for (let i = idx.length - 1; i > 0; i--) { const j = (rng() * (i + 1)) | 0; [idx[i], idx[j]] = [idx[j], idx[i]]; }
  const nTest = Math.round(pairs.length * testFrac);
  return { test: idx.slice(0, nTest).map((i) => pairs[i]), train: idx.slice(nTest).map((i) => pairs[i]) };
}

/** Linear Bradley–Terry reward model r(y) = w·φ(y). One full-batch gradient step on the mean NLL (+ L2). */
export function rmStep(w, pairs, lr = 0.5, l2 = 1e-3) {
  const g = new Array(w.length).fill(0); let nll = 0;
  for (const q of pairs) {
    const fa = features(PROMPTS[q.p].c[q.a]), fb = features(PROMPTS[q.p].c[q.b]);
    let z = 0; for (let k = 0; k < w.length; k++) z += w[k] * (fa[k] - fb[k]);
    const s = q.y ? z : -z; nll -= logsig(s);
    const coef = (sigmoid(s) - 1) * (q.y ? 1 : -1); // d(-log σ(s))/dz
    for (let k = 0; k < w.length; k++) g[k] += coef * (fa[k] - fb[k]);
  }
  for (let k = 0; k < w.length; k++) { g[k] = g[k] / pairs.length + l2 * w[k]; w[k] -= lr * g[k]; }
  return nll / pairs.length + 0.5 * l2 * w.reduce((s, x) => s + x * x, 0);
}
export const rmReward = (w, c) => features(c).reduce((s, f, k) => s + w[k] * f, 0);

/** Fraction of pairs where the model's ranking matches the label (ties count as wrong). */
export function pairAccuracy(scoreFn, pairs, labelFn = (q) => q.y) {
  let ok = 0;
  for (const q of pairs) {
    const sa = scoreFn(PROMPTS[q.p].c[q.a]), sb = scoreFn(PROMPTS[q.p].c[q.b]);
    if ((sa > sb ? 1 : 0) === labelFn(q) && sa !== sb) ok++;
  }
  return ok / pairs.length;
}

export const uniformRef = () => PROMPTS.map((P) => P.c.map(() => 1 / P.c.length));

/** Closed-form optimum of E_π[r] − β KL(π‖π_ref): π*(y) ∝ π_ref(y) exp(r(y)/β). */
export function optimalPolicy(ref, rewardFn, beta) {
  return PROMPTS.map((P, p) => {
    const lg = P.c.map((c, i) => Math.log(ref[p][i]) + rewardFn(c) / beta);
    const m = Math.max(...lg), e = lg.map((x) => Math.exp(x - m)), z = e.reduce((a, b) => a + b, 0);
    return e.map((x) => x / z);
  });
}

export function kl(pi, ref) {
  let s = 0; pi.forEach((row, p) => row.forEach((x, i) => { if (x > 0) s += x * Math.log(x / ref[p][i]); }));
  return s / pi.length; // average over prompts
}
export function expect(pi, fn) {
  let s = 0; pi.forEach((row, p) => row.forEach((x, i) => { s += x * fn(PROMPTS[p].c[i]); }));
  return s / pi.length;
}

/** Tabular softmax policy logits (per prompt) → probabilities. */
export function softmaxRows(theta) {
  return theta.map((row) => { const m = Math.max(...row), e = row.map((x) => Math.exp(x - m)), z = e.reduce((a, b) => a + b, 0); return e.map((x) => x / z); });
}

/** One exact policy-gradient ascent step on J = E_π[r] − β KL(π‖π_ref) for a tabular softmax policy.
 *  ∂J/∂θ_y = π_y (A_y − Σ_y' π_y' A_y'), with A_y = r_y − β log(π_y/π_ref_y). */
export function rlhfStep(theta, ref, rewardFn, beta, lr) {
  const pi = softmaxRows(theta);
  theta.forEach((row, p) => {
    const A = row.map((_, i) => rewardFn(PROMPTS[p].c[i]) - beta * Math.log(pi[p][i] / ref[p][i]));
    const mean = A.reduce((s, a, i) => s + pi[p][i] * a, 0);
    row.forEach((_, i) => { row[i] += lr * pi[p][i] * (A[i] - mean); });
  });
  const pi2 = softmaxRows(theta);
  return { objective: expect(pi2, rewardFn) - beta * kl(pi2, ref) };
}

/** One full-batch gradient step on the DPO loss for a tabular policy:
 *  L = −E log σ(β [log π(y_w)/π_ref(y_w) − log π(y_l)/π_ref(y_l)]). */
export function dpoStep(theta, ref, pairs, beta, lr) {
  const pi = softmaxRows(theta), grad = theta.map((row) => row.map(() => 0));
  let loss = 0;
  for (const q of pairs) {
    const w = q.y ? q.a : q.b, l = q.y ? q.b : q.a, p = q.p;
    const z = beta * (Math.log(pi[p][w] / ref[p][w]) - Math.log(pi[p][l] / ref[p][l]));
    loss -= logsig(z);
    const coef = -(1 - sigmoid(z)) * beta; // dL/dz · β
    // d log π_w / dθ_k = 1[k=w] − π_k ; the π_k terms cancel between winner and loser
    grad[p][w] += coef; grad[p][l] -= coef;
  }
  theta.forEach((row, p) => row.forEach((_, i) => { row[i] -= lr * grad[p][i] / pairs.length * PROMPTS.length; }));
  return { loss: loss / pairs.length };
}

/** KL(a‖b) averaged over prompts, between two policies. */
export function klBetween(a, b) {
  let s = 0; a.forEach((row, p) => row.forEach((x, i) => { if (x > 0) s += x * Math.log(x / b[p][i]); }));
  return s / a.length;
}
