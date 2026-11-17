// Published scaling-law fits and the formulas derived from them. Every constant is cited on the page.
//   Hoffmann et al. (2022) "Approach 3" parametric fit, values as reported to 4 digits in Besiroglu et al.
//   (2024), Table 1 ("Hoffmann et al.'s estimate"); the paper's Eq. 10 prints E=1.69, A=406.4, B=410.7,
//   α=0.34, β=0.28.  Besiroglu et al. (2024) refit of the same data: E=1.8172, A=482.01, B=2085.43,
//   α=0.3478, β=0.3658.
//   Kaplan et al. (2020): L(N)=(Nc/N)^αN, αN=0.076, Nc=8.8e13 (non-embedding params); L(D)=(Dc/D)^αD,
//   αD=0.095, Dc=5.4e13 tokens; L(Cmin)=(Cc/Cmin)^αC, αC=0.050, Cc=3.1e8 PF-days; compute-efficient
//   training puts N ∝ Cmin^0.73. Losses are nats/token on WebText2 with GPT-2's tokenizer — not comparable
//   with the Chinchilla fits, which are on MassiveText with a SentencePiece tokenizer.
export const FITS = {
  hoffmann: { key: 'hoffmann', label: 'Hoffmann et al. 2022 (Approach 3)', E: 1.6934, A: 406.4, B: 410.7, alpha: 0.3392, beta: 0.2849 },
  besiroglu: { key: 'besiroglu', label: 'Besiroglu et al. 2024 (replication)', E: 1.8172, A: 482.01, B: 2085.43, alpha: 0.3478, beta: 0.3658 },
};
export const KAPLAN = { alphaN: 0.076, Nc: 8.8e13, alphaD: 0.095, Dc: 5.4e13, alphaC: 0.05, Cc: 3.1e8, nExp: 0.73 };
export const PF_DAY = 1e15 * 86400; // FLOPs in one petaflop/s-day

export const loss = (f, N, D) => f.E + f.A / Math.pow(N, f.alpha) + f.B / Math.pow(D, f.beta);
export const flops = (N, D) => 6 * N * D;

/** Compute-optimal allocation for budget C (FLOPs) under C = 6ND: closed form from minimizing L(N, C/6N). */
export function optimal(f, C) {
  const s = f.alpha + f.beta, G = Math.pow((f.alpha * f.A) / (f.beta * f.B), 1 / s);
  const N = G * Math.pow(C / 6, f.beta / s), D = (C / 6) / N;
  return { N, D, L: loss(f, N, D), a: f.beta / s, b: f.alpha / s, ratio: D / N };
}

/** Smallest compute that reaches target loss (bisection on the compute-optimal frontier). */
export function computeFor(f, target) {
  if (target <= f.E) return Infinity;
  let lo = 1e10, hi = 1e40;
  for (let i = 0; i < 200; i++) { const mid = Math.sqrt(lo * hi); if (optimal(f, mid).L > target) lo = mid; else hi = mid; }
  return hi;
}

/** Kaplan et al. single-variable power laws (nats/token on WebText2). */
export const kaplanN = (N) => Math.pow(KAPLAN.Nc / N, KAPLAN.alphaN);
export const kaplanD = (D) => Math.pow(KAPLAN.Dc / D, KAPLAN.alphaD);
export const kaplanC = (C) => Math.pow(KAPLAN.Cc / (C / PF_DAY), KAPLAN.alphaC);
export const kaplanND = (N, D) => Math.pow(Math.pow(KAPLAN.Nc / N, KAPLAN.alphaN / KAPLAN.alphaD) + KAPLAN.Dc / D, KAPLAN.alphaD);

/** An allocation that grows N ∝ C^0.73 (Kaplan's exponent), drawn through an anchor model (N0, D0). */
export function kaplanAllocation(C, anchor) {
  const C0 = flops(anchor.N, anchor.D), N = anchor.N * Math.pow(C / C0, KAPLAN.nExp);
  return { N, D: C / (6 * N) };
}

// Publicly reported parameter and training-token counts (C ≈ 6ND is an estimate, ignoring attention FLOPs).
export const MODELS = [
  { name: 'GPT-3', N: 175e9, D: 300e9, src: 'Brown et al. 2020' },
  { name: 'Gopher', N: 280e9, D: 300e9, src: 'Rae et al. 2021' },
  { name: 'MT-NLG', N: 530e9, D: 270e9, src: 'Smith et al. 2022' },
  { name: 'PaLM', N: 540e9, D: 780e9, src: 'Chowdhery et al. 2022' },
  { name: 'Chinchilla', N: 70e9, D: 1.4e12, src: 'Hoffmann et al. 2022' },
  { name: 'LLaMA-65B', N: 65.2e9, D: 1.4e12, src: 'Touvron et al. 2023' },
  { name: 'Llama 2 70B', N: 70e9, D: 2e12, src: 'Touvron et al. 2023b' },
  { name: 'Llama 3 8B', N: 8e9, D: 15e12, src: 'Meta 2024 (“over 15T tokens”)' },
];
