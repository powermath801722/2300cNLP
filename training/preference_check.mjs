// Checks the preference lab's math: reward-model recovery, the over-optimization curve, and that exact
// policy-gradient RLHF and DPO converge to the closed-form KL-regularized optimum.
//   node training/preference_check.mjs
import * as C from '../lab/preference/core.js';

const rng = C.mulberry32(1), tau = 1;
const pairs = C.samplePairs(rng, 400, tau);
const { train, test } = C.splitPairs(C.mulberry32(2), pairs);
const w = [0, 0, 0];
let nll; for (let i = 0; i < 3000; i++) nll = C.rmStep(w, train, 0.5, 1e-4);
const rm = (c) => C.rmReward(w, c);
console.log('RM weights [h, words/100, flat]:', w.map((x) => x.toFixed(2)).join(', '), '| annotator weights:', [C.ANNOT.h, C.ANNOT.len * 100, C.ANNOT.flat].join(', '), '| train NLL', nll.toFixed(3));
console.log('held-out accuracy vs annotator labels:', C.pairAccuracy(rm, test).toFixed(3),
  '| Bayes-optimal (true annotator score):', C.pairAccuracy((c) => C.annotScore(c), test).toFixed(3),
  '| agreement with gold ranking:', C.pairAccuracy(rm, test, (q) => (C.gold(C.PROMPTS[q.p].c[q.a]) > C.gold(C.PROMPTS[q.p].c[q.b]) ? 1 : 0)).toFixed(3));

const ref = C.uniformRef();
console.log('\nbeta   KL     sqrtKL  proxy(RM)  gold');
for (const beta of [100, 10, 5, 3, 2, 1.5, 1, 0.7, 0.5, 0.3, 0.2, 0.1, 0.05, 0.02]) {
  const pi = C.optimalPolicy(ref, rm, beta);
  const K = C.kl(pi, ref);
  console.log(beta.toString().padEnd(6), K.toFixed(3).padEnd(6), Math.sqrt(K).toFixed(3).padEnd(7), C.expect(pi, rm).toFixed(3).padEnd(10), C.expect(pi, C.gold).toFixed(3));
}
const beta = 1;
const target = C.optimalPolicy(ref, rm, beta);
const th = ref.map((r) => r.map((x) => Math.log(x)));
for (let i = 0; i < 2000; i++) C.rlhfStep(th, ref, rm, beta, 1);
console.log('\nRLHF (exact policy gradient, β=1) KL(π_RL ‖ π*_RM) =', C.klBetween(C.softmaxRows(th), target).toExponential(2));
const td = ref.map((r) => r.map((x) => Math.log(x)));
let dl; for (let i = 0; i < 4000; i++) dl = C.dpoStep(td, ref, train, beta, 2).loss;
const pDPO = C.softmaxRows(td);
const pAnn = C.optimalPolicy(ref, (c) => C.annotScore(c) / tau, beta);
console.log('DPO (β=1) loss', dl.toFixed(3), '| KL(π_DPO ‖ π*_RM) =', C.klBetween(pDPO, target).toExponential(2), '| KL(π_DPO ‖ π*_annotator) =', C.klBetween(pDPO, pAnn).toExponential(2));
console.log('gold: π_ref', C.expect(ref, C.gold).toFixed(3), '| RLHF', C.expect(C.softmaxRows(th), C.gold).toFixed(3), '| DPO', C.expect(pDPO, C.gold).toFixed(3), '| best possible', (C.PROMPTS.reduce((s, P) => s + Math.max(...P.c.map(C.gold)), 0) / 4).toFixed(3));
