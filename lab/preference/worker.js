// Runs the preference lab's loops off the main thread.
// in:  {cmd:'collect', n, tau, seed} · {cmd:'fitRM', iters, l2} · {cmd:'sweep'} · {cmd:'policy', beta}
//      {cmd:'align', beta, iters} · {cmd:'pause'}
// out: {type:'data'|'rmProgress'|'rm'|'sweep'|'policy'|'alignProgress'|'align'|'paused'|'error', ...}
import * as C from './core.js';

let pairs = null, train = null, test = null, tau = 1, w = null, ref = C.sftRef();
let running = false, job = 0;
const post = (type, d = {}) => self.postMessage({ type, ...d });
const rmFn = () => (c) => C.rmReward(w, c);

function collect({ n, tau: t, seed }) {
  tau = t;
  pairs = C.samplePairs(C.mulberry32(seed), n, tau);
  ({ train, test } = C.splitPairs(C.mulberry32(seed + 1), pairs));
  w = null;
  // how often the longer / flattering answer won, among pairs that differ in that respect
  let longer = 0, nl = 0, flat = 0, nf = 0, goldAgree = 0;
  for (const q of pairs) {
    const A = C.PROMPTS[q.p].c[q.a], B = C.PROMPTS[q.p].c[q.b], win = q.y ? A : B, lose = q.y ? B : A;
    if (A.words !== B.words) { nl++; if (win.words > lose.words) longer++; }
    if (A.flat !== B.flat) { nf++; if (win.flat > lose.flat) flat++; }
    if (C.gold(win) > C.gold(lose)) goldAgree++;
  }
  post('data', { nPairs: pairs.length, nTrain: train.length, nTest: test.length, longerWon: longer / nl, flatWon: flat / nf, goldAgree: goldAgree / pairs.length,
    bayes: C.pairAccuracy((c) => C.annotScore(c), test) });
}

async function fitRM({ iters, l2 }) {
  const my = ++job; running = true;
  w = [0, 0, 0];
  const curve = [];
  for (let i = 1; i <= iters; i++) {
    const nll = C.rmStep(w, train, 0.5, l2);
    if (i % 25 === 0 || i === iters) {
      const test_nll = testNLL();
      curve.push({ i, nll, test: test_nll });
      post('rmProgress', { i, iters, nll, test: test_nll, w: w.slice() });
      await new Promise((r) => setTimeout(r, 0));
      if (!running || my !== job) { post('paused', { what: 'rm' }); return; }
    }
  }
  running = false;
  const rm = rmFn();
  post('rm', {
    w: w.slice(), curve,
    accTest: C.pairAccuracy(rm, test), accTrain: C.pairAccuracy(rm, train),
    goldAgree: C.pairAccuracy(rm, test, (q) => (C.gold(C.PROMPTS[q.p].c[q.a]) > C.gold(C.PROMPTS[q.p].c[q.b]) ? 1 : 0)),
    rewards: C.PROMPTS.map((P) => P.c.map((c) => C.rmReward(w, c))),
  });
}

function testNLL() {
  let s = 0;
  for (const q of test) {
    const z = C.rmReward(w, C.PROMPTS[q.p].c[q.a]) - C.rmReward(w, C.PROMPTS[q.p].c[q.b]);
    const p = C.sigmoid(z); s -= Math.log(Math.max(1e-12, q.y ? p : 1 - p));
  }
  return s / test.length;
}

function sweep() {
  const rm = rmFn(), out = [];
  for (let k = 0; k <= 60; k++) {
    const beta = Math.pow(10, 2 - k * (4 / 60)); // 100 → 0.01
    const pi = C.optimalPolicy(ref, rm, beta), K = C.kl(pi, ref);
    out.push({ beta, kl: K, sqrtKL: Math.sqrt(K), proxy: C.expect(pi, rm), gold: C.expect(pi, C.gold), annot: C.expect(pi, (c) => C.annotScore(c)) });
  }
  const best = out.reduce((a, b) => (b.gold > a.gold ? b : a));
  post('sweep', { points: out, best, refGold: C.expect(ref, C.gold), refProxy: C.expect(ref, rm), bestPossible: C.PROMPTS.reduce((s, P) => s + Math.max(...P.c.map(C.gold)), 0) / C.PROMPTS.length });
}

function policy({ beta }) {
  const rm = rmFn(), pi = C.optimalPolicy(ref, rm, beta), K = C.kl(pi, ref);
  post('policy', { beta, pi, ref, kl: K, proxy: C.expect(pi, rm), gold: C.expect(pi, C.gold) });
}

async function align({ beta, iters }) {
  const my = ++job; running = true;
  const rm = rmFn(), target = C.optimalPolicy(ref, rm, beta);
  const thR = ref.map((r) => r.map((x) => Math.log(x))), thD = ref.map((r) => r.map((x) => Math.log(x)));
  const rt = C.PROMPTS.map((P) => P.c.map(() => 0));
  const curve = [];
  // DPO's implicit reward is r = β·log(π/π_ref). With step size 2/β² on the logits, r moves exactly like a
  // tabular Bradley–Terry reward fit with step size 2, so the two can be compared at every iteration.
  const tabTarget = () => C.PROMPTS.map((P, p) => {
    const lg = P.c.map((_, j) => Math.log(ref[p][j]) + rt[p][j] / beta), m = Math.max(...lg), e = lg.map((x) => Math.exp(x - m)), z = e.reduce((a, b) => a + b, 0);
    return e.map((x) => x / z);
  });
  for (let i = 1; i <= iters; i++) {
    C.rlhfStep(thR, ref, rm, beta, 1);
    const d = C.dpoStep(thD, ref, train, beta, 2 / (beta * beta));
    C.tabularBTStep(rt, train, 2);
    if (i % 20 === 0 || i === iters) {
      const piR = C.softmaxRows(thR), piD = C.softmaxRows(thD), tt = tabTarget();
      const pt = { i, klRL: C.klBetween(piR, target), klDPOtab: C.klBetween(piD, tt), klDPOrm: C.klBetween(piD, target),
        goldRL: C.expect(piR, C.gold), goldDPO: C.expect(piD, C.gold), dpoLoss: d.loss };
      curve.push(pt);
      post('alignProgress', { ...pt, iters });
      await new Promise((r) => setTimeout(r, 0));
      if (!running || my !== job) { post('paused', { what: 'align' }); return; }
    }
  }
  running = false;
  const piR = C.softmaxRows(thR), piD = C.softmaxRows(thD);
  post('align', { beta, curve, piRL: piR, piDPO: piD, target, tabTarget: tabTarget(), ref,
    gold: { ref: C.expect(ref, C.gold), rl: C.expect(piR, C.gold), dpo: C.expect(piD, C.gold), target: C.expect(target, C.gold) },
    proxy: { rl: C.expect(piR, rm), dpo: C.expect(piD, rm) },
    kl: { rl: C.kl(piR, ref), dpo: C.kl(piD, ref) } });
}

self.onmessage = async (e) => {
  const m = e.data;
  try {
    if (m.cmd === 'collect') collect(m);
    else if (m.cmd === 'fitRM') await fitRM(m);
    else if (m.cmd === 'sweep') sweep();
    else if (m.cmd === 'policy') policy(m);
    else if (m.cmd === 'align') await align(m);
    else if (m.cmd === 'pause') { running = false; }
  } catch (err) { running = false; post('error', { message: err.message || String(err) }); }
};
