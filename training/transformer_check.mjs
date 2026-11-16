// Checks the hand-written backward pass of lab/transformer/model.js against finite differences,
// then trains the default induction model and prints how the induction-head score evolves.
//   node training/transformer_check.mjs            (gradient check + default run)
//   node training/transformer_check.mjs grad       (gradient check only)
import { TinyTransformer, TASKS, mulberry32, headScores } from '../lab/transformer/model.js';

function gradCheck(cfg, taskName, taskCfg) {
  const task = TASKS[taskName];
  const shape = task.make(taskCfg);
  const m = new TinyTransformer({ ...cfg, vocab: shape.vocab, ctx: shape.ctx, seed: 7 });
  // Use float64 copies for the numeric derivative to keep rounding error small.
  const rng = mulberry32(3), B = 3;
  const { ids, targets } = task.batch(rng, B, taskCfg);
  const { Tape } = m.constructor === TinyTransformer ? { Tape: null } : {};
  m.zeroGrad();
  const tape = new (class { constructor() { this.ops = []; } push(f) { this.ops.push(f); } backward() { for (let i = this.ops.length - 1; i >= 0; i--) this.ops[i](); } })();
  m.forward(ids, targets, B, tape); tape.backward();
  let worst = 0, checked = 0;
  const r2 = mulberry32(11);
  for (const [name, t] of Object.entries(m.params)) {
    for (let s = 0; s < 6; s++) {
      const i = (r2() * t.data.length) | 0, old = t.data[i], eps = 1e-2;
      t.data[i] = old + eps; const lp = m.forward(ids, targets, B).loss;
      t.data[i] = old - eps; const lm = m.forward(ids, targets, B).loss;
      t.data[i] = old;
      const num = (lp - lm) / (2 * eps), ana = t.grad ? t.grad[i] : 0;
      const rel = Math.abs(num - ana) / Math.max(1e-3, Math.abs(num) + Math.abs(ana));
      if (rel > worst) worst = rel;
      checked++;
      if (rel > 2e-2) console.log(`  ${name}[${i}] analytic ${ana.toExponential(3)} numeric ${num.toExponential(3)} rel ${rel.toExponential(2)}`);
    }
  }
  return { worst, checked };
}

const variants = [
  [{ d: 16, heads: 2, layers: 2, mlp: false, ln: false }, 'induction', { vocab: 12, half: 5 }],
  [{ d: 16, heads: 2, layers: 2, mlp: true, ln: true }, 'induction', { vocab: 12, half: 5 }],
  [{ d: 16, heads: 4, layers: 1, mlp: true, ln: true }, 'reverse', { digits: 6, len: 4 }],
];
let ok = true;
for (const [cfg, task, tc] of variants) {
  const r = gradCheck(cfg, task, tc);
  const pass = r.worst < 2e-2;
  ok &&= pass;
  console.log(`grad check ${JSON.stringify(cfg)} ${task}: ${r.checked} coords, worst relative error ${r.worst.toExponential(2)} ${pass ? 'PASS' : 'FAIL'}`);
}
if (process.argv[2] === 'grad') process.exit(ok ? 0 : 1);

// Default induction run (same defaults as the page).
const D = JSON.parse(process.env.CFG || '{"d":32,"heads":4,"layers":2,"vocab":32,"half":10,"lr":0.005,"batch":32,"steps":1500,"seed":1}');
const task = TASKS.induction, shape = task.make(D);
const m = new TinyTransformer({ vocab: shape.vocab, ctx: shape.ctx, d: D.d, heads: D.heads, layers: D.layers, seed: D.seed });
const rng = mulberry32(D.seed * 7919 + 1), evalRng = mulberry32(12345);
const ev = task.batch(evalRng, 64, D);
const t0 = Date.now();
for (let s = 1; s <= D.steps; s++) {
  const { ids, targets } = task.batch(rng, D.batch, D);
  const { loss } = m.step(ids, targets, D.batch, { lr: D.lr });
  if (s % 100 === 0 || s === 1) {
    const out = m.forward(ev.ids, ev.targets, 64);
    const T = shape.ctx, n = D.half;
    let first = 0, second = 0, c1 = 0, c2 = 0;
    for (let r = 0; r < out.perRow.length; r++) { const t = r % T; if (isNaN(out.perRow[r])) continue; if (t < n - 1) { first += out.perRow[r]; c1++; } else { second += out.perRow[r]; c2++; } }
    const sc = headScores(out.attn, 64, T, n);
    console.log(`step ${s} loss ${loss.toFixed(3)} | first-half ${(first / c1).toFixed(3)} second-half ${(second / c2).toFixed(3)} | induction max ${Math.max(...sc.induction).toFixed(2)} [${sc.induction.map((x) => x.toFixed(2)).join(' ')}] prev [${sc.prevToken.map((x) => x.toFixed(2)).join(' ')}] | ${((Date.now() - t0) / 1000).toFixed(1)}s`);
  }
}
process.exit(ok ? 0 : 1);
