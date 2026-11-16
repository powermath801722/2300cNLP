// Trains the tiny transformer off the main thread.
// Messages in:  {cmd:'init', cfg} · {cmd:'run', until} · {cmd:'pause'} · {cmd:'step', n} · {cmd:'ablate'} · {cmd:'example', seed}
// Messages out: {type:'ready', info} · {type:'progress', ...} · {type:'eval', ...} · {type:'paused'|'done', step} · {type:'ablation', rows} · {type:'error', message}
import { TinyTransformer, TASKS, mulberry32, headScores } from './model.js';

let model = null, cfg = null, task = null, rng = null, evalB = null, example = null;
let step = 0, running = false, until = 0, t0 = 0, trainMs = 0;
const EVAL_EVERY = 25, EVAL_N = 32;

function post(type, data = {}) { self.postMessage({ type, ...data }); }

function init(c) {
  running = false;
  cfg = c; task = TASKS[c.task];
  const shape = task.make(c);
  model = new TinyTransformer({ vocab: shape.vocab, ctx: shape.ctx, d: c.d, heads: c.heads, layers: c.layers, mlp: c.mlp, ln: c.ln, seed: c.seed });
  rng = mulberry32(c.seed * 7919 + 1);
  evalB = task.batch(mulberry32(12345), EVAL_N, c);
  example = task.batch(mulberry32(c.exampleSeed ?? 777), 1, c);
  step = 0; trainMs = 0;
  post('ready', { info: { params: model.nParams(), vocab: shape.vocab, ctx: shape.ctx, task: c.task } });
  evaluate();
}

function evaluate() {
  const T = model.cfg.ctx;
  const out = model.forward(evalB.ids, evalB.targets, EVAL_N);
  const res = { step, evalLoss: out.loss };
  if (cfg.task === 'induction') {
    let a = 0, b = 0, ca = 0, cb = 0;
    for (let r = 0; r < out.perRow.length; r++) {
      if (isNaN(out.perRow[r])) continue;
      if (evalB.rep[r]) { a += out.perRow[r]; ca++; } else { b += out.perRow[r]; cb++; }
    }
    res.recallLoss = a / ca; res.otherLoss = b / cb;
    res.scores = headScores(out.attn, evalB.meta, T);
  } else {
    // exact-token accuracy and whole-sequence accuracy on the scored (output) positions
    const V = model.cfg.vocab; let ok = 0, n = 0, seqOk = 0;
    for (let s = 0; s < EVAL_N; s++) {
      let all = true;
      for (let t = 0; t < T; t++) {
        const r = s * T + t, y = evalB.targets[r]; if (y < 0) continue;
        let best = 0; for (let j = 1; j < V; j++) if (out.logits.data[r * V + j] > out.logits.data[r * V + best]) best = j;
        n++; if (best === y) ok++; else all = false;
      }
      if (all) seqOk++;
    }
    res.tokenAcc = ok / n; res.seqAcc = seqOk / EVAL_N;
  }
  // example: attention maps for every head and per-token loss / prediction
  const ex = model.forward(example.ids, example.targets, 1);
  const V = model.cfg.vocab, preds = [];
  for (let t = 0; t < T; t++) {
    let best = 0; for (let j = 1; j < V; j++) if (ex.logits.data[t * V + j] > ex.logits.data[t * V + best]) best = j;
    preds.push({ p: ex.probs[t * V + best], tok: best, pTarget: example.targets[t] >= 0 ? ex.probs[t * V + example.targets[t]] : null });
  }
  res.example = {
    ids: Array.from(example.ids), targets: Array.from(example.targets),
    rep: example.rep ? Array.from(example.rep) : null,
    loss: Array.from(ex.perRow, (x) => (isNaN(x) ? null : x)), preds,
    attn: ex.attn.map((layer) => layer[0].map((A) => Array.from(A))),
  };
  res.trainMs = trainMs;
  post('eval', res);
}

function loop() {
  if (!running) return;
  const tStart = performance.now();
  // train in short slices so pause/reset messages are handled promptly
  while (running && step < until && performance.now() - tStart < 60) {
    const { ids, targets } = task.batch(rng, cfg.batch, cfg);
    const s0 = performance.now();
    const { loss, gradNorm } = model.step(ids, targets, cfg.batch, { lr: cfg.lr });
    trainMs += performance.now() - s0;
    step++;
    if (step % 5 === 0 || step === until) post('progress', { step, loss, gradNorm, until });
    if (step % EVAL_EVERY === 0 || step === until) evaluate();
  }
  if (step >= until) { running = false; post('done', { step }); return; }
  setTimeout(loop, 0);
}

function ablate() {
  // loss on recallable tokens (induction) or all scored tokens, with each head switched off in turn
  const L = model.cfg.layers, H = model.cfg.heads, rows = [];
  const lossOf = (mask) => {
    const out = model.forward(evalB.ids, evalB.targets, EVAL_N, null, mask);
    if (cfg.task !== 'induction') return out.loss;
    let a = 0, c = 0; for (let r = 0; r < out.perRow.length; r++) if (evalB.rep[r] && !isNaN(out.perRow[r])) { a += out.perRow[r]; c++; }
    return a / c;
  };
  const full = lossOf(null);
  for (let l = 0; l < L; l++) for (let h = 0; h < H; h++) {
    const mask = Array.from({ length: L }, (_, i) => Array.from({ length: H }, (_, j) => !(i === l && j === h)));
    rows.push({ layer: l, head: h, loss: lossOf(mask) });
  }
  // whole layers: shows redundancy when two heads do the same job
  if (H > 1) for (let l = 0; l < L; l++) {
    const mask = Array.from({ length: L }, (_, i) => Array.from({ length: H }, () => i !== l));
    rows.push({ layer: l, head: -1, loss: lossOf(mask) });
  }
  post('ablation', { full, rows, step });
}

self.onmessage = (e) => {
  const m = e.data;
  try {
    if (m.cmd === 'init') init(m.cfg);
    else if (m.cmd === 'run') { until = m.until; if (!running) { running = true; setTimeout(loop, 0); } }
    else if (m.cmd === 'step') { until = step + m.n; if (!running) { running = true; setTimeout(loop, 0); } }
    else if (m.cmd === 'pause') { running = false; post('paused', { step }); }
    else if (m.cmd === 'ablate') ablate();
    else if (m.cmd === 'example') { example = task.batch(mulberry32(m.seed), 1, cfg); evaluate(); }
  } catch (err) { running = false; post('error', { message: err.message || String(err) }); }
};
