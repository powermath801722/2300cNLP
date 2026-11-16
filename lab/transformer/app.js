import { el, $, ready, fail, plotly, plotLayout, PLOT_CONFIG, COLORS } from '/lab/shared/lab.js';

const TASK_DEFAULTS = {
  induction: { vocab: 24, ctx: 32, jump: 0.25 },
  copy: { digits: 10, len: 6 },
  reverse: { digits: 10, len: 6 },
};
const TASK_NOTES = {
  induction: 'Each sequence has its own random successor map π over 24 tokens; the next token is π(current) 75% of the time. Nothing can be memorized across sequences, so only looking back in context helps. With the default settings and seed 1, the induction heads form at roughly step 600–900.',
  copy: 'Six random digits, a separator, then the same six digits. Only the part after the separator is scored. One layer is enough: attend to the matching position in the first half.',
  reverse: 'Six random digits, a separator, then the digits in reverse order. The head has to learn an anti-diagonal positional pattern.',
};

const worker = new Worker(new URL('./worker.js', import.meta.url), { type: 'module' });
const lab = (window.__lab = window.__lab || {});
let cfg = null, info = null, running = false;
let hist = null, last = null, waiters = [];

const fmt = (x, d = 3) => (x == null || isNaN(x) ? '—' : Number(x).toFixed(d));

function readCfg() {
  const task = $('#task').value;
  return {
    task, ...TASK_DEFAULTS[task],
    layers: +$('#layers').value, heads: +$('#heads').value, d: +$('#d').value,
    ln: $('#ln').checked, mlp: $('#mlp').checked,
    lr: +$('#lr').value, batch: +$('#batch').value,
    seed: Math.max(1, Math.min(9999, Math.round(+$('#seed').value || 1))),
    exampleSeed: 777,
  };
}

function setStatus(text, frac) {
  const s = $('#status'); s.querySelector('.t').textContent = text;
  if (frac != null) s.querySelector('i').style.width = Math.round(Math.min(1, frac) * 100) + '%';
}

function setRunning(on) {
  running = on;
  $('#train').disabled = on; $('#step25').disabled = on; $('#pause').disabled = !on;
  $('#cfgpanel').querySelectorAll('select, input').forEach((x) => (x.disabled = on));
}

let prevTask = null;
function reset() {
  cfg = readCfg();
  if (cfg.task !== prevTask) { $('#until').value = cfg.task === 'induction' ? 1500 : 300; prevTask = cfg.task; }
  hist = { steps: [], loss: [], evalSteps: [], evalLoss: [], recall: [], other: [], tokenAcc: [], seqAcc: [], ind: [], prev: [] };
  last = null;
  $('#tasknote').textContent = TASK_NOTES[cfg.task];
  $('#scorepanel').hidden = cfg.task !== 'induction';
  $('#curvesub').textContent = cfg.task === 'induction'
    ? 'Loss on held-out sequences, split into tokens that can be recalled from earlier in the sequence and tokens that can’t. The right-hand chart tracks, for every head, how much attention it pays to the previous token (dotted) and to the token after an earlier occurrence of the current one (solid).'
    : 'Loss and accuracy on held-out sequences, scored only on the output half.';
  $('#ablt tbody').innerHTML = '<tr><td colspan="3" class="note">Train for a while, then run the ablations.</td></tr>';
  $('#ablnote').textContent = '';
  setRunning(false);
  setStatus('Building the model…', 0);
  worker.postMessage({ cmd: 'init', cfg });
}

function train(until) {
  setRunning(true);
  worker.postMessage({ cmd: 'run', until });
}

worker.onmessage = (e) => {
  const m = e.data;
  if (m.type === 'ready') {
    info = m.info;
    setStatus(`Ready · ${info.params.toLocaleString('en-US')} parameters · vocabulary ${info.vocab} · context ${info.ctx}`, 0);
    renderStats();
  } else if (m.type === 'progress') {
    hist.steps.push(m.step); hist.loss.push(m.loss);
    setStatus(`Training… step ${m.step} of ${m.until} · batch loss ${m.loss.toFixed(3)}`, m.step / m.until);
  } else if (m.type === 'eval') {
    last = m;
    hist.evalSteps.push(m.step); hist.evalLoss.push(m.evalLoss);
    if (m.scores) { hist.recall.push(m.recallLoss); hist.other.push(m.otherLoss); hist.ind.push(m.scores.induction); hist.prev.push(m.scores.prevToken); }
    else { hist.tokenAcc.push(m.tokenAcc); hist.seqAcc.push(m.seqAcc); }
    renderStats(); renderPlots(); renderHeads(); renderStrip();
    lab.last = m;
  } else if (m.type === 'done' || m.type === 'paused') {
    setRunning(false);
    setStatus(`${m.type === 'done' ? 'Finished' : 'Paused'} at step ${m.step} · ${info.params.toLocaleString('en-US')} parameters`, null);
    const w = waiters; waiters = []; w.forEach((f) => f(m));
  } else if (m.type === 'ablation') {
    renderAblation(m); lab.ablation = m;
  } else if (m.type === 'error') {
    setRunning(false); setStatus('Error: ' + m.message); fail(m.message);
  }
};
worker.onerror = (e) => { setRunning(false); setStatus('Worker failed: ' + (e.message || 'unknown error')); fail(e.message || 'Worker failed'); };

function renderStats() {
  const s = $('#stats'); s.innerHTML = '';
  const add = (v, l) => s.append(el('div', {}, [el('b', { text: v }), el('span', { text: l })]));
  const step = last ? last.step : 0;
  add(String(step), 'training steps');
  add(hist.loss.length ? fmt(hist.loss[hist.loss.length - 1]) : '—', 'latest batch loss');
  if (cfg.task === 'induction') {
    add(last ? fmt(last.recallLoss) : '—', 'held-out loss, recallable tokens');
    add(last ? fmt(last.otherLoss) : '—', 'held-out loss, other tokens');
    const ind = last ? Math.max(...last.scores.induction) : null;
    add(ind == null ? '—' : fmt(ind, 2), 'best induction score');
  } else {
    add(last ? (last.tokenAcc * 100).toFixed(1) + '%' : '—', 'output tokens correct');
    add(last ? (last.seqAcc * 100).toFixed(0) + '%' : '—', 'sequences fully correct');
  }
  add(info ? info.params.toLocaleString('en-US') : '—', 'parameters');
  add(last && step ? (last.trainMs / step).toFixed(0) + ' ms' : '—', 'per training step');
}

const headName = (i) => `L${Math.floor(i / cfg.heads) + 1}H${(i % cfg.heads) + 1}`;

async function renderPlots() {
  let P; try { P = await plotly(); } catch (e) { return; }
  const lossTraces = [{ x: hist.steps, y: hist.loss, name: 'training batch', mode: 'lines', line: { color: 'rgba(228,231,235,.25)', width: 1 } }];
  if (cfg.task === 'induction') {
    lossTraces.push({ x: hist.evalSteps, y: hist.recall, name: 'held-out · recallable', mode: 'lines', line: { color: COLORS[0], width: 2.5 } });
    lossTraces.push({ x: hist.evalSteps, y: hist.other, name: 'held-out · other', mode: 'lines', line: { color: COLORS[3], width: 2 } });
    lossTraces.push({ x: [0, Math.max(25, ...hist.evalSteps)], y: [Math.log(info.vocab), Math.log(info.vocab)], name: `chance (ln ${info.vocab})`, mode: 'lines', line: { color: 'rgba(228,231,235,.35)', dash: 'dot', width: 1 } });
  } else {
    lossTraces.push({ x: hist.evalSteps, y: hist.evalLoss, name: 'held-out loss', mode: 'lines', line: { color: COLORS[0], width: 2.5 } });
    lossTraces.push({ x: hist.evalSteps, y: hist.tokenAcc, name: 'token accuracy', mode: 'lines', yaxis: 'y2', line: { color: COLORS[2], width: 2 } });
  }
  const L1 = plotLayout({
    margin: { l: 48, r: cfg.task === 'induction' ? 12 : 44, t: 12, b: 70 },
    xaxis: { ...plotLayout().xaxis, title: { text: 'training step' } },
    yaxis: { ...plotLayout().yaxis, title: { text: 'cross-entropy (nats)' }, rangemode: 'tozero' },
    legend: { orientation: 'h', y: -0.28, bgcolor: 'rgba(0,0,0,0)', font: { size: 11 } },
  });
  if (cfg.task !== 'induction') L1.yaxis2 = { overlaying: 'y', side: 'right', range: [0, 1], color: '#9aa3ad', tickformat: '.0%', showgrid: false };
  P.react('lossplot', lossTraces, L1, PLOT_CONFIG);
  if (cfg.task === 'induction' && hist.ind.length) {
    const nH = hist.ind[0].length, tr = [];
    for (let i = 0; i < nH; i++) {
      const c = COLORS[i % COLORS.length];
      tr.push({ x: hist.evalSteps, y: hist.ind.map((s) => s[i]), name: headName(i) + ' induction', mode: 'lines', line: { color: c, width: 2.2 } });
      tr.push({ x: hist.evalSteps, y: hist.prev.map((s) => s[i]), name: headName(i) + ' prev-token', mode: 'lines', line: { color: c, width: 1.5, dash: 'dot' } });
    }
    P.react('scoreplot', tr, plotLayout({
      margin: { l: 48, r: 12, t: 12, b: 70 },
      xaxis: { ...plotLayout().xaxis, title: { text: 'training step' } },
      yaxis: { ...plotLayout().yaxis, title: { text: 'score' }, range: [-0.1, 1] },
      legend: { orientation: 'h', y: -0.28, bgcolor: 'rgba(0,0,0,0)', font: { size: 11 } },
    }), PLOT_CONFIG);
  }
}

function colorFor(a) {
  // dark navy → accent blue → near white
  const t = Math.max(0, Math.min(1, a));
  const c0 = [18, 24, 32], c1 = [126, 163, 200], c2 = [236, 244, 252];
  const [p, q, u] = t < 0.6 ? [c0, c1, t / 0.6] : [c1, c2, (t - 0.6) / 0.4];
  return `rgb(${p.map((v, i) => Math.round(v + (q[i] - v) * u)).join(',')})`;
}

function renderHeads() {
  if (!last) return;
  const ex = last.example, T = ex.ids.length, host = $('#headgrid');
  const L = ex.attn.length, H = ex.attn[0].length;
  if (host.children.length !== L * H) {
    host.innerHTML = '';
    for (let l = 0; l < L; l++) for (let h = 0; h < H; h++) {
      const cv = el('canvas', { width: T * 8, height: T * 8, role: 'img' });
      host.append(el('div', { class: 'head' }, [cv, el('div', { class: 'cap' }, [el('b', { text: `Layer ${l + 1} · head ${h + 1}` }), el('span', { class: 'sc' })])]));
    }
  }
  // positions right after earlier occurrences (expected induction targets) for the example
  const succ = [];
  if (cfg.task === 'induction') for (let t = 0; t < T; t++) { const s = []; for (let u = 0; u < t; u++) if (ex.ids[u] === ex.ids[t]) s.push(u + 1); succ.push(s); }
  [...host.children].forEach((box, i) => {
    const l = Math.floor(i / H), h = i % H, A = ex.attn[l][h], cv = box.querySelector('canvas'), g = cv.getContext('2d'), c = cv.width / T;
    g.fillStyle = '#121820'; g.fillRect(0, 0, cv.width, cv.height);
    for (let t = 0; t < T; t++) for (let u = 0; u <= t; u++) { g.fillStyle = colorFor(Math.sqrt(A[t * T + u])); g.fillRect(u * c, t * c, c, c); }
    if (cfg.task === 'induction') {
      g.strokeStyle = 'rgba(216,182,106,.9)'; g.lineWidth = 1;
      succ.forEach((s, t) => s.forEach((u) => g.strokeRect(u * c + 1.5, t * c + 1.5, c - 3, c - 3)));
    }
    const sc = box.querySelector('.sc');
    if (last.scores) {
      const k = l * H + h, ind = last.scores.induction[k], pv = last.scores.prevToken[k];
      sc.textContent = `ind ${fmt(ind, 2)} · prev ${fmt(pv, 2)}`;
      cv.setAttribute('aria-label', `Attention pattern of layer ${l + 1}, head ${h + 1}. Induction score ${fmt(ind, 2)}, previous-token score ${fmt(pv, 2)}.`);
    } else {
      sc.textContent = '';
      cv.setAttribute('aria-label', `Attention pattern of layer ${l + 1}, head ${h + 1}.`);
    }
  });
  $('#attnhint').textContent = cfg.task === 'induction'
    ? 'Gold outlines mark where an induction head should look: the position just after an earlier occurrence of the current token. A previous-token head lights up the line just below the diagonal.'
    : cfg.task === 'reverse' ? 'For reversal, output position k+1+j has to look at input position k−1−j: an anti-diagonal.' : 'For copying, output position k+j has to look at input position j: a line parallel to the diagonal.';
  $('#attnlegend').innerHTML = cfg.task === 'induction' ? '<span><i class="sw" style="background:#ecf4fc"></i>high attention</span><span><i class="sw" style="background:#3c5470"></i>low</span><span><i class="sw" style="border-color:#d8b66a"></i>expected induction target</span>' : '<span><i class="sw" style="background:#ecf4fc"></i>high attention</span><span><i class="sw" style="background:#3c5470"></i>low</span>';
}

function renderStrip() {
  if (!last) return;
  const ex = last.example, host = $('#strip'); host.innerHTML = '';
  const SEP = cfg.task === 'induction' ? -1 : cfg.digits;
  ex.ids.forEach((id, t) => {
    const loss = ex.loss[t];
    const bg = loss == null ? 'transparent' : `hsl(${Math.round(Math.max(0, 120 - loss * 40))} 45% 28%)`;
    const pr = ex.preds[t];
    const title = loss == null ? `position ${t}: token ${id} (next token not scored)` : `position ${t}: token ${id}; next is ${ex.targets[t]}; model’s top guess ${pr.tok} (p = ${pr.p.toFixed(2)}); p(correct) = ${pr.pTarget.toFixed(2)}; loss ${loss.toFixed(2)}`;
    host.append(el('span', { class: 'tok' + (ex.rep && ex.rep[t] ? ' rep' : ''), style: { background: bg }, title, text: id === SEP ? '|' : String(id) }));
  });
  $('#stripnote').innerHTML = cfg.task === 'induction'
    ? 'Each box is a token id; its colour is the model’s loss when predicting the <em>next</em> token from there (green = confident and right, red = surprised). Dashed outlines mark positions where the next token can be recalled from earlier in the sequence. Hover a box for the prediction.'
    : 'Digits, then <b>|</b> (the separator), then the output. Colour is the loss of the next-token prediction on the scored output positions.';
}

function renderAblation(m) {
  const tb = $('#ablt tbody'); tb.innerHTML = '';
  tb.append(el('tr', {}, [el('td', { text: 'none (full model)' }), el('td', { text: fmt(m.full) }), el('td', { text: '—' })]));
  const rows = m.rows.slice().sort((a, b) => b.loss - a.loss);
  for (const r of rows) {
    const dlt = r.loss - m.full;
    tb.append(el('tr', {}, [el('td', { text: r.head < 0 ? `Layer ${r.layer + 1} · all heads` : `Layer ${r.layer + 1} · head ${r.head + 1}` }), el('td', { text: fmt(r.loss) }), el('td', { text: (dlt >= 0 ? '+' : '') + fmt(dlt), style: { color: dlt > 0.3 ? '#d98c7a' : 'inherit' } })]));
  }
  $('#ablnote').textContent = `Measured at step ${m.step} on ${cfg.task === 'induction' ? 'recallable tokens of 32 held-out sequences' : 'the scored tokens of 32 held-out sequences'}.`;
}

// ---------- wiring ----------
$('#train').addEventListener('click', () => {
  const until = Math.max(25, Math.round((+$('#until').value || 1500) / 25) * 25);
  if (last && last.step >= until) { setStatus(`Already at step ${last.step}; raise “Train until step” to keep going.`); return; }
  train(until);
});
$('#step25').addEventListener('click', () => { setRunning(true); worker.postMessage({ cmd: 'step', n: 25 }); });
$('#pause').addEventListener('click', () => worker.postMessage({ cmd: 'pause' }));
$('#reset').addEventListener('click', reset);
$('#ablate').addEventListener('click', () => worker.postMessage({ cmd: 'ablate' }));
let exSeed = 777;
$('#newex').addEventListener('click', () => { exSeed++; worker.postMessage({ cmd: 'example', seed: exSeed }); });
for (const id of ['task', 'layers', 'heads', 'd', 'lr', 'batch', 'seed', 'ln', 'mlp']) $('#' + id).addEventListener('change', reset);

// test/automation hooks
lab.cfg = () => cfg;
lab.hist = () => hist;
lab.trainUntil = (until) => new Promise((res) => { waiters.push(res); train(until); });
lab.ablate = () => new Promise((res) => { const h = (e) => { if (e.data.type === 'ablation') { worker.removeEventListener('message', h); res(e.data); } }; worker.addEventListener('message', h); worker.postMessage({ cmd: 'ablate' }); });
lab.setTask = (task) => { $('#task').value = task; reset(); };

try {
  reset();
  ready();
} catch (e) { fail(e.message); }
