import { el, $, ready, fail, plotly, plotLayout, PLOT_CONFIG, COLORS, getJSON } from '/lab/shared/lab.js';
import { progressUI } from '/lab/shared/hf.js';
import { lexModels, distilbert, sigmoid } from '/lab/attribution/models.js';
import { rng } from '/lab/attribution/text.js';

const NAMES = { bert: 'DistilBERT', lr: 'Logistic regression', nb: 'Naive Bayes' };
const KEYS = ['bert', 'lr', 'nb'];

// ---------- metrics ----------
/** z: log-odds of positive, y: 0/1 labels, T: temperature. */
function preds(z, y, T = 1) {
  return z.map((v, i) => { const p = sigmoid(v / T), yhat = p >= 0.5 ? 1 : 0; return { p, conf: Math.max(p, 1 - p), ok: yhat === y[i] ? 1 : 0, y: y[i] }; });
}
function bins(P, nb, scheme) {
  const out = Array.from({ length: nb }, () => ({ n: 0, acc: 0, conf: 0 }));
  if (scheme === 'mass') {
    const s = P.map((p, i) => [p.conf, i]).sort((a, b) => a[0] - b[0]);
    s.forEach(([c, i], r) => { const b = Math.min(nb - 1, Math.floor((r * nb) / s.length)); const o = out[b]; o.n++; o.acc += P[i].ok; o.conf += P[i].conf; o.min = Math.min(o.min ?? c, c); o.max = Math.max(o.max ?? c, c); });
  } else {
    // binary task: confidence ∈ [0.5, 1], so equal-width bins span that interval
    for (const p of P) { const b = Math.min(nb - 1, Math.floor(((p.conf - 0.5) / 0.5) * nb)); out[b].n++; out[b].acc += p.ok; out[b].conf += p.conf; }
  }
  out.forEach((b, i) => {
    if (b.n) { b.acc /= b.n; b.conf /= b.n; }
    if (scheme === 'mass') { b.lo = b.min ?? 0.5; b.hi = Math.max(b.max ?? 0.5, b.lo + 0.004); } // adaptive bins: actual confidence range
    else { b.lo = 0.5 + (0.5 * i) / nb; b.hi = 0.5 + (0.5 * (i + 1)) / nb; }
  });
  return out;
}
function metrics(P, nb = 10, scheme = 'width') {
  const n = P.length, B = bins(P, nb, scheme);
  const ece = B.reduce((a, b) => a + (b.n / n) * Math.abs(b.acc - b.conf), 0);
  const mce = Math.max(...B.filter((b) => b.n).map((b) => Math.abs(b.acc - b.conf)));
  const eps = 1e-12;
  const nll = -P.reduce((a, p) => a + Math.log(Math.max(eps, p.y ? p.p : 1 - p.p)), 0) / n;
  const brier = P.reduce((a, p) => a + (p.p - p.y) ** 2, 0) / n;
  const acc = P.reduce((a, p) => a + p.ok, 0) / n;
  const meanConf = P.reduce((a, p) => a + p.conf, 0) / n;
  return { acc, ece, mce, nll, brier, meanConf, B };
}
/** Fit T by golden-section search on log T minimizing NLL. */
function fitT(z, y) {
  const nll = (lt) => { const T = Math.exp(lt); let s = 0; for (let i = 0; i < z.length; i++) { const p = sigmoid(z[i] / T); s -= Math.log(Math.max(1e-12, y[i] ? p : 1 - p)); } return s / z.length; };
  let a = Math.log(0.05), b = Math.log(50); const g = (Math.sqrt(5) - 1) / 2;
  let c = b - g * (b - a), d = a + g * (b - a), fc = nll(c), fd = nll(d);
  for (let it = 0; it < 80; it++) { if (fc < fd) { b = d; d = c; fd = fc; c = b - g * (b - a); fc = nll(c); } else { a = c; c = d; fc = fd; d = a + g * (b - a); fd = nll(d); } }
  return Math.exp((a + b) / 2);
}
/** Risk–coverage curve, AURC and E-AURC (Geifman et al. 2019). */
function riskCoverage(P) {
  const s = P.slice().sort((a, b) => b.conf - a.conf);
  const n = s.length; let err = 0; const cov = [], risk = [];
  s.forEach((p, i) => { err += 1 - p.ok; cov.push((i + 1) / n); risk.push(err / (i + 1)); });
  const aurc = risk.reduce((a, r) => a + r, 0) / n;
  const E = n - P.reduce((a, p) => a + p.ok, 0); // oracle answers all correct first
  let oracle = 0; for (let i = 1; i <= n; i++) oracle += Math.max(0, i - (n - E)) / i;
  oracle /= n;
  return { cov, risk, aurc, eaurc: aurc - oracle };
}
function coverageAt(rc, target) {
  let best = 0; rc.risk.forEach((r, i) => { if (r <= target) best = rc.cov[i]; }); return best;
}

// ---------- data ----------
let D, Z = {}, split;
const state = window.__cal = {};

async function load() {
  const [data, lex] = await Promise.all([getJSON('./data/sst2.json'), lexModels()]);
  D = data;
  Z.bert = D.distilbert_logodds;
  Z.lr = D.sentence.map((s) => lex.lrSync(s));
  Z.nb = D.sentence.map((s) => lex.nbSync(s));
  // seeded 50/50 split: first half calibration, second half test
  const r = rng(2017), idx = [...D.label.keys()];
  for (let i = idx.length - 1; i > 0; i--) { const j = Math.floor(r() * (i + 1)); [idx[i], idx[j]] = [idx[j], idx[i]]; }
  split = { cal: idx.slice(0, idx.length >> 1), test: idx.slice(idx.length >> 1) };
}
const pick = (arr, ids) => ids.map((i) => arr[i]);

function compute() {
  const nb = +$('#bins').value, scheme = $('#scheme').value;
  const R = {};
  for (const k of KEYS) {
    const yc = pick(D.label, split.cal), yt = pick(D.label, split.test);
    const T = fitT(pick(Z[k], split.cal), yc);
    const before = metrics(preds(pick(Z[k], split.test), yt, 1), nb, scheme);
    const after = metrics(preds(pick(Z[k], split.test), yt, T), nb, scheme);
    const full = preds(Z[k], D.label, 1);
    R[k] = { T, before, after, rc: riskCoverage(full), fullAcc: full.reduce((a, p) => a + p.ok, 0) / full.length };
  }
  state.results = R;
  return R;
}

const f3 = (v) => v.toFixed(3), pct = (v) => (v * 100).toFixed(1) + '%';

async function render() {
  const R = compute(), k = $('#model').value, r = R[k];
  $('#big').replaceChildren(...[
    [pct(r.fullAcc), 'accuracy, all 872'],
    [f3(r.before.ece), 'ECE before (test split)'],
    [f3(r.after.ece), 'ECE after temperature scaling'],
    ['T = ' + r.T.toFixed(2), r.T > 1 ? 'fitted temperature (> 1: was overconfident)' : 'fitted temperature (< 1: was underconfident)'],
    [pct(coverageAt(r.rc, +$('#risk').value)), `max coverage with risk ≤ ${pct(+$('#risk').value)}`],
  ].map(([b, s]) => el('div', {}, [el('b', { text: b }), el('span', { text: s })])));

  $('#tbl').replaceChildren(
    el('thead', {}, el('tr', {}, ['Model', 'Accuracy', 'Mean confidence', 'ECE', 'MCE', 'NLL', 'Brier', 'T', 'ECE after', 'NLL after', 'Brier after'].map((h) => el('th', { text: h })))),
    el('tbody', {}, KEYS.map((m) => { const x = R[m]; return el('tr', { class: m === k ? 'sel' : '', 'data-model': m }, [
      el('td', { text: NAMES[m] }), el('td', { text: pct(x.before.acc) }), el('td', { text: pct(x.before.meanConf) }),
      el('td', { text: f3(x.before.ece) }), el('td', { text: f3(x.before.mce) }), el('td', { text: f3(x.before.nll) }), el('td', { text: f3(x.before.brier) }),
      el('td', { text: x.T.toFixed(2) }),
      el('td', { class: x.after.ece <= x.before.ece ? 'good' : 'bad', text: f3(x.after.ece) }),
      el('td', { class: x.after.nll <= x.before.nll ? 'good' : 'bad', text: f3(x.after.nll) }),
      el('td', { class: x.after.brier <= x.before.brier ? 'good' : 'bad', text: f3(x.after.brier) })]); })));

  const rt = +$('#risk').value;
  $('#seltbl').replaceChildren(
    el('thead', {}, el('tr', {}, ['Model (all 872)', 'Accuracy', 'AURC ↓', 'E-AURC ↓', `Coverage at risk ≤ ${pct(rt)}`].map((h) => el('th', { text: h })))),
    el('tbody', {}, KEYS.map((m) => el('tr', { class: m === k ? 'sel' : '' }, [el('td', { text: NAMES[m] }), el('td', { text: pct(R[m].fullAcc) }), el('td', { text: f3(R[m].rc.aurc) }), el('td', { text: f3(R[m].rc.eaurc) }), el('td', { text: pct(coverageAt(R[m].rc, rt)) })]))));

  const Plotly = await plotly();
  const rel = (M, title) => {
    const B = M.B, xs = B.map((b) => (b.lo + b.hi) / 2), w = B.map((b) => (b.hi - b.lo) * 0.92);
    const xAcc = xs, width = w;
    return {
      data: [
        { type: 'bar', x: xAcc, y: B.map((b) => (b.n ? b.acc : null)), width, name: 'accuracy', marker: { color: COLORS[0] }, customdata: B.map((b) => [b.n, b.conf]), hovertemplate: 'accuracy %{y:.3f}<br>mean confidence %{customdata[1]:.3f}<br>%{customdata[0]} examples<extra></extra>' },
        { type: 'bar', x: xAcc, y: B.map((b) => (b.n ? b.conf - b.acc : null)), base: B.map((b) => (b.n ? b.acc : null)), width, name: 'gap to confidence', marker: { color: 'rgba(217,140,122,.45)', line: { color: '#d98c7a', width: 1 } }, hoverinfo: 'skip' },
        { type: 'scatter', x: [0.5, 1], y: [0.5, 1], mode: 'lines', name: 'perfect calibration', line: { color: '#9aa3ad', dash: 'dot' } },
        { type: 'scatter', x: B.map((b) => b.conf), y: B.map((b) => b.n / M.B.reduce((a, c) => a + c.n, 0)), yaxis: 'y2', mode: 'lines+markers', name: 'share of examples', line: { color: COLORS[1], width: 1 }, marker: { size: 4 } },
      ],
      layout: plotLayout({ title: { text: title, font: { size: 14 } }, barmode: 'overlay', margin: { l: 48, r: 48, t: 36, b: 44 },
        xaxis: { ...plotLayout().xaxis, title: { text: 'confidence' }, range: [0.48, 1.02] }, yaxis: { ...plotLayout().yaxis, title: { text: 'accuracy' }, range: [0, 1.02] },
        yaxis2: { overlaying: 'y', side: 'right', range: [0, 1], showgrid: false, color: '#9aa3ad', tickfont: { size: 10 }, title: { text: 'share', font: { size: 11 } } },
        legend: { orientation: 'h', y: -0.28, font: { size: 11 } } }),
    };
  };
  const a = rel(r.before, `${NAMES[k]} · before · ECE ${f3(r.before.ece)}`), b = rel(r.after, `after, T = ${r.T.toFixed(2)} · ECE ${f3(r.after.ece)}`);
  Plotly.react('rel0', a.data, a.layout, PLOT_CONFIG);
  Plotly.react('rel1', b.data, b.layout, PLOT_CONFIG);
  Plotly.react('rc', KEYS.map((m, i) => ({ x: R[m].rc.cov, y: R[m].rc.risk, name: `${NAMES[m]} (AURC ${f3(R[m].rc.aurc)})`, mode: 'lines', line: { color: COLORS[i], width: m === k ? 3 : 1.5 } }))
    .concat([{ x: [0, 1], y: [rt, rt], mode: 'lines', name: `target risk ${pct(rt)}`, line: { color: '#9aa3ad', dash: 'dot' } }]),
  plotLayout({ margin: { l: 52, r: 16, t: 16, b: 44 }, xaxis: { ...plotLayout().xaxis, title: { text: 'coverage (fraction answered, most confident first)' }, range: [0, 1] }, yaxis: { ...plotLayout().yaxis, title: { text: 'risk (error rate on answered)' }, rangemode: 'tozero' }, legend: { orientation: 'h', y: -0.25 } }), PLOT_CONFIG);
  window.__lab.rendered = (window.__lab.rendered || 0) + 1;
}

let liveCtrl = null;
async function live() {
  if (liveCtrl) return;
  liveCtrl = new AbortController(); $('#live').disabled = true; $('#livecancel').disabled = false;
  const out = $('#liveout'); out.textContent = '';
  try {
    const m = await distilbert(progressUI($('#liveload'), 'DistilBERT'));
    const n = +$('#liven').value, r = rng(Date.now() % 100000), ids = [...D.label.keys()];
    for (let i = ids.length - 1; i > 0; i--) { const j = Math.floor(r() * (i + 1)); [ids[i], ids[j]] = [ids[j], ids[i]]; }
    const sel = ids.slice(0, n), t0 = performance.now();
    const z = await m.score(sel.map((i) => D.sentence[i]), { signal: liveCtrl.signal, onProgress: (d, t) => { out.textContent = `Scoring ${d}/${t}…`; } });
    const diff = Math.max(...z.map((v, j) => Math.abs(v - Z.bert[sel[j]])));
    const agree = z.filter((v, j) => (v >= 0) === (Z.bert[sel[j]] >= 0)).length;
    out.textContent = `${n} sentences in ${((performance.now() - t0) / 1000).toFixed(1)} s · predictions agree with the precomputed run on ${agree}/${n} · max |Δ log-odds| = ${diff.toExponential(1)}. Sentences are scored one at a time: this 8-bit model quantizes activations with per-batch scales, so batching sentences together would make each score depend on its batch-mates.`;
    window.__lab.live = { n, agree, diff };
  } catch (e) {
    out.textContent = e.name === 'AbortError' ? 'Cancelled.' : 'Couldn’t run the model: ' + (e.message || e);
  } finally { liveCtrl = null; $('#live').disabled = false; $('#livecancel').disabled = true; }
}

async function init() {
  try {
    $('#status').textContent = 'Loading 872 sentences and scoring the two classic models…';
    await load();
    $('#status').textContent = `${D.sentence.length} sentences (${D.label.filter((v) => v).length} positive) · calibration split ${split.cal.length}, test split ${split.test.length} · ${D.source}.`;
    for (const id of ['#model', '#bins', '#scheme', '#risk']) $(id).addEventListener('change', render);
    $('#live').addEventListener('click', live);
    $('#livecancel').addEventListener('click', () => liveCtrl && liveCtrl.abort());
    await render();
    state.metrics = metrics; state.preds = preds; state.fitT = fitT;
    ready({ calibration: true });
  } catch (e) { fail('Couldn’t load the evaluation data: ' + (e.message || e)); }
}
init();
