import { el, $, ready, fail, getJSON, plotly, plotLayout, PLOT_CONFIG } from '/lab/shared/lab.js';

// Hand-written labels for the trained K=15 model (training/topics_train.py, random_state=7).
const LABELS = [
  'Currencies & FX', 'Banks & interest rates', 'Quarterly results tables', 'Executives & steel', 'Analysts & debt markets',
  'Earnings outlook', 'Divestitures & units', 'Trade balance & economy', 'Mergers & tender offers', 'UK money markets & labor',
  'Sugar & monthly exports', 'Oil & energy', 'Grain & agriculture', 'Trade diplomacy', 'Gold, mining & cocoa',
];
const PALETTE = ['#7ea3c8', '#d8b66a', '#8fbf9a', '#d98c7a', '#b49bd6', '#6cc3c3', '#e0a35c', '#9cb4d8', '#c3d07a', '#e08fb0', '#7fc7a5', '#cfa0e0', '#e6c87a', '#8ab8c9', '#c79a86'];
const EXAMPLES = [
  ['Oil', 'Crude oil prices climbed on Monday after OPEC ministers agreed to cut production, and analysts expect gasoline and heating oil to follow as refiners pay more per barrel.'],
  ['Grain', 'The agriculture department raised its estimate of the winter wheat crop, while corn and soybean farmers waited for export subsidies to be approved by Congress.'],
  ['Merger', 'The board recommended that shareholders accept the tender offer of 45 dollars a share, and the merger agreement lets the buyer acquire the remaining common stock.'],
  ['Mixed', 'The central bank cut interest rates as the dollar fell against the yen, and the trade deficit with Japan widened despite talks between officials.'],
];

let D, K, V, vidx, stop, re, sel = 0;

const prevalence = () => { const s = new Array(K).fill(0); for (const d of D.docs) d.theta.forEach((p, k) => (s[k] += p)); return s.map((x) => x / D.docs.length); };

function topicCards() {
  const host = $('#topics'); host.innerHTML = '';
  D.top.forEach((words, k) => {
    const cv = D.coherence_per_topic[k], max = words[0][1];
    host.append(el('button', { class: 'tcard', type: 'button', 'aria-pressed': String(k === sel), 'data-k': k, onclick: () => select(k) }, [
      el('h3', {}, [el('span', { class: 'sw', style: { background: PALETTE[k] } }), `${k + 1}. ${LABELS[k]}`]),
      el('span', { class: 'meta', html: `C<sub>v</sub> ${cv.toFixed(2)}${cv < 0.45 ? ' · <span class="weak">mixed</span>' : ''}` }),
      el('div', { class: 'wbars' }, words.slice(0, 10).map(([w, p]) => el('div', { title: `P(${w} | topic ${k + 1}) = ${p.toFixed(4)}` }, [el('span', { text: w }), el('i', { style: { width: (p / max * 100).toFixed(1) + '%', background: PALETTE[k] } })]))),
    ]));
  });
}

function select(k) {
  sel = k;
  document.querySelectorAll('.tcard').forEach((b) => b.setAttribute('aria-pressed', String(+b.dataset.k === k)));
  $('#seltitle').textContent = `Documents using topic ${k + 1} (${LABELS[k]}) most`;
  const docs = D.docs.map((d, i) => [d.theta[k], i]).sort((a, b) => b[0] - a[0]).slice(0, 4);
  $('#topdocs').replaceChildren(...docs.map(([p, i]) => el('div', { class: 'doc' }, [el('b', { text: `${Math.round(p * 100)}% · ${title(D.docs[i].title)}` }), D.docs[i].snippet.slice(0, 140) + '…'])));
  drawMap();
}
const title = (t) => t.replace(/&lt;/g, '<').replace(/&gt;/g, '>').replace(/\s+/g, ' ');

function mixEls(theta, mixHost, listHost, n = 5) {
  mixHost.replaceChildren(...theta.map((p, k) => el('i', { title: `${LABELS[k]} ${(p * 100).toFixed(1)}%`, style: { width: (p * 100) + '%', background: PALETTE[k] } })));
  const top = theta.map((p, k) => [p, k]).sort((a, b) => b[0] - a[0]).slice(0, n);
  mixHost.setAttribute('aria-label', 'Topic mixture: ' + top.map(([p, k]) => `${LABELS[k]} ${Math.round(p * 100)}%`).join(', '));
  listHost.replaceChildren(...top.map(([p, k]) => el('div', {}, [el('div', {}, [el('span', { text: `${k + 1}. ${LABELS[k]}` }), el('div', { class: 'bar' }, el('i', { style: { width: (p * 100) + '%', background: PALETTE[k] } }))]), el('span', { text: (p * 100).toFixed(1) + '%' })])));
}

function showDoc(i) {
  const d = D.docs[i];
  $('#docbody').replaceChildren(el('b', { text: title(d.title) }), d.snippet + '…', el('br'), el('small', { class: 'note', text: `Reuters-21578 ${d.id}` }));
  mixEls(d.theta, $('#docmix'), $('#doclist'));
}

// ---- Jensen–Shannon + classical MDS for the topic map ----
function jsd(p, q) {
  let s = 0;
  for (let i = 0; i < p.length; i++) { const m = (p[i] + q[i]) / 2; if (p[i] > 0) s += p[i] * Math.log(p[i] / m); if (q[i] > 0) s += q[i] * Math.log(q[i] / m); }
  return s / 2;
}
function mds2(D2) {
  const n = D2.length, J = (i, j) => D2[i][j];
  const rm = D2.map((r) => r.reduce((a, b) => a + b, 0) / n), gm = rm.reduce((a, b) => a + b, 0) / n;
  const B = D2.map((r, i) => r.map((_, j) => -0.5 * (J(i, j) - rm[i] - rm[j] + gm)));
  const eig = [];
  let M = B.map((r) => r.slice());
  for (let c = 0; c < 2; c++) {
    let v = Array.from({ length: n }, (_, i) => Math.sin(i + 1 + c)), lam = 0;
    for (let it = 0; it < 300; it++) {
      const w = M.map((r) => r.reduce((s, x, j) => s + x * v[j], 0));
      const norm = Math.hypot(...w) || 1; lam = norm; v = w.map((x) => x / norm);
    }
    eig.push(v.map((x) => x * Math.sqrt(Math.max(lam, 0))));
    M = M.map((r, i) => r.map((x, j) => x - lam * v[i] * v[j]));
  }
  return eig;
}
let mapXY, prev;
async function drawMap() {
  const P = await plotly();
  const size = prev.map((p) => 18 + Math.sqrt(p) * 90);
  P.react('map', [{
    type: 'scatter', mode: 'markers+text', x: mapXY[0], y: mapXY[1], text: LABELS.map((_, k) => String(k + 1)),
    textfont: { family: 'Barlow', size: 13, color: '#121820' },
    customdata: LABELS.map((l, k) => [l, (prev[k] * 100).toFixed(1), D.top[k].slice(0, 5).map((w) => w[0]).join(', ')]),
    hovertemplate: '<b>%{customdata[0]}</b><br>%{customdata[1]}% of sample<br>%{customdata[2]}<extra></extra>',
    marker: { size, color: PALETTE, opacity: LABELS.map((_, k) => (k === sel ? 1 : 0.6)), line: { color: LABELS.map((_, k) => (k === sel ? '#ffffff' : 'rgba(0,0,0,0)')), width: 2 } },
  }], plotLayout({ margin: { l: 20, r: 20, t: 20, b: 20 }, xaxis: { visible: false }, yaxis: { visible: false, scaleanchor: 'x' }, showlegend: false }), PLOT_CONFIG);
  const g = document.getElementById('map');
  if (!g._bound) { g._bound = true; g.on('plotly_click', (e) => select(e.points[0].pointIndex)); }
}
async function drawSweep() {
  const P = await plotly();
  P.react('sweep', [{ type: 'scatter', mode: 'lines+markers', x: D.sweep.map((s) => s.k), y: D.sweep.map((s) => s.c_v), line: { color: '#7ea3c8' },
    marker: { size: D.sweep.map((s) => (s.k === K ? 16 : 9)), color: D.sweep.map((s) => (s.k === K ? '#d8b66a' : '#7ea3c8')) }, hovertemplate: 'K = %{x}<br>C_v = %{y:.3f}<extra></extra>' }],
  plotLayout({ xaxis: { title: { text: 'Number of topics K' }, gridcolor: 'rgba(228,231,235,.08)', color: '#9aa3ad' }, yaxis: { title: { text: 'Coherence C_v' }, gridcolor: 'rgba(228,231,235,.08)', color: '#9aa3ad' },
    annotations: [{ x: K, y: D.sweep.find((s) => s.k === K).c_v, text: 'chosen', showarrow: true, arrowcolor: '#d8b66a', font: { color: '#d8b66a' }, ay: 30 }] }), PLOT_CONFIG);
}

// ---- collapsed Gibbs inference with fixed phi ----
function rng(seed) { return () => { seed |= 0; seed = (seed + 0x6d2b79f5) | 0; let t = Math.imul(seed ^ (seed >>> 15), 1 | seed); t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t; return ((t ^ (t >>> 14)) >>> 0) / 4294967296; }; }
function tokenize(text) {
  re.lastIndex = 0;
  return (text.toLowerCase().match(re) || []).filter((w) => !stop.has(w));
}
function infer(text, sweeps = 300) {
  const toks = tokenize(text), ids = toks.map((w) => vidx.get(w)).filter((i) => i != null);
  if (!ids.length) return { theta: null, known: 0, total: toks.length };
  const r = rng(12345), n = new Float64Array(K), z = new Int32Array(ids.length), p = new Float64Array(K), acc = new Float64Array(K), zc = ids.map(() => new Float64Array(K));
  ids.forEach((w, i) => { z[i] = Math.floor(r() * K); n[z[i]]++; });
  const burn = Math.floor(sweeps / 2);
  for (let it = 0; it < sweeps; it++) {
    for (let i = 0; i < ids.length; i++) {
      n[z[i]]--;
      let s = 0;
      for (let k = 0; k < K; k++) { p[k] = (n[k] + D.alpha[k]) * D.phi[k][ids[i]]; s += p[k]; }
      let u = r() * s, k = 0; while ((u -= p[k]) > 0 && k < K - 1) k++;
      z[i] = k; n[k]++;
      if (it >= burn) zc[i][k]++;
    }
    if (it >= burn) { const tot = ids.length + D.alpha.reduce((a, b) => a + b, 0); for (let k = 0; k < K; k++) acc[k] += (n[k] + D.alpha[k]) / tot; }
  }
  const theta = Array.from(acc, (x) => x / (sweeps - burn));
  const zbest = zc.map((c) => c.indexOf(Math.max(...c)));
  return { theta, known: ids.length, total: toks.length, words: ids.map((w, i) => [D.vocab[w], zbest[i]]) };
}
function runInfer() {
  const t0 = performance.now();
  const r = infer($('#txt').value, +$('#iters').value);
  if (!r.theta) { $('#infnote').textContent = 'None of these words are in the 4,000-word vocabulary. Try news-style text.'; $('#infmix').replaceChildren(); $('#inflist').replaceChildren(); $('#infwords').replaceChildren(); return null; }
  mixEls(r.theta, $('#infmix'), $('#inflist'));
  $('#infwords').replaceChildren(...r.words.flatMap(([w, k], i) => [i ? ' ' : '', el('span', { title: `${w} → ${LABELS[k]}`, style: { background: PALETTE[k] }, text: w })]));
  $('#infnote').textContent = `${r.known} of ${r.total} content words are in the vocabulary · ${$('#iters').value} sweeps in ${(performance.now() - t0).toFixed(0)} ms.`;
  return r;
}

async function init() {
  try { D = await getJSON('./data/lda.json'); } catch (e) { fail(e.message); return; }
  K = D.k; V = D.vocab.length; vidx = new Map(D.vocab.map((w, i) => [w, i])); stop = new Set(D.stopwords); re = new RegExp(D.token_re, 'g');
  prev = prevalence();
  $('#status').textContent = `${D.meta.n_docs.toLocaleString()} documents · ${V.toLocaleString()}-word vocabulary · K = ${K}.`;
  topicCards();
  const Dm = D.phi.map((p) => D.phi.map((q) => jsd(p, q)));
  mapXY = mds2(Dm.map((r) => r.map((x) => x * x)));
  $('#docsel').append(...D.docs.map((d, i) => el('option', { value: i, text: title(d.title) })));
  $('#docsel').addEventListener('change', (e) => showDoc(+e.target.value));
  showDoc(0);
  $('#chips').append(...EXAMPLES.map(([l, t]) => el('button', { class: 'chip', type: 'button', text: l, onclick: () => { $('#txt').value = t; runInfer(); } })));
  $('#txt').value = EXAMPLES[3][1];
  $('#infer').addEventListener('click', runInfer); $('#infer').disabled = false;
  runInfer();
  select(0);
  drawSweep();
  ready({ topics: true, infer: (t, s) => infer(t, s), labels: LABELS });
}
init();
