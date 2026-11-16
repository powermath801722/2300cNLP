import { el, $, ready, fail, plotly, plotLayout, PLOT_CONFIG, COLORS } from '/lab/shared/lab.js';
import { progressUI } from '/lab/shared/hf.js';
import { lexModels, distilbert, sigmoid, BERT_MB } from './models.js';
import { segment } from './text.js';
import { objective, occlusion, lime, kernelShap, faithfulness, spearman, topkOverlap } from './explain.js';

const SAMPLES = [
  ['A warm, funny and quietly brilliant film, though the ending drags.', 'Mixed'],
  ['The plot was not bad, but the acting was dreadful.', 'Negation + contrast'],
  ['Good, good, good: the cast is good and the music is good.', 'Redundancy'],
  ['I expected a disaster and got a gem.', 'Expectation'],
  ['Visually stunning yet emotionally empty.', 'Short, two-sided'],
];
const METHODS = [['occ', 'Occlusion'], ['lime', 'LIME'], ['shap', 'KernelSHAP']];
const MAX_WORDS = 30;

let lex, bert, ctrl = null;
const state = window.__attr = { last: null };

function shade(v, scale) {
  if (!v || !scale) return '';
  const a = Math.min(Math.abs(v) / scale, 1) * 0.6 + 0.06;
  return v > 0 ? `rgba(143,191,154,${a.toFixed(2)})` : `rgba(217,140,122,${a.toFixed(2)})`;
}
function heatRow(name, sub, parts, W, attr) {
  const scale = Math.max(...attr.map(Math.abs)) || 1;
  const pos = new Map(W.map((i, j) => [i, j]));
  const p = el('p', { class: 'words' });
  parts.forEach((pt, i) => {
    if (i) p.append(' ');
    const j = pos.get(i);
    if (j == null) { p.append(pt.t); return; }
    p.append(el('span', { class: 'w', tabindex: '0', title: `${pt.t}: ${attr[j] >= 0 ? '+' : ''}${attr[j].toFixed(3)}`, 'aria-label': `${pt.t} ${attr[j].toFixed(3)}`, style: { background: shade(attr[j], scale) }, text: pt.t }));
  });
  return el('div', { class: 'hrow', 'data-method': name }, [el('b', {}, [name, el('small', { text: sub })]), p]);
}

async function scorer(kind) {
  if (kind === 'bert') {
    if (!bert) bert = await distilbert(progressUI($('#bertload'), 'DistilBERT'));
    return bert.score;
  }
  return kind === 'nb' ? lex.nb : lex.lr;
}

function exactFor(kind, parts, W) {
  if (kind === 'nb') return W.map((i) => lex.nbWord(parts[i].t));
  if (kind === 'lr') {
    const cnt = new Map(); for (const i of W) { const k = parts[i].t.toLowerCase(); cnt.set(k, (cnt.get(k) || 0) + 1); }
    return W.map((i) => lex.lrWord(parts[i].t) / cnt.get(parts[i].t.toLowerCase()));
  }
  return null;
}

async function explainAll(kind, text, { samples, seed, signal, status }) {
  const parts = segment(text);
  const W = parts.map((p, i) => (p.word ? i : -1)).filter((i) => i >= 0);
  if (!W.length) throw new Error('Type a sentence with at least one word.');
  if (W.length > MAX_WORDS) throw new Error(`Please keep it to ${MAX_WORDS} words or fewer (this has ${W.length}); the cost grows with sentence length.`);
  const score = await scorer(kind);
  const prog = (what) => (d, n) => status(`${what}: ${d}/${n} model evaluations…`);
  const timed = async (what, fn) => {
    const obj = objective(parts, score, { signal, onProgress: prog(what) });
    const t0 = performance.now(); const r = await fn(obj); return { r, obj, ms: performance.now() - t0 };
  };
  const res = { kind, text, parts, W, M: W.length, attr: {}, cost: {} };
  status('Occlusion…');
  let t = await timed('Occlusion', occlusion); res.attr.occ = t.r; res.cost.occ = [t.obj.queries, t.ms];
  const base = t.obj; // reuse its cache for curves
  status('LIME…');
  t = await timed('LIME', (o) => lime(o, { samples, seed })); res.attr.lime = t.r; res.cost.lime = [t.obj.queries, t.ms];
  status('KernelSHAP…');
  t = await timed('KernelSHAP', (o) => kernelShap(o, { samples, seed })); res.attr.shap = t.r.phi; res.shapExact = t.r.exact; res.cost.shap = [t.obj.queries, t.ms];
  const [z] = await base.f([new Uint8Array(res.M).fill(1)]);
  res.z = z; res.p = sigmoid(z); res.label = z >= 0 ? 'pos' : 'neg';
  res.exact = exactFor(kind, parts, W);

  // Faithfulness: rank words toward the predicted class.
  const sgn = z >= 0 ? 1 : -1, toProb = (v) => sigmoid(sgn * v);
  status('Faithfulness curves…');
  res.faith = {};
  const order = (a) => [...a.keys()].sort((x, y) => sgn * a[y] - sgn * a[x]);
  for (const [m] of METHODS) res.faith[m] = await faithfulness(base, order(res.attr[m]), toProb);
  if (res.exact) res.faith.exact = await faithfulness(base, order(res.exact), toProb);
  const rnd = (() => { let s = seed * 7919; return () => ((s = (s * 1103515245 + 12345) % 2147483648) / 2147483648); })();
  const rand = [];
  for (let r = 0; r < 5; r++) { const o = [...Array(res.M).keys()]; for (let i = o.length - 1; i > 0; i--) { const j = Math.floor(rnd() * (i + 1)); [o[i], o[j]] = [o[j], o[i]]; } rand.push(await faithfulness(base, o, toProb)); }
  const avg = (k) => rand[0][k].map((_, i) => rand.reduce((a, r) => a + r[k][i], 0) / rand.length);
  res.faith.random = { del: avg('del'), ins: avg('ins'), aopc: rand.reduce((a, r) => a + r.aopc, 0) / 5, delAUC: rand.reduce((a, r) => a + r.delAUC, 0) / 5, insAUC: rand.reduce((a, r) => a + r.insAUC, 0) / 5 };

  // Sanity checks
  status('Sanity checks…');
  res.sanity = {};
  if (kind === 'lr') {
    const R = lex.lrRandomized(seed);
    const o = objective(parts, R.score);
    const rOcc = await occlusion(o), rLime = await lime(o, { samples, seed }), rShap = (await kernelShap(o, { samples, seed })).phi;
    res.sanity.randomized = { occ: spearman(res.attr.occ, rOcc), lime: spearman(res.attr.lime, rLime), shap: spearman(res.attr.shap, rShap) };
  }
  const o2 = objective(parts, score, { signal });
  res.sanity.stability = { lime: spearman(res.attr.lime, await lime(o2, { samples, seed: seed + 1 })), shap: spearman(res.attr.shap, (await kernelShap(o2, { samples, seed: seed + 1 })).phi) };
  res.sanity.stabilityQueries = o2.queries;
  return res;
}

function render(res) {
  $('#results').hidden = false;
  const { parts, W } = res;
  const name = { lr: 'Logistic regression', nb: 'Naive Bayes', bert: 'DistilBERT (SST-2)' }[res.kind];
  $('#pred').replaceChildren(
    el('span', {}, [name + ' says ', el('span', { class: 'lab ' + res.label, text: res.label === 'pos' ? 'positive' : 'negative' })]),
    el('span', { text: `P(positive) = ${res.p.toFixed(3)} · log-odds f(x) = ${res.z.toFixed(3)}` }),
    el('span', { text: `${res.M} words` }));
  const rows = [];
  if (res.exact) rows.push(heatRow('Exact', res.kind === 'nb' ? 'additive model: true contributions' : 'Shapley values of the linear model', parts, W, res.exact));
  rows.push(heatRow('Occlusion', 'leave one word out', parts, W, res.attr.occ));
  rows.push(heatRow('LIME', `${$('#budget').value} samples, ridge`, parts, W, res.attr.lime));
  rows.push(heatRow('KernelSHAP', res.shapExact ? 'all coalitions enumerated: exact' : `${$('#budget').value} sampled coalitions`, parts, W, res.attr.shap));
  $('#heat').replaceChildren(...rows);

  // agreement
  const keys = [...(res.exact ? [['exact', 'Exact']] : []), ...METHODS];
  const A = { exact: res.exact, ...res.attr };
  const t = $('#agree'); t.replaceChildren();
  t.append(el('thead', {}, el('tr', {}, [el('th', { text: '' }), ...keys.map(([, n]) => el('th', { text: n }))])));
  const tb = el('tbody');
  for (const [a, na] of keys) tb.append(el('tr', {}, [el('th', { text: na }), ...keys.map(([b]) => el('td', { text: a === b ? '—' : `ρ ${spearman(A[a], A[b]).toFixed(2)} · top-3 ${(topkOverlap(A[a], A[b], 3) * 100).toFixed(0)}%` }))]));
  t.append(tb);

  // cost
  $('#cost').replaceChildren(el('thead', {}, el('tr', {}, ['Method', 'Model calls', 'Time'].map((h) => el('th', { text: h })))),
    el('tbody', {}, METHODS.map(([m, n]) => el('tr', {}, [el('td', { text: n }), el('td', { text: String(res.cost[m][0]) }), el('td', { text: `${(res.cost[m][1] / 1000).toFixed(2)} s` })]))));

  // faithfulness table + plots
  const fk = [...METHODS, ...(res.exact ? [['exact', 'Exact']] : []), ['random', 'Random order (mean of 5)']];
  $('#faith').replaceChildren(el('thead', {}, el('tr', {}, ['Ranking', 'AOPC (deletion) ↑', 'Deletion AUC ↓', 'Insertion AUC ↑'].map((h) => el('th', { text: h })))),
    el('tbody', {}, fk.map(([k, n]) => { const f = res.faith[k]; return el('tr', { 'data-k': k }, [el('td', { text: n }), el('td', { text: f.aopc.toFixed(3) }), el('td', { text: f.delAUC.toFixed(3) }), el('td', { text: f.insAUC.toFixed(3) })]); })));
  plotly().then((Plotly) => {
    const xs = [...Array(res.M + 1).keys()];
    const tr = (key) => fk.map(([k, n], i) => ({ x: xs, y: res.faith[k][key], name: n, mode: 'lines+markers', line: { color: k === 'random' ? '#9aa3ad' : k === 'exact' ? '#e4e7eb' : COLORS[i], dash: k === 'random' ? 'dot' : 'solid', width: 2 }, marker: { size: 5 } }));
    const lay = (title, x) => plotLayout({ title: { text: title, font: { size: 14 } }, margin: { l: 48, r: 12, t: 36, b: 44 }, xaxis: { ...plotLayout().xaxis, title: { text: x } }, yaxis: { ...plotLayout().yaxis, title: { text: 'P(predicted class)' }, range: [0, 1] }, legend: { orientation: 'h', y: -0.3, font: { size: 11 } }, showlegend: true });
    Plotly.react('pdel', tr('del'), lay('Deletion (lower is better)', 'words removed'), PLOT_CONFIG);
    Plotly.react('pins', tr('ins'), lay('Insertion (higher is better)', 'words added back'), PLOT_CONFIG);
  });

  // sanity
  const san = [];
  if (res.exact) {
    const r = METHODS.map(([m, n]) => { const err = Math.max(...res.attr[m].map((v, i) => Math.abs(v - res.exact[i]))); return el('tr', {}, [el('td', { text: n }), el('td', { text: spearman(res.attr[m], res.exact).toFixed(3) }), el('td', { text: err.toFixed(3) })]); });
    san.push(el('p', { class: 'note', text: 'Against the exact attributions of this linear model:' }), el('div', { class: 'scroll-x' }, el('table', { class: 'data', id: 'vsexact' }, [el('thead', {}, el('tr', {}, ['Method', 'Spearman ρ with exact', 'Max |error|'].map((h) => el('th', { text: h })))), el('tbody', {}, r)])));
  }
  if (res.sanity.randomized) {
    const s = res.sanity.randomized;
    san.push(el('p', { class: 'note', text: 'Model randomization (Adebayo et al. 2018): logistic-regression weights randomly permuted. A method that depends on the model should produce an unrelated ranking (ρ near 0):' }),
      el('div', { class: 'scroll-x' }, el('table', { class: 'data', id: 'randtbl' }, [el('thead', {}, el('tr', {}, ['Method', 'ρ(original, randomized model)'].map((h) => el('th', { text: h })))), el('tbody', {}, METHODS.map(([m, n]) => el('tr', {}, [el('td', { text: n }), el('td', { class: Math.abs(s[m]) < 0.5 ? 'good' : 'bad', text: s[m].toFixed(3) })])))])));
  }
  const st = res.sanity.stability;
  san.push(el('p', { class: 'note', text: `Estimator stability: rerunning the sampling methods with seed ${+$('#seed').value + 1} gives rank correlations LIME ρ = ${st.lime.toFixed(3)}, KernelSHAP ρ = ${st.shap.toFixed(3)} (occlusion is deterministic). Low values mean the budget is too small for this sentence.` }));
  if (res.kind === 'bert') san.push(el('p', { class: 'note', text: 'Randomizing DistilBERT’s 66 M weights isn’t practical in the browser, so the model-randomization check runs on the logistic regression only.' }));
  $('#sanity').replaceChildren(...san);

  // table
  const head = ['Word', ...(res.exact ? ['Exact'] : []), 'Occlusion', 'LIME', 'KernelSHAP'];
  $('#tbl').replaceChildren(el('thead', {}, el('tr', {}, head.map((h) => el('th', { text: h })))),
    el('tbody', {}, W.map((i, j) => el('tr', {}, [el('td', { text: parts[i].t }), ...(res.exact ? [res.exact[j]] : []).concat([res.attr.occ[j], res.attr.lime[j], res.attr.shap[j]]).map((v) => el('td', { text: (v >= 0 ? '+' : '') + v.toFixed(3) }))]))));
}

async function run() {
  if (ctrl) return;
  const kind = $('#model').value, text = $('#txt').value.trim();
  const samples = +$('#budget').value, seed = Math.max(1, +$('#seed').value || 1);
  $('#msg').hidden = true; ctrl = new AbortController();
  $('#go').disabled = true; $('#cancel').disabled = false;
  const status = (s) => { $('#status').textContent = s; };
  const t0 = performance.now();
  try {
    const res = await explainAll(kind, text, { samples, seed, signal: ctrl.signal, status });
    state.last = res;
    render(res);
    status(`Done in ${((performance.now() - t0) / 1000).toFixed(1)} s.`);
    window.__lab.runs = (window.__lab.runs || 0) + 1;
  } catch (e) {
    if (e.name === 'AbortError') status('Cancelled.');
    else { $('#msg').textContent = e.message || String(e); $('#msg').hidden = false; status(''); }
  } finally { ctrl = null; $('#go').disabled = false; $('#cancel').disabled = true; }
}

async function init() {
  try {
    lex = await lexModels();
    $('#txt').value = SAMPLES[0][0];
    $('#samples').append(...SAMPLES.map(([s, tag]) => el('button', { class: 'chip', type: 'button', title: s, text: tag, onclick: () => { $('#txt').value = s; run(); } })));
    $('#go').addEventListener('click', run);
    $('#cancel').addEventListener('click', () => ctrl && ctrl.abort());
    $('#model').addEventListener('change', () => { if ($('#model').value === 'bert' && !bert) $('#status').textContent = `DistilBERT downloads once (≈${BERT_MB} MB) the first time you click Explain.`; });
    state.explainAll = explainAll;
    ready({ attribution: true });
    run();
  } catch (e) { fail('Couldn’t load the sentiment models: ' + e.message); }
}
init();
