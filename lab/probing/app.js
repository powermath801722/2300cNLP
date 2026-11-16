import { el, $, ready, fail, getJSON, plotly, plotLayout, PLOT_CONFIG, COLORS } from '/lab/shared/lab.js';
import { loadModel, progressUI, transformers as T } from '/lab/shared/hf.js';

const ENC = 'Xenova/all-MiniLM-L6-v2';
const DIM = 384;
let D, nTrain, tokTrain = [], tokTest = [], emb = null, rows = {}, preds = null, sIdx = 0;

// ---------- data helpers ----------
function rng(seed) { let s = seed >>> 0 || 1; return () => { s ^= s << 13; s >>>= 0; s ^= s >>> 17; s ^= s << 5; s >>>= 0; return s / 4294967296; }; }
function gauss(r) { let u = 0; while (!u) u = r(); return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * r()); }
const lower = (w) => w.toLowerCase();

function flatten() {
  // token list: { s (sentence), i (index), w, t }
  D.sentences.forEach((s, si) => s.w.forEach((w, i) => (si < nTrain ? tokTrain : tokTest).push({ s: si, i, w, t: s.t[i] })));
}

// Control task (Hewitt & Liang 2019): each word type gets a fixed random tag drawn from the train tag distribution.
function controlLabels() {
  const r = rng(42), counts = new Array(D.tags.length).fill(0);
  for (const t of tokTrain) counts[t.t]++;
  const cum = []; let c = 0; for (const x of counts) { c += x / tokTrain.length; cum.push(c); }
  const map = new Map();
  const lab = (w) => { const k = lower(w); if (!map.has(k)) { const u = r(); map.set(k, cum.findIndex((x) => u <= x)); } return map.get(k); };
  return { train: Int32Array.from(tokTrain, (t) => lab(t.w)), test: Int32Array.from(tokTest, (t) => lab(t.w)) };
}

// Random fixed vector per word type: word identity and nothing else.
function randomVectors(toks) {
  const r = rng(7), map = new Map(), X = new Float32Array(toks.length * DIM);
  toks.forEach((t, n) => { const k = lower(t.w); if (!map.has(k)) map.set(k, Float32Array.from({ length: DIM }, () => gauss(r))); X.set(map.get(k), n * DIM); });
  return X;
}

// ---------- embedding ----------
const ui = () => (ui.v ||= progressUI($('#load'), 'MiniLM'));
async function embedAll() {
  $('#embed').disabled = true;
  const { tokenizer, model } = await loadModel(ENC, 'base', { ui: ui(), readyText: 'Ready · all-MiniLM-L6-v2 (q8, 23 MB, cached in your browser)' });
  const cls = 101, sep = 102;
  const out = [];
  const t0 = performance.now();
  for (let si = 0; si < D.sentences.length; si++) {
    const words = D.sentences[si].w;
    const pieces = words.map((w) => Array.from(tokenizer(w, { add_special_tokens: false }).input_ids.data, Number));
    const ids = [cls, ...pieces.flat(), sep], n = ids.length;
    const mk = (a) => new T.Tensor('int64', BigInt64Array.from(a, (x) => BigInt(x)), [1, n]);
    const o = await model({ input_ids: mk(ids), attention_mask: mk(ids.map(() => 1)), token_type_ids: mk(ids.map(() => 0)) });
    const H = o.last_hidden_state.data;
    let p = 1; const vecs = [];
    for (const pc of pieces) {
      const v = new Float32Array(DIM), m = Math.max(pc.length, 1);
      for (let q = 0; q < pc.length; q++) for (let d = 0; d < DIM; d++) v[d] += H[(p + q) * DIM + d] / m;
      vecs.push(v); p += pc.length;
    }
    out.push(vecs);
    if (si % 10 === 9 || si === D.sentences.length - 1) {
      const el2 = (performance.now() - t0) / 1000;
      $('#prog').textContent = `Embedded ${si + 1} / ${D.sentences.length} sentences (${el2.toFixed(0)} s)`;
      await new Promise((r) => setTimeout(r, 0));
    }
  }
  const pack = (toks) => { const X = new Float32Array(toks.length * DIM); toks.forEach((t, n) => X.set(out[t.s][t.i], n * DIM)); return X; };
  emb = { train: pack(tokTrain), test: pack(tokTest) };
  $('#prog').textContent = `Embedded all ${D.sentences.length} sentences in ${((performance.now() - t0) / 1000).toFixed(1)} s. ${tokTrain.length.toLocaleString('en-US')} training and ${tokTest.length.toLocaleString('en-US')} test words.`;
  $('#train').disabled = false;
  window.__lab.embedded = true;
}

// ---------- probes ----------
// A small pool of workers so independent probes train in parallel.
let pool = null, jobId = 0, rr = 0; const pending = new Map();
function runProbe(args) {
  if (!pool) {
    const n = Math.max(1, Math.min(4, (navigator.hardwareConcurrency || 2) - 1));
    pool = Array.from({ length: n }, () => { const w = new Worker(new URL('./worker.js', import.meta.url)); w.onmessage = (e) => { const p = pending.get(e.data.id); pending.delete(e.data.id); e.data.error ? p.reject(new Error(e.data.error)) : p.resolve(e.data); }; return w; });
  }
  const id = ++jobId, w = pool[rr++ % pool.length];
  return new Promise((resolve, reject) => {
    pending.set(id, { resolve, reject });
    w.postMessage({ id, d: DIM, k: D.tags.length, epochs: 30, lr: 0.01, l2: +$('#l2').value, seed: 3, ...args });
  });
}
const yTrain = () => Int32Array.from(tokTrain, (t) => t.t), yTest = () => Int32Array.from(tokTest, (t) => t.t);
const sliceTrain = (X, maxSent) => { const idx = tokTrain.map((t, n) => (t.s < maxSent ? n : -1)).filter((n) => n >= 0); const A = new Float32Array(idx.length * DIM); idx.forEach((n, k) => A.set(X.subarray(n * DIM, (n + 1) * DIM), k * DIM)); return { X: A, idx }; };

function baselines() {
  const yt = yTest(), counts = new Map(), byWord = new Map();
  for (const t of tokTrain) { counts.set(t.t, (counts.get(t.t) || 0) + 1); const k = lower(t.w); if (!byWord.has(k)) byWord.set(k, new Map()); const m = byWord.get(k); m.set(t.t, (m.get(t.t) || 0) + 1); }
  const maj = [...counts.entries()].sort((a, b) => b[1] - a[1])[0][0];
  const look = (w) => { const m = byWord.get(lower(w)); return m ? [...m.entries()].sort((a, b) => b[1] - a[1])[0][0] : maj; };
  const majAcc = tokTest.filter((t) => t.t === maj).length / tokTest.length;
  const lookAcc = tokTest.filter((t) => look(t.w) === t.t).length / tokTest.length;
  const unseen = tokTest.filter((t) => !byWord.has(lower(t.w))).length / tokTest.length;
  // the lookup baseline on the control task (memorizing word types solves it for seen words)
  const ctl = controlLabels(), cByWord = new Map(); tokTrain.forEach((t, n) => cByWord.set(lower(t.w), ctl.train[n]));
  const cMaj = (() => { const c = new Map(); for (const v of ctl.train) c.set(v, (c.get(v) || 0) + 1); return [...c.entries()].sort((a, b) => b[1] - a[1])[0][0]; })();
  const lookCtl = tokTest.filter((t, n) => (cByWord.has(lower(t.w)) ? cByWord.get(lower(t.w)) : cMaj) === ctl.test[n]).length / tokTest.length;
  return { majAcc, lookAcc, lookCtl, unseen, majTag: D.tags[maj], yt };
}

async function trainProbes() {
  $('#train').disabled = true; $('#prog').textContent = 'Training linear probes (4 runs, in a Web Worker)…';
  const ctl = controlLabels(), rv = { train: randomVectors(tokTrain), test: null };
  // random vectors must be consistent between train and test: build on the concatenation
  const allR = randomVectors([...tokTrain, ...tokTest]); rv.train = allR.subarray(0, tokTrain.length * DIM); rv.test = allR.subarray(tokTrain.length * DIM);
  const t0 = performance.now();
  const [mPos, mCtl, rPos, rCtl] = await Promise.all([
    runProbe({ kind: 'linear', X: emb.train, y: yTrain(), Xt: emb.test, yt: yTest() }),
    runProbe({ kind: 'linear', X: emb.train, y: ctl.train, Xt: emb.test, yt: ctl.test }),
    runProbe({ kind: 'linear', X: rv.train.slice(), y: yTrain(), Xt: rv.test.slice(), yt: yTest() }),
    runProbe({ kind: 'linear', X: rv.train.slice(), y: ctl.train, Xt: rv.test.slice(), yt: ctl.test }),
  ]);
  rows.mini = { name: 'MiniLM (contextual) · linear probe', pos: mPos.acc, ctl: mCtl.acc, train: mPos.trainAcc };
  rows.rand = { name: 'Random vector per word type · linear probe', pos: rPos.acc, ctl: rCtl.acc, train: rPos.trainAcc };
  preds = mPos.pred;
  const b = baselines();
  $('#insight').textContent = `With L2 = ${$('#l2').value}: the MiniLM probe reaches ${pct(mPos.acc)} against ${pct(b.lookAcc)} for a plain word lookup, so context adds ${((mPos.acc - b.lookAcc) * 100).toFixed(1)} points here. Its train accuracy is ${pct(mPos.trainAcc)}, so the gap to test accuracy is overfitting. Random word-type vectors do better on the control task (${pct(rCtl.acc)}) than on real tags (${pct(rPos.acc)}): a random label per word type is exactly what memorizing word identity solves, while real tags need context for ambiguous and unseen words.`;
  $('#prog').textContent = `Linear probes trained in ${((performance.now() - t0) / 1000).toFixed(1)} s.`;
  await renderResults(); await renderConfusion(); renderSentence();
  $('#mlp').disabled = false; $('#lc').disabled = false;
  window.__lab.results = summary();
}

async function trainMLP() {
  $('#mlp').disabled = true; $('#prog').textContent = 'Training MLP probes (64 hidden units, 2 runs)… this takes a little longer.';
  const ctl = controlLabels(), t0 = performance.now();
  const [p, c] = await Promise.all([
    runProbe({ kind: 'mlp', hidden: 64, X: emb.train, y: yTrain(), Xt: emb.test, yt: yTest() }),
    runProbe({ kind: 'mlp', hidden: 64, X: emb.train, y: ctl.train, Xt: emb.test, yt: ctl.test }),
  ]);
  $('#insight2').hidden = false;
  rows.mlp = { name: 'MiniLM (contextual) · MLP probe (64 hidden)', pos: p.acc, ctl: c.acc, train: p.trainAcc };
  const lin = rows.mini;
  $('#insight2').textContent = `The MLP probe changes POS accuracy by ${((p.acc - lin.pos) * 100).toFixed(1)} points but control accuracy by ${((c.acc - lin.ctl) * 100).toFixed(1)}, so selectivity goes from ${((lin.pos - lin.ctl) * 100).toFixed(1)} to ${((p.acc - c.acc) * 100).toFixed(1)} points: a more expressive probe memorizes more.`;
  $('#prog').textContent = `MLP probes trained in ${((performance.now() - t0) / 1000).toFixed(1)} s.`;
  await renderResults();
  window.__lab.results = summary();
}

async function learningCurve() {
  $('#lc').disabled = true;
  const sizes = [25, 50, 100, 200, 400], yt = yTest();
  const allR = randomVectors([...tokTrain, ...tokTest]), rTrain = allR.subarray(0, tokTrain.length * DIM), rTest = allR.subarray(tokTrain.length * DIM);
  const series = { mini: [], rand: [] }, y = yTrain();
  for (const s of sizes) {
    $('#prog').textContent = `Learning curve: ${s} training sentences…`;
    const a = sliceTrain(emb.train, s), b = sliceTrain(rTrain, s), ys = Int32Array.from(a.idx, (n) => y[n]);
    const [m, r] = await Promise.all([runProbe({ kind: 'linear', X: a.X, y: ys, Xt: emb.test, yt }), runProbe({ kind: 'linear', X: b.X, y: ys, Xt: rTest.slice(), yt })]);
    series.mini.push(m.acc); series.rand.push(r.acc);
  }
  const P = await plotly();
  P.react($('#curve'), [
    { x: sizes, y: series.mini.map((v) => v * 100), name: 'MiniLM · linear', mode: 'lines+markers', line: { color: COLORS[0] } },
    { x: sizes, y: series.rand.map((v) => v * 100), name: 'Random word vectors · linear', mode: 'lines+markers', line: { color: COLORS[3] } },
  ], plotLayout({ margin: { l: 50, r: 12, t: 10, b: 46 }, xaxis: { type: 'log', title: { text: 'training sentences (log scale)' }, color: '#9aa3ad', gridcolor: 'rgba(228,231,235,.08)', tickvals: sizes }, yaxis: { title: { text: 'test POS accuracy %' }, color: '#9aa3ad', gridcolor: 'rgba(228,231,235,.08)' }, legend: { orientation: 'h', y: -0.25 } }), PLOT_CONFIG);
  $('#prog').textContent = 'Learning curve done.';
  $('#lc').disabled = false;
  window.__lab.curve = { sizes, ...series };
}

function summary() { const b = baselines(); return { majority: b.majAcc, lookup: b.lookAcc, lookupControl: b.lookCtl, ...Object.fromEntries(Object.entries(rows).map(([k, v]) => [k, { pos: v.pos, ctl: v.ctl, sel: v.pos - v.ctl, train: v.train }])) }; }

const pct = (v) => (v == null ? '—' : (v * 100).toFixed(1) + '%');
async function renderResults() {
  const b = baselines();
  const list = [
    { name: `Majority tag (always ${b.majTag})`, pos: b.majAcc, ctl: null },
    { name: 'Most frequent training tag per word (lookup; unseen → majority)', pos: b.lookAcc, ctl: b.lookCtl },
    ...['rand', 'mini', 'mlp'].filter((k) => rows[k]).map((k) => rows[k]),
  ];
  const bestSel = Math.max(...list.filter((r) => r.ctl != null && r.train != null).map((r) => r.pos - r.ctl));
  $('#res tbody').replaceChildren(...list.map((r) => el('tr', {}, [
    el('td', { text: r.name }), el('td', { class: 'num', text: pct(r.pos) }), el('td', { class: 'num', text: pct(r.ctl) }),
    el('td', { class: 'num' + (r.ctl != null && r.train != null && r.pos - r.ctl === bestSel ? ' sel' : ''), text: r.ctl == null ? '—' : ((r.pos - r.ctl) * 100).toFixed(1) + ' pts' }),
    el('td', { class: 'num', text: pct(r.train) }),
  ])));
  const P = await plotly();
  const named = list.filter((r) => r.ctl != null);
  P.react($('#bars'), [
    { type: 'bar', name: 'POS (real task)', x: named.map((r) => r.name.split(' · ')[0].replace(/ \(.*/, '') + (r.name.includes('MLP') ? ' · MLP' : r.name.includes('linear') ? ' · linear' : '')), y: named.map((r) => r.pos * 100), marker: { color: COLORS[0] } },
    { type: 'bar', name: 'Control task', x: named.map((r) => r.name.split(' · ')[0].replace(/ \(.*/, '') + (r.name.includes('MLP') ? ' · MLP' : r.name.includes('linear') ? ' · linear' : '')), y: named.map((r) => r.ctl * 100), marker: { color: COLORS[3] } },
  ], plotLayout({ barmode: 'group', margin: { l: 50, r: 10, t: 10, b: 90 }, yaxis: { range: [0, 100], title: { text: 'test accuracy %' }, color: '#9aa3ad', gridcolor: 'rgba(228,231,235,.08)' }, xaxis: { color: '#9aa3ad', tickfont: { size: 11 } }, legend: { orientation: 'h', y: 1.1 } }), PLOT_CONFIG);
}

async function renderConfusion() {
  const k = D.tags.length, C = Array.from({ length: k }, () => new Array(k).fill(0));
  tokTest.forEach((t, n) => C[t.t][preds[n]]++);
  const Z = C.map((row) => { const s = row.reduce((a, b) => a + b, 0); return row.map((v) => (s ? v / s : null)); });
  const P = await plotly();
  P.react($('#conf'), [{ type: 'heatmap', z: Z, x: D.tags, y: D.tags, colorscale: [[0, '#121820'], [0.5, '#416180'], [1, '#d6ebff']], zmin: 0, zmax: 1,
    customdata: C, hovertemplate: 'gold %{y} → predicted %{x}<br>%{customdata} words (%{z:.0%})<extra></extra>', colorbar: { tickfont: { color: '#9aa3ad' }, outlinewidth: 0 } }],
  plotLayout({ margin: { l: 60, r: 10, t: 10, b: 60 }, yaxis: { autorange: 'reversed', title: { text: 'gold' }, color: '#9aa3ad', tickfont: { size: 10 } }, xaxis: { title: { text: 'predicted' }, color: '#9aa3ad', tickfont: { size: 10 } } }), PLOT_CONFIG);
}

function renderSentence() {
  const ids = [...new Set(tokTest.map((t) => t.s))];
  sIdx = (sIdx + ids.length) % ids.length;
  const si = ids[sIdx], toks = tokTest.map((t, n) => [t, n]).filter(([t]) => t.s === si);
  $('#sidx').textContent = `Test sentence ${sIdx + 1} of ${ids.length}`;
  $('#sent').replaceChildren(...toks.map(([t, n]) => {
    const p = preds ? D.tags[preds[n]] : '—', ok = preds && preds[n] === t.t;
    return el('div', { class: 'tw' }, [el('span', { class: 'w', text: t.w }), el('span', { class: 'g', text: D.tags[t.t] }), el('span', { class: 'p ' + (preds ? (ok ? 'ok' : 'bad') : ''), text: p })]);
  }));
}

async function init() {
  try { D = await getJSON('/lab/probing/data/sentences.json'); } catch (e) { fail(e.message); return; }
  nTrain = D.train; flatten();
  const types = new Set(tokTrain.map((t) => lower(t.w))), unseen = tokTest.filter((t) => !types.has(lower(t.w))).length;
  $('#dstats').replaceChildren(...[[D.sentences.length, 'sentences'], [tokTrain.length.toLocaleString('en-US'), 'training words'], [tokTest.length.toLocaleString('en-US'), 'test words'], [D.tags.length, 'UPOS tags'], [(unseen / tokTest.length * 100).toFixed(1) + '%', 'test words unseen in training']].map(([b, s]) => el('div', {}, [el('b', { text: String(b) }), el('span', { text: s })])));
  await renderResults();
  renderSentence();
  $('#embed').disabled = false;
  $('#embed').onclick = () => embedAll().catch((e) => { $('#prog').textContent = 'Embedding failed: ' + e.message; $('#embed').disabled = false; });
  $('#train').onclick = () => trainProbes().catch((e) => { $('#prog').textContent = 'Training failed: ' + e.message; });
  $('#mlp').onclick = () => trainMLP().catch((e) => { $('#prog').textContent = 'Training failed: ' + e.message; });
  $('#lc').onclick = () => learningCurve().catch((e) => { $('#prog').textContent = 'Learning curve failed: ' + e.message; });
  $('#prevs').onclick = () => { sIdx--; renderSentence(); }; $('#nexts').onclick = () => { sIdx++; renderSentence(); };
  window.__lab = Object.assign(window.__lab || {}, { baselines: () => { const b = baselines(); delete b.yt; return b; } });
  ready();
}
init();
