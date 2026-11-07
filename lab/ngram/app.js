import { el, $, ready, fail, plotly, plotLayout, PLOT_CONFIG, COLORS, getJSON } from '/lab/shared/lab.js';

const worker = new Worker(new URL('./worker.js', import.meta.url), { type: 'module' });
let seq = 0;
const pending = new Map();
worker.onmessage = (e) => { const { id, ok, result, error } = e.data; const p = pending.get(id); pending.delete(id); ok ? p.res(result) : p.rej(new Error(error)); };
worker.onerror = (e) => { for (const p of pending.values()) p.rej(new Error(e.message || 'Worker failed')); pending.clear(); };
const call = (cmd, args = {}) => new Promise((res, rej) => { const id = ++seq; pending.set(id, { res, rej }); worker.postMessage({ id, cmd, ...args }); });

const METHOD_NOTES = {
  mle: 'Raw relative frequencies. Unseen n-grams get probability 0, so held-out perplexity is infinite above unigrams.',
  addk: 'Adds k to every count. k = 1 is Laplace smoothing; try 1 vs 0.01 and compare perplexity.',
  interp: 'λ-weighted mix of trigram, bigram and unigram estimates (0.6 / 0.3 / 0.1; bigram model: 0.7 / 0.3).',
  kn: 'Absolute discounting with continuation-count lower orders; the standard strong n-gram baseline.',
  backoff: 'Uses the longest matching n-gram, multiplying by 0.4 per back-off step. Scores are renormalized for display; not used for perplexity.',
};
const NAMES = { mle: 'MLE', addk: 'Add-k', interp: 'Interpolation', kn: 'Kneser–Ney' };
const fmt = (n) => Number(n).toLocaleString('en-US');
const fmtP = (p) => (p >= 0.001 ? p.toFixed(4) : p.toExponential(2));

let stats = null, trained = false;
const params = () => ({ k: Math.max(1e-4, +$('#k').value || 0.01), D: Math.min(0.99, Math.max(0.01, +$('#D').value || 0.75)), lambdas: { 1: [1, 0, 0], 2: [0.3, 0.7, 0], 3: [0.1, 0.3, 0.6] } });
const order = () => +$('#order').value;
const method = () => $('#method').value;

function status(text, frac) {
  const s = $('#status'); s.querySelector('.t').textContent = text; if (frac != null) s.querySelector('i').style.width = Math.round(frac * 100) + '%';
}

function detok(words) {
  return words.join(' ').replace(/ ([.,;:!?])/g, '$1').replace(/(^|[.!?] )([a-z])/g, (m, a, b) => a + b.toUpperCase()).replace(/\bi\b/g, 'I');
}

async function train() {
  trained = false;
  $('#modelpanel').classList.add('busy');
  status('Loading and counting…', 0.3);
  const key = $('#corpus').value;
  try {
    stats = await call('train', { url: `/lab/ngram/data/${key}.txt`, minCount: 2 });
  } catch (e) { status('Training failed: ' + e.message); fail(e.message); return; }
  $('#modelpanel').classList.remove('busy');
  trained = true;
  status(`Trained in ${fmt(stats.ms)} ms (in a background thread).`, 1);
  $('#mstats').replaceChildren(...[[fmt(stats.trainTokens), 'training tokens'], [fmt(stats.V - 1), 'vocabulary (count ≥ 2, + <UNK>)'], [fmt(stats.bigrams), 'bigram types'], [fmt(stats.trigrams), 'trigram types'], [fmt(stats.testTokens), 'held-out tokens']]
    .map(([n, l]) => el('div', {}, [el('b', { text: n }), el('span', { text: l })])));
  await Promise.all([predict(), generate(), score(), perplexity()]);
}

async function predict() {
  if (!trained) return;
  const r = await call('next', { context: $('#ctx').value, order: order(), method: method(), p: params() });
  const ctx = r.context.length ? r.context.map((w) => `“${w}”`).join(' ') : '(none: unigram)';
  $('#ctxnote').textContent = `Context used: ${ctx} · seen ${fmt(r.contextCount)} times in training.` + (method() === 'mle' && !r.contextCount ? ' MLE has nothing to say about an unseen context.' : '');
  const tb = $('#next tbody'); tb.innerHTML = '';
  const max = r.top.length ? r.top[0].p : 1;
  for (const t of r.top) tb.append(el('tr', {}, [el('td', { text: t.w }), el('td', { text: fmtP(t.p) }), el('td', { text: fmt(t.count) }), el('td', { class: 'bar' }, el('div', { class: 'pbar', style: { width: (t.p / max * 100).toFixed(1) + '%' } }))]));
  if (!r.top.length) tb.append(el('tr', {}, el('td', { colspan: 4, class: 'note', text: 'Every word has probability 0 here.' })));
  window.__lab.next = r;
}

async function generate() {
  if (!trained) return;
  const r = await call('generate', { seed: $('#seed').value, order: order(), method: method(), p: params(), temperature: +$('#temp').value });
  const g = $('#gen'); g.innerHTML = '';
  const all = [...r.seed, ...r.words];
  const text = detok(all);
  const seedText = detok(r.seed);
  const p = el('p');
  if (seedText) p.append(el('span', { class: 'seed', text: seedText }), ' ');
  p.append(text.slice(seedText.length).trim() || (method() === 'mle' ? '(stopped: this context never occurred in training)' : '…'));
  g.append(p);
  window.__lab.generated = r;
}

async function score() {
  if (!trained) return;
  const r = await call('score', { sentence: $('#sent').value, order: order(), method: method(), p: params() });
  const tb = $('#stab tbody'); tb.innerHTML = '';
  for (const t of r.rows) {
    const nl = t.p > 0 ? -Math.log2(t.p) : Infinity;
    const heat = t.p > 0 ? Math.min(1, nl / 20) : 1;
    tb.append(el('tr', {}, [
      el('td', {}, [el('span', { class: 'heat', style: { background: `color-mix(in srgb, #d98c7a ${Math.round(heat * 100)}%, #8fbf9a)` } }), t.w + (t.mapped === '<UNK>' && t.w !== '</s>' ? ' → <UNK>' : '')]),
      el('td', { text: `${fmt(t.tri)} / ${fmt(t.bi)} / ${fmt(t.uni)}` }),
      el('td', { text: t.p > 0 ? fmtP(t.p) : '0' }),
      el('td', { text: Number.isFinite(nl) ? nl.toFixed(2) : '∞' }),
    ]));
  }
  $('#snote').textContent = Number.isFinite(r.log2) ? `log₂ P(sentence) = ${r.log2.toFixed(2)} · per-token perplexity ${r.ppl.toFixed(1)}` : 'Probability 0: at least one n-gram was never seen (try a smoothed method).';
  window.__lab.score = r;
}

async function perplexity() {
  if (!trained) return;
  const r = await call('perplexity', { p: params() });
  const methods = Object.keys(r);
  const tab = $('#ptab'); tab.innerHTML = '';
  tab.append(el('thead', {}, el('tr', {}, [el('th', { scope: 'col', text: 'Method' }), ...[1, 2, 3].map((o) => el('th', { scope: 'col', text: ['Unigram', 'Bigram', 'Trigram'][o - 1] }))])));
  const tb = el('tbody');
  for (const m of methods) tb.append(el('tr', {}, [el('td', { text: NAMES[m] + (m === 'addk' ? ` (k=${params().k})` : m === 'kn' ? ` (D=${params().D})` : '') }), ...r[m].map((x) => el('td', { text: x.ppl === 'Infinity' ? `∞ (${fmt(x.zeros)} zero-prob tokens)` : x.ppl.toFixed(1) }))]));
  tab.append(tb);
  window.__lab.perplexity = r;
  try {
    const P = await plotly();
    const traces = methods.map((m, i) => ({ type: 'bar', name: NAMES[m], x: ['Unigram', 'Bigram', 'Trigram'], y: r[m].map((x) => (x.ppl === 'Infinity' ? null : x.ppl)), marker: { color: COLORS[i] }, hovertemplate: '%{x}: %{y:.1f}<extra>' + NAMES[m] + '</extra>' }));
    P.react($('#pplot'), traces, plotLayout({ barmode: 'group', yaxis: { type: 'log', title: { text: 'perplexity (log scale)', font: { size: 12, color: '#9aa3ad' } }, gridcolor: 'rgba(228,231,235,.08)', color: '#9aa3ad' }, legend: { orientation: 'h', x: 0, y: 1.02, yanchor: 'bottom' }, margin: { l: 56, r: 10, t: 40, b: 36 } }), PLOT_CONFIG);
  } catch (e) { $('#pplot').textContent = e.message; }
}

function syncMethodUI() {
  $('#kwrap').hidden = method() !== 'addk';
  $('#dwrap').hidden = method() !== 'kn';
  $('#methodnote').textContent = METHOD_NOTES[method()];
}

async function init() {
  let corpora;
  try { corpora = await getJSON('/lab/ngram/data/corpora.json'); } catch (e) { fail(e.message); return; }
  const sel = $('#corpus');
  for (const c of corpora) sel.append(el('option', { value: c.key, text: `${c.name} (${fmt(c.tokens)} tokens)` }));
  sel.value = 'austen';
  syncMethodUI();
  sel.addEventListener('change', train);
  const refresh = () => { syncMethodUI(); predict(); generate(); score(); };
  $('#order').addEventListener('change', refresh);
  $('#method').addEventListener('change', refresh);
  for (const id of ['#k', '#D']) $(id).addEventListener('change', () => { refresh(); perplexity(); });
  $('#nform').addEventListener('submit', (e) => { e.preventDefault(); predict(); });
  $('#gform').addEventListener('submit', (e) => { e.preventDefault(); generate(); });
  $('#sform').addEventListener('submit', (e) => { e.preventDefault(); score(); });
  $('#temp').addEventListener('input', () => { $('#tval').textContent = (+$('#temp').value).toFixed(1); });
  $('#temp').addEventListener('change', generate);
  window.__lab = Object.assign(window.__lab || {}, { call });
  await train();
  if (trained) ready({ stats });
}
init();
