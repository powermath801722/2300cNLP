// Per-token surprisal, perplexity and GLTR rank coloring from GPT-2 / distilgpt2.
import { el, $, ready, fail, plotly, plotLayout, PLOT_CONFIG } from '/lab/shared/lab.js';
import { progressUI, transformers as T } from '/lab/shared/hf.js';

// Load the merged decoder (decoder_model_merged_quantized.onnx: ≈85 MB distilgpt2 / ≈128 MB gpt2).
// The shared loadModel() would fetch model_quantized.onnx, which for these Xenova repos is ≈237 MB.
const _gpt = {};
function loadGPT(id, ui, readyText) {
  const cb = ui ? (p) => ui.onProgress(p) : undefined;
  return (_gpt[id] ||= Promise.all([
    T.AutoTokenizer.from_pretrained(id, { progress_callback: cb }),
    T.AutoModelForCausalLM.from_pretrained(id, { dtype: 'q8', device: 'wasm', model_file_name: 'decoder_model_merged', progress_callback: cb }),
  ]).then(([tokenizer, model]) => { ui && ui.done(readyText); return { tokenizer, model }; })
    .catch((e) => { delete _gpt[id]; ui && ui.error(e); throw e; }));
}


const EOS = 50256, V = 50257, MAX_TOKENS = 300;
const LN2 = Math.LN2;

const SAMPLES = [
  { label: 'Human · Thoreau, Walden (1854)', text: 'After a still winter night I awoke with the impression that some question had been put to me, which I had been endeavoring in vain to answer in my sleep, as what—how—when—where? But there was dawning Nature, in whom all creatures live, looking in at my broad windows with serene and satisfied face, and no question on her lips.' },
  { label: 'Machine · distilgpt2 greedy', note: 'Generated on this site by distilgpt2 with greedy decoding (repetition penalty 1.2) from the prompt “The history of the river valley begins”.', text: 'The history of the river valley begins with a series that began in 1812. The first two rivers were named after an area known as "the River Valley" and later became part of the Great Lakes, but they are now considered by historians to be one of the most important historical events ever recorded on this historic site' },
  { label: 'Garden path · The horse raced past the barn fell.', text: 'The horse raced past the barn fell.' },
  { label: 'Control · The horse that was raced past the barn fell.', text: 'The horse that was raced past the barn fell.' },
  { label: 'Garden path · The old man the boat.', text: 'The old man the boat.' },
  { label: 'Garden path · While the man hunted the deer ran into the woods.', text: 'While the man hunted the deer ran into the woods.' },
];

const BUCKETS = [
  { name: 'top-10', max: 10, color: '#8fbf9a' },
  { name: 'top-100', max: 100, color: '#d8b66a' },
  { name: 'top-1000', max: 1000, color: '#d98c7a' },
  { name: 'beyond 1000', max: Infinity, color: '#b49bd6' },
];
const bucketOf = (rank) => BUCKETS.findIndex((b) => rank <= b.max);

let view = 'surprisal', current = null, selected = -1, sampleLabel = null;
const ui = progressUI($('#load'), 'model');
const models = {};

async function getModel(id) {
  const label = id.endsWith('distilgpt2') ? 'distilgpt2 (≈85 MB)' : 'gpt2 (≈128 MB)';
  ui.set(`Preparing ${label}…`, 0);
  const m = await loadGPT(id, ui);
  ui.done(`Ready · ${id.replace('Xenova/', '')} (8-bit), cached in your browser`);
  return m;
}

function top5(row) {
  const idx = [];
  for (let i = 0; i < row.length; i++) {
    if (idx.length < 5) { idx.push(i); idx.sort((a, b) => row[b] - row[a]); }
    else if (row[i] > row[idx[4]]) { idx[4] = i; idx.sort((a, b) => row[b] - row[a]); }
  }
  return idx;
}

async function analyze(text, modelId) {
  const { tokenizer, model } = await getModel(modelId);
  const ids = tokenizer.encode(text);
  if (!ids.length) throw new Error('Type or paste some text first.');
  const used = ids.slice(0, MAX_TOKENS);
  const input = [EOS].concat(used);
  const n = input.length;
  const o = await model({
    input_ids: new T.Tensor('int64', BigInt64Array.from(input.map(BigInt)), [1, n]),
    attention_mask: new T.Tensor('int64', new BigInt64Array(n).fill(1n), [1, n]),
  });
  const d = o.logits.data, toks = [];
  for (let i = 0; i < used.length; i++) {
    const row = d.subarray(i * V, (i + 1) * V), tok = used[i];
    let m = -Infinity; for (let j = 0; j < V; j++) if (row[j] > m) m = row[j];
    let s = 0, rank = 1; const zt = row[tok];
    for (let j = 0; j < V; j++) { s += Math.exp(row[j] - m); if (row[j] > zt) rank++; }
    const lse = m + Math.log(s);
    const bits = (lse - zt) / LN2;
    const top = top5(row).map((j) => ({ id: j, p: Math.exp(row[j] - lse) }));
    toks.push({ id: tok, text: tokenizer.decode([tok]), bits, p: Math.exp(zt - lse), rank, top: top.map((x) => ({ ...x, text: tokenizer.decode([x.id]) })) });
    if (i % 40 === 39) await new Promise((r) => setTimeout(r, 0));
  }
  const total = toks.reduce((a, t) => a + t.bits, 0), mean = total / toks.length;
  const buckets = BUCKETS.map((_, b) => toks.filter((t) => bucketOf(t.rank) === b).length);
  return { modelId, toks, total, mean, ppl: Math.pow(2, mean), buckets, truncated: ids.length > MAX_TOKENS };
}

// ---------- rendering ----------
const pct = (x, n) => (100 * x / n).toFixed(0) + '%';
function tokenStyle(t) {
  if (view === 'surprisal') {
    const a = Math.min(1, t.bits / 16);
    return { background: `rgba(217,140,122,${(a * 0.85).toFixed(3)})`, edge: 'transparent' };
  }
  const c = BUCKETS[bucketOf(t.rank)].color;
  return { background: c + '55', edge: c };
}
function renderTokens() {
  const host = $('#tokens'); host.innerHTML = '';
  current.toks.forEach((t, i) => {
    const st = tokenStyle(t);
    const b = el('button', {
      class: 'tk', type: 'button', 'aria-pressed': String(i === selected),
      'aria-label': `“${t.text}”: ${t.bits.toFixed(1)} bits, rank ${t.rank}`,
      text: t.text.replace(/\n/g, '↵\n'),
      onclick: () => { selected = i; renderTokens(); renderDetail(); },
    });
    b.style.background = st.background; b.style.setProperty('--edge', st.edge);
    host.append(b);
  });
  if (current.truncated) host.append(el('p', { class: 'note', text: `Only the first ${MAX_TOKENS} tokens were analyzed.` }));
  const lg = $('#legend'); lg.innerHTML = '';
  if (view === 'surprisal') lg.append(el('span', {}, ['0 bits', el('span', { class: 'ramp', 'aria-hidden': 'true' }), '16+ bits']), el('span', { text: 'Darker = more surprising to the model.' }));
  else lg.append(...BUCKETS.map((b) => el('span', {}, [el('i', { style: { background: b.color + '88', boxShadow: `inset 0 -2px 0 ${b.color}` }, 'aria-hidden': 'true' }), `${b.name}`])));
}
function renderDetail() {
  const host = $('#detail');
  host.innerHTML = '';
  host.append(el('h3', { style: { fontSize: '22px' }, text: 'Token detail' }));
  const t = current && current.toks[selected];
  if (!t) { host.append(el('p', { class: 'note', text: 'Click any token to see its probability, rank and what the model expected instead.' })); return; }
  const prefix = current.toks.slice(Math.max(0, selected - 8), selected).map((x) => x.text).join('');
  host.append(
    el('p', { class: 'note', text: `…${prefix}` }, []),
    el('p', { style: { margin: '2px 0 0', fontSize: '20px' } }, [el('code', { text: JSON.stringify(t.text) })]),
    el('dl', {}, [
      el('dt', { text: 'Surprisal' }), el('dd', { text: `${t.bits.toFixed(2)} bits` }),
      el('dt', { text: 'Probability' }), el('dd', { text: t.p < 1e-4 ? t.p.toExponential(2) : (t.p * 100).toFixed(2) + '%' }),
      el('dt', { text: 'Rank' }), el('dd', { text: `${t.rank.toLocaleString()} of 50,257 (${BUCKETS[bucketOf(t.rank)].name})` }),
    ]),
    el('p', { class: 'note', style: { marginTop: '10px' }, text: 'The model’s top-5 predictions at this position:' }),
    el('ol', {}, t.top.map((x) => el('li', {}, [el('code', { text: JSON.stringify(x.text) }), ` ${(x.p * 100).toFixed(1)}%`]))),
  );
}
function renderMetrics() {
  const c = current, n = c.toks.length, host = $('#metrics'); host.innerHTML = '';
  const peak = c.toks.reduce((a, t, i) => (t.bits > c.toks[a].bits ? i : a), 0);
  const items = [[n, 'tokens'], [c.total.toFixed(0), 'total surprisal (bits)'], [c.mean.toFixed(2), 'mean bits per token'], [c.ppl.toFixed(1), 'perplexity'], [pct(c.buckets[0], n), 'tokens in model’s top-10'], [JSON.stringify(c.toks[peak].text), `peak: ${c.toks[peak].bits.toFixed(1)} bits`]];
  host.append(...items.map(([v, l]) => el('div', {}, [el('b', { text: String(v) }), el('span', { text: l })])));
}
async function renderCharts() {
  const P = await plotly(), c = current;
  const x = c.toks.map((_, i) => i + 1), base = plotLayout();
  P.react('line', [{
    type: 'scatter', mode: 'lines+markers', x, y: c.toks.map((t) => t.bits), text: c.toks.map((t) => JSON.stringify(t.text)),
    line: { color: '#d98c7a', width: 1.5 }, marker: { size: 5, color: c.toks.map((t) => BUCKETS[bucketOf(t.rank)].color) },
    hovertemplate: '%{text}<br>%{y:.2f} bits<extra></extra>',
  }], plotLayout({ margin: { l: 48, r: 12, t: 28, b: 40 }, title: { text: 'Surprisal per token', font: { size: 14 }, x: 0.02 }, xaxis: { ...base.xaxis, title: { text: 'token position', font: { size: 12 } } }, yaxis: { ...base.yaxis, title: { text: 'bits', font: { size: 12 } }, rangemode: 'tozero' } }), PLOT_CONFIG);
  P.react('hist', [{
    type: 'bar', x: BUCKETS.map((b) => b.name), y: c.buckets.map((k) => k / c.toks.length), marker: { color: BUCKETS.map((b) => b.color) },
    text: c.buckets.map((k) => String(k)), textposition: 'outside', hovertemplate: '%{x}: %{text} tokens (%{y:.0%})<extra></extra>', cliponaxis: false,
  }], plotLayout({ margin: { l: 48, r: 12, t: 28, b: 40 }, title: { text: 'GLTR rank buckets', font: { size: 14 }, x: 0.02 }, yaxis: { ...base.yaxis, tickformat: '.0%', range: [0, 1.1] } }), PLOT_CONFIG);
}
function logRun(label) {
  const c = current, n = c.toks.length, tb = $('#runs tbody');
  tb.prepend(el('tr', {}, [
    el('td', { text: label }), el('td', { text: c.modelId.replace('Xenova/', '') }), el('td', { text: String(n) }),
    el('td', { text: c.mean.toFixed(2) }), el('td', { text: c.ppl.toFixed(1) }),
    ...c.buckets.map((k) => el('td', { text: pct(k, n) })),
  ]));
}

function showMsg(t) { const m = $('#msg'); m.hidden = !t; m.textContent = t || ''; }
async function run() {
  showMsg('');
  const text = $('#text').value, modelId = $('#model').value;
  if (!text.trim()) { showMsg('Type or paste some text first.'); return; }
  $('#go').disabled = true;
  try {
    current = await analyze(text, modelId);
    const peak = current.toks.reduce((a, t, i) => (t.bits > current.toks[a].bits ? i : a), 0);
    selected = peak;
    renderTokens(); renderDetail(); renderMetrics(); await renderCharts();
    const label = sampleLabel && SAMPLES.find((s) => s.label === sampleLabel && s.text === text) ? sampleLabel : (text.trim().slice(0, 40) + (text.trim().length > 40 ? '…' : ''));
    logRun(label);
    window.__lab.last = { model: modelId, mean: current.mean, ppl: current.ppl, buckets: current.buckets, tokens: current.toks.map((t) => ({ text: t.text, bits: t.bits, rank: t.rank })) };
  } catch (e) { showMsg(e.message || String(e)); }
  finally { $('#go').disabled = false; }
}
function setView(v) {
  view = v;
  $('#v-s').setAttribute('aria-selected', String(v === 'surprisal'));
  $('#v-r').setAttribute('aria-selected', String(v === 'rank'));
  if (current) renderTokens();
}

try {
  $('#text').value = SAMPLES[2].text; sampleLabel = SAMPLES[2].label;
  $('#samples').append(...SAMPLES.map((s) => el('button', { class: 'chip', type: 'button', title: s.note || null, text: s.label, onclick: () => { $('#text').value = s.text; sampleLabel = s.label; run(); } })));
  $('#text').addEventListener('input', () => { sampleLabel = null; });
  $('#samples').after(el('p', { class: 'note', text: 'The machine sample was ' + SAMPLES[1].note.charAt(0).toLowerCase() + SAMPLES[1].note.slice(1) + ' The human sample is public-domain prose.' }));
  $('#go').addEventListener('click', run);
  $('#v-s').addEventListener('click', () => setView('surprisal'));
  $('#v-r').addEventListener('click', () => setView('rank'));
  for (const [a, b] of [['#v-s', '#v-r'], ['#v-r', '#v-s']]) $(a).addEventListener('keydown', (e) => { if (e.key === 'ArrowRight' || e.key === 'ArrowLeft') { $(b).focus(); $(b).click(); } });
  ready({ analyze: (text, model) => analyze(text, model || $('#model').value).then((r) => ({ mean: r.mean, ppl: r.ppl, buckets: r.buckets, tokens: r.toks.map((t) => ({ text: t.text, bits: t.bits, rank: t.rank })) })) });
} catch (e) { fail(e.message || e); }
