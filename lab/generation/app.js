// Decoding strategies on distilgpt2, implemented by hand on top of raw logits with a KV cache.
import { el, $, ready, fail, plotly, plotLayout, PLOT_CONFIG, REDUCED } from '/lab/shared/lab.js';
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


const MODEL = 'Xenova/distilgpt2';
const EOS = 50256;
const V = 50257;

const PROMPTS = [
  'The old lighthouse keeper opened the door and',
  'In a shocking finding, scientists discovered',
  'The best way to learn a new language is',
  'Once upon a time, there was a river that',
];

// ---------- utilities ----------
function mulberry32(seed) {
  let a = seed >>> 0;
  return () => { a = (a + 0x6D2B79F5) >>> 0; let t = a; t = Math.imul(t ^ (t >>> 15), t | 1); t ^= t + Math.imul(t ^ (t >>> 7), t | 61); return ((t ^ (t >>> 14)) >>> 0) / 4294967296; };
}
function logSoftmax(z) {
  let m = -Infinity; for (let i = 0; i < z.length; i++) if (z[i] > m) m = z[i];
  let s = 0; for (let i = 0; i < z.length; i++) s += Math.exp(z[i] - m);
  const lse = m + Math.log(s), out = new Float64Array(z.length);
  for (let i = 0; i < z.length; i++) out[i] = z[i] - lse;
  return out;
}
function topIdx(arr, k) { // indices of the k largest values (partial selection)
  const idx = []; let minV = -Infinity, minPos = -1;
  for (let i = 0; i < arr.length; i++) {
    const v = arr[i];
    if (idx.length < k) { idx.push(i); if (idx.length === k) { minPos = 0; for (let j = 1; j < k; j++) if (arr[idx[j]] < arr[idx[minPos]]) minPos = j; minV = arr[idx[minPos]]; } }
    else if (v > minV) { idx[minPos] = i; minPos = 0; for (let j = 1; j < k; j++) if (arr[idx[j]] < arr[idx[minPos]]) minPos = j; minV = arr[idx[minPos]]; }
  }
  return idx.sort((a, b) => arr[b] - arr[a]);
}
const tick = () => new Promise((r) => setTimeout(r, 0));

// ---------- model ----------
let M = null; // { tokenizer, model }
const ui = progressUI($('#load'), 'distilgpt2 (≈85 MB)');
async function getModel() {
  if (!M) M = await loadGPT(MODEL, ui, 'Ready · distilgpt2 (8-bit), cached in your browser');
  return M;
}
const ids2t = (rows) => new T.Tensor('int64', BigInt64Array.from(rows.flat().map(BigInt)), [rows.length, rows[0].length]);
const mask = (B, n) => new T.Tensor('int64', new BigInt64Array(B * n).fill(1n), [B, n]);
const toPast = (o) => { const p = {}; for (const k of Object.keys(o)) if (k.startsWith('present.')) p[k.replace('present.', 'past_key_values.')] = o[k]; return p; };
function gatherPast(past, rows) { // reorder batch dimension of every cached tensor (beam search)
  const out = {};
  for (const [k, t] of Object.entries(past)) {
    const [, H, L, D] = t.dims, per = H * L * D, data = new t.data.constructor(rows.length * per);
    rows.forEach((r, i) => data.set(t.data.subarray(r * per, (r + 1) * per), i * per));
    out[k] = new T.Tensor(t.type, data, [rows.length, H, L, D]);
  }
  return out;
}
/** Run prompt through the model: returns {past, logits: Float32Array[] per batch row (last position)}. */
async function prefill(model, ids, B = 1) {
  const o = await model({ input_ids: ids2t(Array.from({ length: B }, () => ids)), attention_mask: mask(B, ids.length) });
  const n = ids.length, d = o.logits.data;
  return { past: toPast(o), len: n, logits: Array.from({ length: B }, (_, b) => d.slice((b * n + n - 1) * V, (b * n + n) * V)) };
}
async function step(model, state, lastTokens) {
  const B = lastTokens.length, len = state.len + 1;
  const o = await model({ input_ids: ids2t(lastTokens.map((t) => [t])), attention_mask: mask(B, len), past_key_values: state.past });
  const d = o.logits.data;
  return { past: toPast(o), len, logits: Array.from({ length: B }, (_, b) => d.slice(b * V, (b + 1) * V)) };
}

// ---------- distributions ----------
function applyRepPenalty(z, context, theta) {
  if (theta === 1) return z;
  const out = Float32Array.from(z);
  for (const t of new Set(context)) out[t] = out[t] > 0 ? out[t] / theta : out[t] * theta;
  return out;
}
/** Build the sampling distribution. Returns {pAdj (Float64Array), kept (Set|null), raw (Float64Array, T=1 model probs)} */
function distribution(zRaw, zPen, cfg) {
  const raw = logSoftmax(zRaw).map(Math.exp);
  const Tm = cfg.strategy === 'greedy' || cfg.strategy === 'beam' ? 1 : cfg.temp;
  const z = new Float64Array(zPen.length); for (let i = 0; i < z.length; i++) z[i] = zPen[i] / Tm;
  let p = logSoftmax(z).map(Math.exp), kept = null;
  if (cfg.strategy === 'topk') {
    kept = new Set(topIdx(p, Math.min(cfg.topk, V)));
  } else if (cfg.strategy === 'topp') {
    // sort only plausible candidates (prob > 1e-9 is ample for p ≤ 1 at double precision)
    const cand = []; for (let i = 0; i < p.length; i++) if (p[i] > 1e-9) cand.push(i);
    cand.sort((a, b) => p[b] - p[a]);
    kept = new Set(); let c = 0;
    for (const i of cand) { kept.add(i); c += p[i]; if (c >= cfg.topp) break; }
  }
  if (kept) { let s = 0; for (const i of kept) s += p[i]; const q = new Float64Array(p.length); for (const i of kept) q[i] = p[i] / s; p = q; }
  return { pAdj: p, kept, raw };
}
function sampleFrom(p, rand) {
  let r = rand(), c = 0;
  for (let i = 0; i < p.length; i++) { c += p[i]; if (r < c) return i; }
  let best = 0; for (let i = 1; i < p.length; i++) if (p[i] > p[best]) best = i; return best;
}
function rankOf(arr, id) { let r = 1; const v = arr[id]; for (let i = 0; i < arr.length; i++) if (arr[i] > v) r++; return r; }
/** Top-10 by model probability, plus (when truncation keeps more than 8) the last kept and first cut tokens around the boundary. */
function candList(dist, chosen, zPen, extraKeep) {
  const ids = topIdx(dist.raw, 10);
  let boundary = [];
  if (dist.kept && dist.kept.size > 8) {
    const kept = [...dist.kept].sort((a, b) => dist.pAdj[b] - dist.pAdj[a]);
    const lastKept = kept.slice(-2).filter((i) => !ids.includes(i));
    let c1 = -1, c2 = -1; // two most probable cut tokens (by model probability)
    for (let i = 0; i < dist.raw.length; i++) {
      if (dist.kept.has(i)) continue;
      if (c1 < 0 || dist.raw[i] > dist.raw[c1]) { c2 = c1; c1 = i; } else if (c2 < 0 || dist.raw[i] > dist.raw[c2]) c2 = i;
    }
    boundary = lastKept.concat([c1, c2].filter((i) => i >= 0 && !ids.includes(i)));
  }
  const extra = !ids.includes(chosen) && !boundary.includes(chosen) ? [chosen] : [];
  const mk = (id, gap) => ({ id, gap, rank: rankOf(dist.raw, id), pRaw: dist.raw[id], pAdj: dist.pAdj[id], cut: dist.kept ? !dist.kept.has(id) : extraKeep ? !extraKeep.has(id) : false, chosen: id === chosen });
  return ids.map((id) => mk(id, false)).concat(extra.map((id) => mk(id, true)), boundary.map((id, i) => mk(id, i === 0 && !extra.length)));
}

// ---------- decoding ----------
let abort = false;
async function decode(cfg, onStep) {
  const { tokenizer, model } = await getModel();
  const prompt = tokenizer.encode(cfg.prompt);
  if (!prompt.length) throw new Error('Type a prompt first.');
  if (prompt.length > 200) throw new Error('Please keep the prompt under about 150 words.');
  return cfg.strategy === 'beam' ? beam(model, prompt, cfg, onStep) : single(model, prompt, cfg, onStep);
}

async function single(model, prompt, cfg, onStep) {
  const rand = mulberry32(cfg.seed);
  let st = await prefill(model, prompt);
  const ctx = prompt.slice(), steps = [];
  for (let t = 0; t < cfg.maxnew && !abort; t++) {
    const zRaw = st.logits[0], zPen = applyRepPenalty(zRaw, ctx, cfg.reppen);
    const dist = distribution(zRaw, zPen, cfg);
    let tok;
    if (cfg.strategy === 'greedy') { tok = 0; for (let i = 1; i < V; i++) if (zPen[i] > zPen[tok]) tok = i; }
    else tok = sampleFrom(dist.pAdj, rand);
    const rec = { tok, pRaw: dist.raw[tok], pAdj: dist.pAdj[tok], cands: candList(dist, tok), poolSize: dist.kept ? dist.kept.size : null };
    steps.push(rec); ctx.push(tok); onStep && onStep(steps, prompt);
    if (tok === EOS) break;
    await tick();
    st = await step(model, st, [tok]);
  }
  return { prompt, steps };
}

async function beam(model, prompt, cfg, onStep) {
  const W = cfg.beams;
  let st = await prefill(model, prompt);
  // each beam: {toks, score (sum logp), steps[], done}
  let beams = [{ toks: [], score: 0, steps: [] }];
  const finished = [];
  for (let t = 0; t < cfg.maxnew && !abort && beams.length; t++) {
    const cands = [];
    beams.forEach((b, bi) => {
      const zRaw = st.logits[bi], zPen = applyRepPenalty(zRaw, prompt.concat(b.toks), cfg.reppen);
      const lp = logSoftmax(zPen), raw = logSoftmax(zRaw).map(Math.exp);
      b._dist = { raw, pAdj: lp.map(Math.exp), kept: null };
      for (const id of topIdx(lp, 2 * W)) cands.push({ bi, id, score: b.score + lp[id], lp: lp[id] });
    });
    cands.sort((a, b) => b.score - a.score);
    const next = [], rows = [], survivors = beams.map(() => new Set());
    for (const c of cands) {
      survivors[c.bi].add(c.id);
      const parent = beams[c.bi];
      const rec = { tok: c.id, pRaw: parent._dist.raw[c.id], pAdj: Math.exp(c.lp), cands: null, parentDist: parent._dist, parentSurv: survivors[c.bi] };
      const nb = { toks: parent.toks.concat(c.id), score: c.score, steps: parent.steps.concat(rec) };
      if (c.id === EOS) finished.push(nb); else { next.push(nb); rows.push(c.bi); }
      if (next.length === W) break;
    }
    // finalize candidate lists now that survivors per parent are known
    for (const b of next.concat(finished)) { const r = b.steps[b.steps.length - 1]; if (!r.cands) { r.cands = candList(r.parentDist, r.tok, null, r.parentSurv); r.beamPruned = true; delete r.parentDist; delete r.parentSurv; } }
    for (const b of beams) delete b._dist; // free the 50k-entry arrays
    beams = next;
    const best = bestOf(beams.concat(finished));
    if (best) onStep && onStep(best.steps, prompt);
    if (!beams.length) break;
    await tick();
    st = await step(model, { past: gatherPast(st.past, rows), len: st.len }, beams.map((b) => b.toks[b.toks.length - 1]));
  }
  const best = bestOf(beams.concat(finished));
  return { prompt, steps: best ? best.steps : [] };
}
const bestOf = (list) => list.reduce((a, b) => (!a || b.score / b.toks.length > a.score / a.toks.length ? b : a), null);

// ---------- metrics ----------
function metrics(steps) {
  const toks = steps.map((s) => s.tok), n = toks.length || 1;
  const meanLog2 = steps.reduce((s, r) => s + Math.log2(Math.max(r.pRaw, 1e-30)), 0) / n;
  const ngrams = (k) => { const out = []; for (let i = 0; i + k <= toks.length; i++) out.push(toks.slice(i, i + k).join(',')); return out; };
  const d2 = ngrams(2), tri = ngrams(3);
  const seen = new Map(); tri.forEach((g) => seen.set(g, (seen.get(g) || 0) + 1));
  const repeatedTri = tri.filter((g) => seen.get(g) > 1).length;
  return {
    tokens: toks.length,
    meanLog2, ppl: Math.pow(2, -meanLog2),
    distinct2: d2.length ? new Set(d2).size / d2.length : 1,
    rep3: tri.length ? repeatedTri / tri.length : 0,
    repSet: new Set([...seen].filter(([, c]) => c > 1).map(([g]) => g)),
  };
}

// ---------- rendering ----------
let current = null, selected = -1;
const decodeTok = (id) => (M ? M.tokenizer.decode([id]) : '');
const showTok = (s) => s.replace(/\n/g, '↵ ');
const shade = (p) => { // green for confident picks → coral for long shots
  const t = Math.min(1, Math.max(0, (Math.log10(Math.max(p, 1e-6)) + 3) / 3)); // 0.001..1
  const g = [143, 191, 154], r = [217, 140, 122];
  return `rgb(${g.map((c, i) => Math.round(r[i] + (c - r[i]) * t)).join(',')})`;
};

function renderOut(res, live = false) {
  const out = $('#out'); out.innerHTML = '';
  out.append(el('span', { class: 'prompt', text: M.tokenizer.decode(res.prompt) }));
  res.steps.forEach((s, i) => {
    const b = el('button', { class: 'tk', type: 'button', 'aria-pressed': String(i === selected), 'aria-label': `Step ${i + 1}: “${decodeTok(s.tok)}”, probability ${(s.pRaw * 100).toFixed(1)}%`, text: showTok(decodeTok(s.tok)), onclick: () => { selected = i; renderOut(current); renderStep(); } });
    b.style.setProperty('--c', shade(s.pRaw)); b.style.setProperty('--a', '22%');
    out.append(b);
  });
  if (live) out.append(el('span', { class: 'note', text: ' ▍' }));
}

function renderStep() {
  const host = $('#step'); host.innerHTML = '';
  if (!current || selected < 0 || !current.steps[selected]) { host.append(el('p', { class: 'note', text: 'Generate, then click a token.' })); return; }
  const s = current.steps[selected];
  const pool = s.beamPruned ? 'Beam search: struck-out candidates were pruned from this hypothesis’s expansions.' : s.poolSize != null ? `Sampling pool after truncation: ${s.poolSize.toLocaleString()} of 50,257 tokens.` : current.cfg.strategy === 'greedy' ? 'Greedy: the highest-scoring token is always taken.' : 'Pure sampling: every token in the vocabulary can be drawn.';
  host.append(
    el('p', { style: { margin: '0' } }, [el('strong', { text: `Step ${selected + 1}` }), ` · chose “${showTok(decodeTok(s.tok))}” · model probability ${(s.pRaw * 100).toFixed(2)}%`, current.cfg.strategy !== 'greedy' && current.cfg.strategy !== 'beam' ? ` · sampled with ${(s.pAdj * 100).toFixed(2)}%` : '']),
    el('p', { class: 'note', text: pool }),
  );
  const max = Math.max(...s.cands.map((c) => Math.max(c.pRaw, c.pAdj)));
  const ul = el('ul', { class: 'cands', 'aria-label': 'Top candidates at this step' });
  ul.append(el('li', { 'aria-hidden': 'true' }, [el('span', { class: 'hdr', text: 'Token' }), el('span', { class: 'hdr', text: 'Probability' }), el('span', { class: 'hdr num', text: 'Model' }), el('span', { class: 'hdr num', text: 'Used' })]));
  for (const c of s.cands) {
    if (c.gap) ul.append(el('li', { class: 'note', 'aria-hidden': 'true', text: '⋯ around the truncation boundary', style: { display: 'block', fontSize: '12px' } }));
    const li = el('li', { class: [c.cut ? 'cut' : '', c.chosen ? 'chosen' : ''].join(' ').trim() || null });
    li.append(
      el('span', { class: 'w', title: `rank ${c.rank}`, text: (c.rank > 10 ? `#${c.rank} ` : '') + JSON.stringify(decodeTok(c.id)).slice(1, -1) + (c.chosen ? ' ◂' : '') }),
      el('span', { class: 'bars', 'aria-hidden': 'true' }, [el('i', { class: 'raw', style: { width: (100 * c.pRaw / max) + '%' } }), el('i', { class: 'adj', style: { width: (100 * c.pAdj / max) + '%' } })]),
      el('span', { class: 'num', text: (c.pRaw * 100).toFixed(1) + '%' }),
      el('span', { class: 'num', text: c.cut ? 'cut' : (c.pAdj * 100).toFixed(1) + '%' }),
    );
    li.setAttribute('aria-label', `rank ${c.rank}, ${decodeTok(c.id)}: model ${(c.pRaw * 100).toFixed(1)}%, ${c.cut ? 'cut by truncation' : 'used ' + (c.pAdj * 100).toFixed(1) + '%'}${c.chosen ? ', chosen' : ''}`);
    ul.append(li);
  }
  host.append(ul);
}

function renderMetrics(res) {
  const m = metrics(res.steps), host = $('#metrics'); host.innerHTML = '';
  const items = [[m.tokens, 'tokens generated'], [m.meanLog2.toFixed(2), 'mean log₂ P per token'], [m.ppl.toFixed(1), 'perplexity of the continuation'], [(m.distinct2 * 100).toFixed(0) + '%', 'distinct bigrams'], [(m.rep3 * 100).toFixed(0) + '%', 'tokens in repeated 3-grams']];
  host.append(...items.map(([v, l]) => el('div', {}, [el('b', { text: String(v) }), el('span', { text: l })])));
  return m;
}

async function renderChart(res) {
  const P = await plotly();
  const x = res.steps.map((_, i) => i + 1), y = res.steps.map((s) => s.pRaw);
  const text = res.steps.map((s) => JSON.stringify(decodeTok(s.tok)));
  P.react('chart', [{ type: 'bar', x, y, text, marker: { color: res.steps.map((s) => shade(s.pRaw)) }, hovertemplate: 'step %{x}: %{text}<br>P = %{y:.3f}<extra></extra>', textposition: 'none' }],
    plotLayout({ margin: { l: 48, r: 12, t: 12, b: 40 }, yaxis: { ...plotLayout().yaxis, title: { text: 'P(chosen token)', font: { size: 12 } }, range: [0, 1] }, xaxis: { ...plotLayout().xaxis, title: { text: 'step', font: { size: 12 } }, dtick: 5 }, bargap: 0.15 }),
    { ...PLOT_CONFIG, staticPlot: false });
}

// ---------- controls ----------
const controls = ['temp', 'topk', 'topp', 'beams', 'reppen', 'maxnew'];
function cfgFromUI() {
  return {
    prompt: $('#prompt').value.trim(), strategy: $('#strategy').value,
    temp: +$('#temp').value, topk: +$('#topk').value, topp: +$('#topp').value, beams: +$('#beams').value,
    reppen: +$('#reppen').value, maxnew: +$('#maxnew').value, seed: Math.max(0, parseInt($('#seed').value, 10) || 0),
  };
}
function syncUI() {
  for (const id of controls) $(`#${id}-v`).textContent = $('#' + id).value;
  const s = $('#strategy').value;
  for (const c of document.querySelectorAll('.ctl[data-for]')) c.hidden = !c.dataset.for.split(' ').includes(s);
}
function busy(on) {
  for (const id of ['gen', 'cmp']) $('#' + id).disabled = on;
  $('#stop').disabled = !on;
  if (!on) abort = false;
}
function showMsg(t) { const m = $('#msg'); m.hidden = !t; m.textContent = t || ''; }

async function runOne() {
  showMsg(''); busy(true);
  const cfg = cfgFromUI();
  try {
    selected = -1;
    current = { cfg, prompt: [], steps: [] };
    let last = 0;
    const res = await decode(cfg, (steps, prompt) => {
      current = { cfg, prompt, steps }; selected = steps.length - 1;
      const now = performance.now(); if (now - last > 60 || REDUCED) { last = now; renderOut(current, true); }
    });
    current = { cfg, ...res }; selected = res.steps.length - 1;
    renderOut(current); renderStep(); renderMetrics(current); await renderChart(current);
    window.__lab.last = { strategy: cfg.strategy, text: M.tokenizer.decode(res.steps.map((s) => s.tok)), tokens: res.steps.map((s) => s.tok), probs: res.steps.map((s) => s.pRaw) };
  } catch (e) { showMsg(e.message || String(e)); } finally { busy(false); }
}

async function runCompare() {
  showMsg(''); busy(true);
  const base = cfgFromUI();
  const runs = [
    { name: 'Greedy', cfg: { ...base, strategy: 'greedy' } },
    { name: `Beam search (width ${base.beams})`, cfg: { ...base, strategy: 'beam' } },
    { name: `Nucleus p = ${base.topp} (T = ${base.temp}, seed ${base.seed})`, cfg: { ...base, strategy: 'topp' } },
  ];
  const host = $('#cmpout'); host.innerHTML = '';
  const cards = runs.map((r) => { const c = el('div', { class: 'panel card' }, [el('h3', { text: r.name }), el('p', { class: 'note', text: 'Waiting…' })]); host.append(c); return c; });
  const results = [];
  try {
    for (let i = 0; i < runs.length && !abort; i++) {
      cards[i].querySelector('.note').textContent = 'Generating…';
      const res = await decode(runs[i].cfg, null);
      const m = metrics(res.steps);
      results.push({ name: runs[i].name, text: M.tokenizer.decode(res.steps.map((s) => s.tok)), ...m });
      const txt = el('div', { class: 'txt' }, [el('span', { class: 'prompt', text: M.tokenizer.decode(res.prompt) })]);
      const toks = res.steps.map((s) => s.tok);
      const inRep = new Array(toks.length).fill(false);
      for (let j = 0; j + 3 <= toks.length; j++) if (m.repSet.has(toks.slice(j, j + 3).join(','))) inRep[j] = inRep[j + 1] = inRep[j + 2] = true;
      toks.forEach((t, j) => txt.append(el('span', { class: inRep[j] ? 'rep' : null, text: decodeTok(t).replace(/\n/g, '↵ ') })));
      cards[i].replaceChildren(el('h3', { text: runs[i].name }), txt, el('dl', {}, [
        el('dt', { text: 'Mean log₂ P / token' }), el('dd', { text: m.meanLog2.toFixed(2) }),
        el('dt', { text: 'Perplexity' }), el('dd', { text: m.ppl.toFixed(1) }),
        el('dt', { text: 'Distinct bigrams' }), el('dd', { text: (m.distinct2 * 100).toFixed(0) + '%' }),
        el('dt', { text: 'In repeated 3-grams' }), el('dd', { text: (m.rep3 * 100).toFixed(0) + '%' }),
      ]));
      await tick();
    }
    window.__lab.compare = results.map(({ repSet, ...r }) => r);
  } catch (e) { showMsg(e.message || String(e)); } finally { busy(false); }
}

try {
  $('#prompts').append(...PROMPTS.map((p) => el('button', { class: 'chip', type: 'button', text: p, onclick: () => { $('#prompt').value = p; } })));
  for (const id of controls.concat(['strategy'])) $('#' + id).addEventListener('input', syncUI);
  syncUI();
  $('#gen').addEventListener('click', runOne);
  $('#cmp').addEventListener('click', runCompare);
  $('#stop').addEventListener('click', () => { abort = true; });
  renderStep();
  ready({ decode: (cfg) => decode({ ...cfgFromUI(), ...cfg }, null).then((r) => ({ tokens: r.steps.map((s) => s.tok), text: M.tokenizer.decode(r.steps.map((s) => s.tok)), probs: r.steps.map((s) => s.pRaw) })) });
} catch (e) { fail(e.message || e); }
