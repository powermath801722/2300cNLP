// Speculative sampling (Leviathan et al. 2023; Chen et al. 2023): distilgpt2 drafts, GPT-2 verifies.
// Both models keep KV caches; rejected drafts are rolled back by truncating the caches.
import { el, $, ready, fail, plotly, plotLayout, PLOT_CONFIG } from '/lab/shared/lab.js';
import { progressUI, transformers as T } from '/lab/shared/hf.js';

window.__lab = window.__lab || {};
const DRAFT = 'Xenova/distilgpt2', TARGET = 'Xenova/gpt2';
const V = 50257, EOS = 50256;
const PROMPTS = ['The history of the printing press begins', 'In the morning, the fishermen', 'The theory of evolution by natural selection', 'Once upon a time, in a small village by the sea,'];

// ---------- models ----------
const _m = {};
function load(id, ui, label) {
  const cb = (p) => ui.onProgress(p);
  return (_m[id] ||= Promise.all([
    T.AutoTokenizer.from_pretrained(id, { progress_callback: cb }),
    T.AutoModelForCausalLM.from_pretrained(id, { dtype: 'q8', device: 'wasm', model_file_name: 'decoder_model_merged', progress_callback: cb }),
  ]).then(([tokenizer, model]) => { ui.done(`Ready · ${label} (8-bit, cached)`); return { tokenizer, model }; })
    .catch((e) => { delete _m[id]; ui.error(e); throw e; }));
}
const ids2t = (ids) => new T.Tensor('int64', BigInt64Array.from(ids.map(BigInt)), [1, ids.length]);
const ones = (n) => new T.Tensor('int64', new BigInt64Array(n).fill(1n), [1, n]);
const toPast = (o) => { const p = {}; for (const k of Object.keys(o)) if (k.startsWith('present.')) p[k.replace('present.', 'past_key_values.')] = o[k]; return p; };
/** Keep only the first L cached positions of every key/value tensor ([1, H, len, D]). */
function truncPast(past, L) {
  const out = {};
  for (const [k, t] of Object.entries(past)) {
    const [B, H, len, D] = t.dims;
    if (L === len) { out[k] = t; continue; }
    const data = new t.data.constructor(B * H * L * D);
    for (let h = 0; h < B * H; h++) data.set(t.data.subarray(h * len * D, h * len * D + L * D), h * L * D);
    out[k] = new T.Tensor(t.type, data, [B, H, L, D]);
  }
  return out;
}
/** Feed tokens after the cached prefix. Returns per-position logits rows and the new cache state. Counts calls. */
async function feed(model, st, tokens, stats, key) {
  const t0 = performance.now();
  const args = { input_ids: ids2t(tokens), attention_mask: ones(st.len + tokens.length) };
  if (st.past) args.past_key_values = st.past;
  const o = await model(args);
  stats[key + 'Calls']++; stats[key + 'Ms'] += performance.now() - t0; stats[key + 'Pos'] += tokens.length;
  const d = o.logits.data, rows = [];
  for (let i = 0; i < tokens.length; i++) rows.push(d.subarray(i * V, (i + 1) * V));
  return { rows, st: { past: toPast(o), len: st.len + tokens.length } };
}
/** Empty-cache state: the merged decoder accepts a first call without past_key_values. */
const EMPTY = { past: null, len: 0 };

// ---------- distributions & sampling ----------
function mulberry32(seed) { let a = seed >>> 0; return () => { a = (a + 0x6D2B79F5) >>> 0; let t = a; t = Math.imul(t ^ (t >>> 15), t | 1); t ^= t + Math.imul(t ^ (t >>> 7), t | 61); return ((t ^ (t >>> 14)) >>> 0) / 4294967296; }; }
const argmax = (a) => { let b = 0; for (let i = 1; i < a.length; i++) if (a[i] > a[b]) b = i; return b; };
/** Probabilities at temperature temp (temp 0 → one-hot argmax). EOS is blocked so runs have a fixed length. */
function probs(logits, temp) {
  const p = new Float64Array(V);
  if (temp === 0) { let b = 0; for (let i = 1; i < V; i++) if (i !== EOS && logits[i] > logits[b]) b = i; p[b] = 1; return p; }
  let m = -Infinity; for (let i = 0; i < V; i++) if (i !== EOS && logits[i] > m) m = logits[i];
  let s = 0; for (let i = 0; i < V; i++) { const v = i === EOS ? 0 : Math.exp((logits[i] - m) / temp); p[i] = v; s += v; }
  for (let i = 0; i < V; i++) p[i] /= s;
  return p;
}
function sample(p, rand) { let r = rand(), c = 0; for (let i = 0; i < p.length; i++) { c += p[i]; if (r < c) return i; } return argmax(p); }
function residual(p, q) { const r = new Float64Array(V); let s = 0; for (let i = 0; i < V; i++) { const v = Math.max(0, p[i] - q[i]); r[i] = v; s += v; } if (s <= 0) return p; for (let i = 0; i < V; i++) r[i] /= s; return r; }
/** Gap between the two largest logits (how close a greedy choice was to a tie). */
function margin(row) { let a = -Infinity, b = -Infinity; for (let i = 0; i < V; i++) { if (i === EOS) continue; const v = row[i]; if (v > a) { b = a; a = v; } else if (v > b) b = v; } return a - b; }
const overlap = (p, q) => { let s = 0; for (let i = 0; i < V; i++) s += Math.min(p[i], q[i]); return s; };
const tick = () => new Promise((r) => setTimeout(r, 0));

// ---------- speculative decoding ----------
async function speculative(D, Tm, promptIds, { gamma, temp, n, seed }, onRound) {
  const rand = mulberry32(seed), stats = { dCalls: 0, dMs: 0, dPos: 0, tCalls: 0, tMs: 0, tPos: 0 };
  // Caches hold everything except the pending tokens, which are fed at the start of the next call.
  const pre = promptIds.slice(0, -1);
  let dSt = pre.length ? (await feed(D, EMPTY, pre, stats, 'd')).st : EMPTY;
  let tSt = pre.length ? (await feed(Tm, EMPTY, pre, stats, 't')).st : EMPTY;
  stats.dCalls = stats.tCalls = 0; stats.dMs = stats.tMs = 0; stats.dPos = stats.tPos = 0; // count decoding only (prefill shared with baseline)
  let dPend = [promptIds.at(-1)], tPend = [promptIds.at(-1)];
  const out = [], origin = [], rounds = [], betas = [], margins = [];
  while (out.length < n) {
    // 1. draft γ tokens
    const drafts = [], qs = [];
    let r = await feed(D, dSt, dPend, stats, 'd'); dSt = r.st;
    let q = probs(r.rows.at(-1), temp);
    for (let i = 0; i < gamma; i++) {
      const x = temp === 0 ? argmax(q) : sample(q, rand);
      drafts.push(x); qs.push(q);
      if (i < gamma - 1) { r = await feed(D, dSt, [x], stats, 'd'); dSt = r.st; q = probs(r.rows[0], temp); }
    }
    // 2. one target pass over pending + drafts → p_1 … p_{γ+1}
    const base = tSt.len; // cache length before this round
    const tr = await feed(Tm, tSt, [...tPend, ...drafts], stats, 't');
    const off = tPend.length - 1, ps = [];
    for (let i = 0; i <= gamma; i++) ps.push(probs(tr.rows[off + i], temp));
    // 3. accept / reject
    let a = 0, extra, kind;
    for (; a < gamma; a++) {
      const x = drafts[a], pi = ps[a], qi = qs[a];
      betas.push(overlap(pi, qi));
      const ok = temp === 0 ? argmax(pi) === x : rand() < Math.min(1, pi[x] / qi[x]);
      if (!ok) break;
    }
    if (a < gamma) { extra = temp === 0 ? argmax(ps[a]) : sample(residual(ps[a], qs[a]), rand); kind = 'res'; }
    else { extra = temp === 0 ? argmax(ps[gamma]) : sample(ps[gamma], rand); kind = 'bon'; }
    const accepted = drafts.slice(0, a);
    // 4. roll caches back to the accepted prefix; the new token becomes pending
    const keepT = base + tPend.length + a; // cache now covers prefix + pending + accepted drafts
    tSt = { past: truncPast(tr.st.past, keepT), len: keepT }; tPend = [extra];
    const dCovered = dSt.len, dKeep = Math.min(dCovered, (dCovered - (gamma - 1)) + a); // draft cache covers prefix + pending + d1..d_{γ-1}
    dSt = { past: truncPast(dSt.past, dKeep), len: dKeep };
    dPend = a === gamma ? [drafts[gamma - 1], extra] : [extra];
    for (let i = 0; i < a; i++) { out.push(accepted[i]); origin.push('d'); margins.push(margin(tr.rows[off + i])); }
    out.push(extra); origin.push(kind === 'res' ? 't' : 'b'); margins.push(margin(tr.rows[off + a]));
    rounds.push({ drafts, a, extra, kind });
    if (onRound) onRound(rounds.length, out.length);
    await tick();
  }
  return { ids: out.slice(0, n), origin: origin.slice(0, n), margins: margins.slice(0, n), rounds, stats, alpha: betas.reduce((x, y) => x + y, 0) / (betas.length || 1) };
}

async function plain(Tm, promptIds, { temp, n, seed }, onTok) {
  const rand = mulberry32(seed + 1000003), stats = { tCalls: 0, tMs: 0, tPos: 0, dCalls: 0, dMs: 0, dPos: 0 };
  const pre = promptIds.slice(0, -1);
  let st = pre.length ? (await feed(Tm, EMPTY, pre, stats, 't')).st : EMPTY;
  stats.tCalls = 0; stats.tMs = 0; stats.tPos = 0;
  let pend = [promptIds.at(-1)]; const out = [], margins = [];
  while (out.length < n) {
    const r = await feed(Tm, st, pend, stats, 't'); st = r.st;
    const p = probs(r.rows.at(-1), temp);
    const x = temp === 0 ? argmax(p) : sample(p, rand);
    out.push(x); margins.push(margin(r.rows.at(-1))); pend = [x];
    if (onTok) onTok(out.length);
    if (out.length % 4 === 0) await tick();
  }
  return { ids: out, margins, stats };
}

// ---------- UI ----------
const S = { busy: false };
let uiD, uiT;
const cfg = () => ({ gamma: +$('#gamma').value, temp: +$('#temp').value, n: +$('#len').value, seed: +$('#seed').value || 0 });
function showMsg(t) { const m = $('#msg'); m.hidden = !t; m.textContent = t || ''; }
const metric = (pairs) => pairs.map(([b, s]) => el('div', {}, [el('b', { text: String(b) }), el('span', { text: s })]));

async function run() {
  if (S.busy) return;
  const prompt = $('#prompt').value.trim();
  if (!prompt) return void showMsg('Type a prompt first.');
  showMsg(''); S.busy = true; $('#run').disabled = $('#verify').disabled = true;
  const c = cfg();
  try {
    $('#status').textContent = 'Loading models…';
    const [D, Tg] = await Promise.all([load(DRAFT, uiD, 'distilgpt2, draft'), load(TARGET, uiT, 'GPT-2, target')]);
    const tok = Tg.tokenizer, pids = Array.from(tok(prompt).input_ids.data, Number);
    const dec = (id) => tok.decode([id]);
    const spec = await speculative(D.model, Tg.model, pids, c, (r, k) => { $('#status').textContent = `Speculative: round ${r}, ${Math.min(k, c.n)} of ${c.n} tokens…`; });
    const base = await plain(Tg.model, pids, c, (k) => { $('#status').textContent = `GPT-2 alone: token ${k} of ${c.n}…`; });
    // output
    const o = $('#out-spec'); o.replaceChildren(el('span', { class: 'prompt', text: prompt }));
    spec.ids.forEach((id, i) => o.append(el('span', { class: 'tk ' + spec.origin[i], text: dec(id) })));
    $('#out-plain').replaceChildren(el('span', { class: 'prompt', text: prompt }), el('span', { text: tok.decode(base.ids) }));
    $('#rounds').replaceChildren(...spec.rounds.map((r, i) => el('li', {}, [
      el('span', { class: 'n', text: '#' + (i + 1) }),
      el('span', { class: 'toks' }, [...r.drafts.map((x, j) => el('span', { class: 'chip2 ' + (j < r.a ? 'acc' : j === r.a ? 'rej' : 'skip'), title: j < r.a ? 'accepted' : j === r.a ? 'rejected' : 'discarded (after a rejection)', text: dec(x) })), el('span', { class: 'chip2 ' + r.kind, title: r.kind === 'res' ? 'resampled from the residual max(0, p − q)' : 'bonus token sampled from p', text: dec(r.extra) })]),
      el('span', { class: 'sum', text: `${r.a}/${r.drafts.length} + 1` }),
    ])));
    const st = spec.stats, tpr = c.n / spec.rounds.length;
    const cRatio = (st.dMs / st.dCalls) / (st.tMs / st.tCalls);
    const accRate = spec.rounds.reduce((s, r) => s + r.a, 0) / spec.rounds.reduce((s, r) => s + r.drafts.length, 0);
    const theory = (1 - spec.alpha ** (c.gamma + 1)) / (1 - spec.alpha);
    const specMs = st.dMs + st.tMs, baseMs = base.stats.tMs;
    $('#m-spec').replaceChildren(...metric([
      [st.tCalls, 'GPT-2 calls'], [st.dCalls, 'distilgpt2 calls'], [tpr.toFixed(2), 'tokens per GPT-2 call'], [theory.toFixed(2), 'predicted (1−α^(γ+1))/(1−α)'],
      [(accRate * 100).toFixed(0) + '%', 'drafts accepted'], [spec.alpha.toFixed(3), 'α = E[Σ min(p, q)]'], [(specMs / 1000).toFixed(2) + ' s', 'model time'], [(baseMs / specMs).toFixed(2) + '×', 'speed-up vs GPT-2 alone'],
    ]));
    $('#m-plain').replaceChildren(...metric([[base.stats.tCalls, 'GPT-2 calls'], ['1.00', 'tokens per call'], [(baseMs / 1000).toFixed(2) + ' s', 'model time'], [cRatio.toFixed(2), 'cost ratio c (draft ÷ target call)']]));
    let same = null, firstDiff = -1, gap = null;
    if (c.temp === 0) {
      firstDiff = spec.ids.findIndex((x, i) => x !== base.ids[i]); same = firstDiff < 0;
      if (!same) gap = Math.min(spec.margins[firstDiff], base.margins[firstDiff]);
    }
    const nearTie = gap != null && gap < 0.5;
    $('#check').replaceChildren(c.temp === 0
      ? same
        ? el('span', {}, [el('b', { text: '✓ Identical to GPT-2 greedy. ' }), 'At temperature 0 speculative decoding reproduced GPT-2’s greedy output token for token.'])
        : el('span', {}, [el('b', { class: nearTie ? '' : 'bad', text: `${nearTie ? '≈' : '✗'} Identical for the first ${firstDiff} tokens. ` }),
          `At token ${firstDiff + 1} GPT-2’s top two candidates differ by only ${gap.toFixed(2)} logits. The model is quantized to 8 bits with activation scales computed per forward pass, so a ${c.gamma + 1}-token verification pass and a 1-token pass can break such a near-tie differently; after that the two greedy texts follow different paths. In exact arithmetic they would match.`])
      : el('span', { class: 'note', text: 'With sampling the two runs use different random draws, so the texts differ; the guarantee is about the distribution, checked in section 02.' }));
    await sweep(spec.alpha, cRatio, c.gamma, tpr);
    window.__lab.last = { gamma: c.gamma, temp: c.temp, n: c.n, targetCalls: st.tCalls, draftCalls: st.dCalls, baselineCalls: base.stats.tCalls, tokensPerCall: tpr, predicted: theory, alpha: spec.alpha, acceptRate: accRate, speedup: baseMs / specMs, costRatio: cRatio, greedyMatch: same, firstDiff, tieGap: gap };
    $('#status').textContent = `Done: ${c.n} tokens in ${st.tCalls} GPT-2 calls (GPT-2 alone needed ${base.stats.tCalls}).`;
  } catch (e) {
    showMsg('Something went wrong: ' + (e.message || e)); window.__lab.runError = String(e.message || e);
  } finally { S.busy = false; $('#run').disabled = $('#verify').disabled = false; }
}

async function sweep(alpha, cRatio, gamma, measured) {
  const gs = [1, 2, 3, 4, 5, 6, 7, 8, 10, 12];
  const tokens = gs.map((g) => (1 - alpha ** (g + 1)) / (1 - alpha));
  const speed = gs.map((g, i) => tokens[i] / (g * cRatio + 1));
  try {
    const P = await plotly();
    const ax = { gridcolor: 'rgba(228,231,235,.08)', color: '#9aa3ad', zerolinecolor: 'rgba(228,231,235,.22)' };
    P.react('sweep', [
      { x: gs, y: tokens, name: 'expected tokens per GPT-2 call', mode: 'lines+markers', line: { color: '#8fbf9a' } },
      { x: gs, y: speed, name: `expected speed-up (c = ${cRatio.toFixed(2)})`, mode: 'lines+markers', line: { color: '#7ea3c8' } },
      { x: [gamma], y: [measured], name: 'measured tokens per call', mode: 'markers', marker: { color: '#d8b66a', size: 12, symbol: 'diamond' } },
      { x: [1, 12], y: [1, 1], name: 'break-even', mode: 'lines', line: { color: '#d98c7a', dash: 'dash', width: 1 } },
    ], plotLayout({ xaxis: { ...ax, title: { text: 'draft length γ' } }, yaxis: { ...ax, title: { text: `α = ${alpha.toFixed(2)}` } }, legend: { orientation: 'h', y: 1.18, bgcolor: 'rgba(0,0,0,0)' }, margin: { l: 52, r: 12, t: 40, b: 48 } }), PLOT_CONFIG);
  } catch { /* optional chart */ }
}

async function verify() {
  if (S.busy) return;
  const prompt = $('#prompt').value.trim(); if (!prompt) return;
  S.busy = true; $('#run').disabled = $('#verify').disabled = true;
  const temp = Math.max(0.1, +$('#temp').value), N = +$('#trials').value;
  try {
    $('#vstatus').textContent = 'Loading models…';
    const [D, Tg] = await Promise.all([load(DRAFT, uiD, 'distilgpt2, draft'), load(TARGET, uiT, 'GPT-2, target')]);
    const tok = Tg.tokenizer, pids = Array.from(tok(prompt).input_ids.data, Number);
    const z = { dCalls: 0, dMs: 0, dPos: 0, tCalls: 0, tMs: 0, tPos: 0 };
    const q = probs((await feed(D.model, EMPTY, pids, z, 'd')).rows.at(-1), temp);
    const p = probs((await feed(Tg.model, EMPTY, pids, z, 't')).rows.at(-1), temp);
    const res = residual(p, q), rand = mulberry32(2024), counts = new Map();
    // cumulative tables for fast sampling
    const cdf = (d) => { const c = new Float64Array(V); let s = 0; for (let i = 0; i < V; i++) { s += d[i]; c[i] = s; } return c; };
    const cq = cdf(q), cr = cdf(res);
    const draw = (c) => { const r = rand() * c[V - 1]; let lo = 0, hi = V - 1; while (lo < hi) { const m = (lo + hi) >> 1; if (c[m] < r) lo = m + 1; else hi = m; } return lo; };
    let accepted = 0;
    for (let t = 0; t < N; t++) {
      const x = draw(cq);
      let y;
      if (rand() < Math.min(1, p[x] / q[x])) { y = x; accepted++; } else y = draw(cr);
      counts.set(y, (counts.get(y) || 0) + 1);
      if (t % 50000 === 49999) { $('#vstatus').textContent = `${t + 1} trials…`; await tick(); }
    }
    let tvSpec = 0, tvQ = 0;
    for (let i = 0; i < V; i++) { tvSpec += Math.abs((counts.get(i) || 0) / N - p[i]); tvQ += Math.abs(q[i] - p[i]); }
    tvSpec /= 2; tvQ /= 2;
    // expected TV of an empirical distribution from N samples of p (for scale): ≈ Σ sqrt(p(1-p)/(2πN))
    let tvNoise = 0; for (let i = 0; i < V; i++) if (p[i] > 1e-7) tvNoise += Math.sqrt(p[i] * (1 - p[i]) / (2 * Math.PI * N));
    const top = [...Array(V).keys()].sort((a, b) => p[b] - p[a]).slice(0, 15);
    const labels = top.map((i) => JSON.stringify(tok.decode([i])).slice(1, -1));
    $('#m-ver').replaceChildren(...metric([[N.toLocaleString('en-US'), 'simulated trials'], [(accepted / N * 100).toFixed(1) + '%', `drafts accepted (theory ${(overlap(p, q) * 100).toFixed(1)}%)`], [tvSpec.toFixed(4), 'TV(speculative, p)'], [tvNoise.toFixed(4), 'expected TV from sampling noise alone'], [tvQ.toFixed(4), 'TV(q, p): naive draft-only']]));
    const P = await plotly();
    const ax = { gridcolor: 'rgba(228,231,235,.08)', color: '#9aa3ad' };
    P.react('hist', [
      { x: labels, y: top.map((i) => p[i]), name: 'GPT-2 p (exact)', type: 'bar', marker: { color: '#7ea3c8' } },
      { x: labels, y: top.map((i) => (counts.get(i) || 0) / N), name: 'speculative (empirical)', type: 'bar', marker: { color: '#8fbf9a' } },
      { x: labels, y: top.map((i) => q[i]), name: 'distilgpt2 q (exact)', type: 'bar', marker: { color: '#d8b66a' } },
    ], plotLayout({ barmode: 'group', xaxis: { ...ax, tickangle: -40 }, yaxis: { ...ax, title: { text: 'probability' } }, legend: { orientation: 'h', y: 1.15, bgcolor: 'rgba(0,0,0,0)' }, margin: { l: 56, r: 12, t: 40, b: 80 } }), PLOT_CONFIG);
    window.__lab.verify = { N, tvSpec, tvQ, tvNoise, acceptEmp: accepted / N, acceptTheory: overlap(p, q) };
    $('#vstatus').textContent = `Temperature ${temp.toFixed(1)}. Speculative output is ${tvSpec <= 2 * tvNoise ? 'within sampling noise of' : 'farther than expected from'} GPT-2’s distribution; the draft model alone is ${(tvQ / Math.max(tvSpec, 1e-9)).toFixed(0)}× farther.`;
  } catch (e) {
    $('#vstatus').textContent = 'Check failed: ' + (e.message || e); window.__lab.runError = String(e.message || e);
  } finally { S.busy = false; $('#run').disabled = $('#verify').disabled = false; }
}

try {
  uiD = progressUI($('#load-d'), 'distilgpt2'); uiD.set('Draft: distilgpt2 (82M parameters) · ≈85 MB, loads on first run', 0);
  uiT = progressUI($('#load-t'), 'GPT-2'); uiT.set('Target: GPT-2 small (124M parameters) · ≈128 MB, loads on first run', 0);
  $('#prompts').replaceChildren(...PROMPTS.map((p) => el('button', { class: 'chip', type: 'button', text: p, onclick: () => { $('#prompt').value = p; run(); } })));
  for (const [id, f] of [['gamma', (v) => v], ['temp', (v) => (+v).toFixed(1)], ['len', (v) => v]]) $('#' + id).addEventListener('input', () => { $(`#${id}-val`).textContent = f($('#' + id).value); });
  $('#run').addEventListener('click', run);
  $('#verify').addEventListener('click', verify);
  window.__lab.run = run; window.__lab.verifyDist = verify;
  ready();
} catch (e) { fail(e); }
