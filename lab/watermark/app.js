// Green-list ("soft red list") watermark of Kirchenbauer et al. (2023) on distilgpt2: keyed green lists,
// biased sampling with a KV cache, z-test detection, and an edit-attack robustness curve.
import { el, $, ready, fail, plotly, plotLayout, PLOT_CONFIG } from '/lab/shared/lab.js';
import { progressUI, transformers as T } from '/lab/shared/hf.js';

window.__lab = window.__lab || {};
const MODEL = 'Xenova/distilgpt2';
const V = 50257, EOS = 50256;
const NL = new Set([198, 628]); // GPT-2 tokens for "\n" and "\n\n"
/** Value of the k-th largest element (k ≥ 1). */
function kth(arr, k) { const top = new Float64Array(k).fill(-Infinity); for (let i = 0; i < arr.length; i++) { const v = arr[i]; if (v > top[k - 1]) { let j = k - 1; while (j > 0 && top[j - 1] < v) { top[j] = top[j - 1]; j--; } top[j] = v; } } return top[k - 1]; }
const PROMPTS = [
  'The town council met on Tuesday to discuss',
  'Scientists studying the river delta found that',
  'My favourite thing about autumn is',
  'Here is a list of the days of the week: Monday, Tuesday,',
];

// ---------- keyed green list ----------
function fmix32(h) { h ^= h >>> 16; h = Math.imul(h, 0x85ebca6b); h ^= h >>> 13; h = Math.imul(h, 0xc2b2ae35); h ^= h >>> 16; return h >>> 0; }
/** Seed for the green list that follows token `prev` (hash of key and previous token). */
const seedFor = (key, prev) => fmix32((Math.imul(key >>> 0, 0x9e3779b1) ^ fmix32(prev + 0x632be5ab)) >>> 0);
/** Token `tok` is green after `prev` iff a keyed hash falls below γ·2³². */
const isGreen = (seed, tok, gamma) => fmix32((seed ^ Math.imul(tok + 1, 0x27d4eb2d)) >>> 0) < gamma * 4294967296;

// ---------- statistics ----------
function erfc(x) { // Numerical Recipes erfcc, |error| < 1.2e-7
  const z = Math.abs(x), t = 1 / (1 + 0.5 * z);
  const r = t * Math.exp(-z * z - 1.26551223 + t * (1.00002368 + t * (0.37409196 + t * (0.09678418 + t * (-0.18628806 + t * (0.27886807 + t * (-1.13520398 + t * (1.48851587 + t * (-0.82215223 + t * 0.17087277)))))))));
  return x >= 0 ? r : 2 - r;
}
const pUpper = (z) => 0.5 * erfc(z / Math.SQRT2);
/** Score a token sequence. ids[0] is context only. Returns per-token flags and z. */
function detect(ids, key, gamma, { from = 1, dedup = false } = {}) {
  const flags = new Array(ids.length).fill(null); let G = 0, Tn = 0; const seen = new Set();
  for (let i = Math.max(1, from); i < ids.length; i++) {
    const pair = ids[i - 1] + ':' + ids[i];
    const g = isGreen(seedFor(key, ids[i - 1]), ids[i], gamma);
    flags[i] = g;
    if (dedup) { if (seen.has(pair)) { flags[i] = 'dup'; continue; } seen.add(pair); }
    Tn++; if (g) G++;
  }
  const z = Tn ? (G - gamma * Tn) / Math.sqrt(Tn * gamma * (1 - gamma)) : 0;
  return { flags, G, T: Tn, z, p: pUpper(z) };
}

// ---------- model ----------
let M = null, ui;
function getModel() {
  const cb = (p) => ui.onProgress(p);
  M ||= Promise.all([
    T.AutoTokenizer.from_pretrained(MODEL, { progress_callback: cb }),
    T.AutoModelForCausalLM.from_pretrained(MODEL, { dtype: 'q8', device: 'wasm', model_file_name: 'decoder_model_merged', progress_callback: cb }),
  ]).then(([tokenizer, model]) => { ui.done('Ready · distilgpt2 (8-bit, ≈85 MB, cached)'); return { tokenizer, model }; })
    .catch((e) => { M = null; ui.error(e); throw e; });
  return M;
}
const ids2t = (ids) => new T.Tensor('int64', BigInt64Array.from(ids.map(BigInt)), [1, ids.length]);
const ones = (n) => new T.Tensor('int64', new BigInt64Array(n).fill(1n), [1, n]);
const toPast = (o) => { const p = {}; for (const k of Object.keys(o)) if (k.startsWith('present.')) p[k.replace('present.', 'past_key_values.')] = o[k]; return p; };
function mulberry32(seed) { let a = seed >>> 0; return () => { a = (a + 0x6D2B79F5) >>> 0; let t = a; t = Math.imul(t ^ (t >>> 15), t | 1); t ^= t + Math.imul(t ^ (t >>> 7), t | 61); return ((t ^ (t >>> 14)) >>> 0) / 4294967296; }; }
const tick = () => new Promise((r) => setTimeout(r, 0));

/** Sample n tokens; if delta>0 add delta to green logits. Returns ids, prompt length, unbiased log-probs and entropies. */
async function generate(prompt, { n, gamma, delta, seed, key, topk }, onTok) {
  const { tokenizer, model } = await getModel();
  const pids = Array.from(tokenizer(prompt).input_ids.data, Number);
  const rand = mulberry32(seed);
  let o = await model({ input_ids: ids2t(pids), attention_mask: ones(pids.length) });
  let past = toPast(o), len = pids.length, logits = o.logits.data.slice((len - 1) * V, len * V);
  const ids = [...pids], lp = [], ent = [];
  for (let s = 0; s < n; s++) {
    // unbiased distribution (for perplexity and entropy)
    let m = -Infinity; for (let i = 0; i < V; i++) if (logits[i] > m) m = logits[i];
    let Z = 0; for (let i = 0; i < V; i++) Z += Math.exp(logits[i] - m);
    const lse = m + Math.log(Z);
    let H = 0; for (let i = 0; i < V; i++) { const lpi = logits[i] - lse; if (lpi > -30) H -= Math.exp(lpi) * lpi; }
    // biased sampling distribution
    const seedG = seedFor(key, ids[ids.length - 1]);
    const w = new Float64Array(V); let m2 = -Infinity;
    for (let i = 0; i < V; i++) { w[i] = logits[i] + (delta && isGreen(seedG, i, gamma) ? delta : 0); if (w[i] > m2) m2 = w[i]; }
    // Same for both runs: no end-of-text (fixed length) and no back-to-back line breaks (distilgpt2 otherwise loops on "\n").
    w[EOS] = -Infinity; if (NL.has(ids[ids.length - 1])) for (const t of NL) w[t] = -Infinity;
    if (topk > 0) { // keep the k largest biased logits (watermark bias is applied before truncation)
      const cut = kth(w, topk); for (let i = 0; i < V; i++) if (w[i] < cut) w[i] = -Infinity;
    }
    let Z2 = 0; for (let i = 0; i < V; i++) { w[i] = Math.exp(w[i] - m2); Z2 += w[i]; }
    let r = rand() * Z2, tok = V - 1;
    for (let i = 0; i < V; i++) { r -= w[i]; if (r < 0) { tok = i; break; } }
    ids.push(tok); lp.push(logits[tok] - lse); ent.push(H / Math.LN2);
    if (onTok) onTok(s + 1);

    o = await model({ input_ids: ids2t([tok]), attention_mask: ones(len + 1), past_key_values: past });
    past = toPast(o); len++; logits = o.logits.data;
    if (s % 4 === 3) await tick();
  }
  return { ids, plen: pids.length, lp, ent };
}

// ---------- rendering ----------
let tok = null; // tokenizer once loaded
const dec = (id) => tok.decode([id], { skip_special_tokens: false });
function renderTokens(host, ids, plen, flags, attacked = new Set()) {
  host.replaceChildren();
  if (plen > 0) host.append(el('span', { class: 'prompt', text: tok.decode(ids.slice(0, plen)) }));
  for (let i = plen; i < ids.length; i++) {
    const f = flags[i], cls = f === true ? 'g' : f === false ? 'r' : 'x';
    host.append(el('span', { class: `t ${cls}${attacked.has(i) ? ' att' : ''}`, title: f === 'dup' ? 'repeated pair, not scored' : f === true ? 'green' : f === false ? 'red' : '', text: ids[i] === EOS ? '⟨eos⟩' : dec(ids[i]) }));
  }
}
const fmtP = (p) => (p < 1e-12 ? '< 1e-12' : p < 1e-3 ? p.toExponential(1) : p.toFixed(3));
function renderMetrics(host, d, extra = []) {
  host.replaceChildren(...[[d.T, 'tokens scored'], [d.G, 'green'], [(d.T ? (100 * d.G / d.T).toFixed(0) : 0) + '%', 'green fraction'], [d.z.toFixed(2), 'z-score'], [fmtP(d.p), 'p-value (one-sided)'], ...extra]
    .map(([b, s]) => el('div', {}, [el('b', { text: String(b) }), el('span', { text: s })])));
}
function setVerdict(node, z) { node.className = 'verdict ' + (z > 4 ? 'yes' : 'no'); node.textContent = z > 4 ? 'detected · z > 4' : 'not detected'; }
const ppl = (lp) => Math.exp(-lp.reduce((a, b) => a + b, 0) / (lp.length || 1));
const mean = (xs) => xs.reduce((a, b) => a + b, 0) / (xs.length || 1);

const S = { busy: false, wm: null };
const cfg = () => ({ gamma: +$('#gamma').value, delta: +$('#delta').value, n: +$('#len').value, seed: +$('#seed').value || 0, key: +$('#key').value >>> 0, topk: +$('#samp').value, dedup: $('#dedup').checked });
function showMsg(t) { const m = $('#msg'); m.hidden = !t; m.textContent = t || ''; }

async function runGen() {
  if (S.busy) return;
  const prompt = $('#prompt').value.trim();
  if (!prompt) return void showMsg('Type a prompt first.');
  showMsg(''); S.busy = true; $('#gen').disabled = true;
  const c = cfg();
  try {
    $('#status').textContent = 'Loading model…';
    await getModel(); tok = (await getModel()).tokenizer;
    const t0 = performance.now();
    const wm = await generate(prompt, c, (k) => { $('#status').textContent = `Watermarked: token ${k} of ${c.n}…`; });
    const pl = await generate(prompt, { ...c, delta: 0 }, (k) => { $('#status').textContent = `Plain: token ${k} of ${c.n}…`; });
    const dw = detect(wm.ids, c.key, c.gamma, { from: wm.plen, dedup: c.dedup }), dp = detect(pl.ids, c.key, c.gamma, { from: pl.plen, dedup: c.dedup });
    renderTokens($('#out-wm'), wm.ids, wm.plen, dw.flags); renderTokens($('#out-pl'), pl.ids, pl.plen, dp.flags);
    renderMetrics($('#m-wm'), dw, [[ppl(wm.lp).toFixed(1), 'perplexity (unbiased model)']]);
    renderMetrics($('#m-pl'), dp, [[ppl(pl.lp).toFixed(1), 'perplexity (unbiased model)']]);
    setVerdict($('#v-wm'), dw.z); setVerdict($('#v-pl'), dp.z);
    $('#log tbody').prepend(el('tr', {}, [c.gamma.toFixed(2), c.delta.toFixed(1), dw.T, dw.z.toFixed(2), dp.z.toFixed(2), ppl(wm.lp).toFixed(1), ppl(pl.lp).toFixed(1), mean(wm.ent).toFixed(2)].map((x) => el('td', { text: String(x) }))));
    S.wm = { ...wm, c }; S.pl = pl;
    window.__lab.last = { zWM: dw.z, zPlain: dp.z, greenWM: dw.G / dw.T, greenPlain: dp.G / dp.T, pplWM: ppl(wm.lp), pplPlain: ppl(pl.lp), tokens: dw.T, entropy: mean(wm.ent), text: tok.decode(wm.ids.slice(wm.plen)) };
    $('#status').textContent = `Generated ${dw.T + dp.T} tokens in ${((performance.now() - t0) / 1000).toFixed(1)} s.`;
    await renderAttack(true);
  } catch (e) {
    showMsg('Something went wrong: ' + (e.message || e)); window.__lab.runError = String(e.message || e);
  } finally { S.busy = false; $('#gen').disabled = false; }
}

/** Re-score the last generation (e.g. after toggling dedup) without regenerating. */
function rescore() {
  if (!S.wm) return;
  const c = { ...S.wm.c, dedup: $('#dedup').checked };
  for (const [r, out, m, v] of [[S.wm, '#out-wm', '#m-wm', '#v-wm'], [S.pl, '#out-pl', '#m-pl', '#v-pl']]) {
    const d = detect(r.ids, c.key, c.gamma, { from: r.plen, dedup: c.dedup });
    renderTokens($(out), r.ids, r.plen, d.flags); renderMetrics($(m), d, [[ppl(r.lp).toFixed(1), 'perplexity (unbiased model)']]); setVerdict($(v), d.z);
  }
}

// ---------- attack ----------
function attack(ids, plen, eps, type, rand) {
  const out = ids.slice(0, plen), hit = new Set();
  for (let i = plen; i < ids.length; i++) {
    if (rand() < eps) {
      if (type === 'del') continue;
      let r; do { r = Math.floor(rand() * EOS); } while (r === ids[i]);
      hit.add(out.length); out.push(r);
    } else out.push(ids[i]);
  }
  return { ids: out, hit };
}
async function renderAttack(redrawChart) {
  if (!S.wm) return;
  const { ids, plen, c } = S.wm, eps = +$('#eps').value, type = $('#att').value;
  const a = attack(ids, plen, eps, type, mulberry32(12345));
  const d = detect(a.ids, c.key, c.gamma, { from: plen, dedup: $('#dedup').checked });
  renderTokens($('#out-att'), a.ids, plen, d.flags, a.hit);
  renderMetrics($('#m-att'), d);
  window.__lab.attack = { eps, type, z: d.z };
  if (!redrawChart) return;
  const xs = [], ysub = [], ydel = [];
  for (let e = 0; e <= 0.6001; e += 0.05) {
    xs.push(+e.toFixed(2));
    for (const [tp, arr] of [['sub', ysub], ['del', ydel]]) {
      const r = mulberry32(999 + Math.round(e * 100)); let s = 0;
      for (let k = 0; k < 30; k++) s += detect(attack(ids, plen, e, tp, r).ids, c.key, c.gamma, { from: plen, dedup: $('#dedup').checked }).z;
      arr.push(s / 30);
    }
  }
  window.__lab.curve = { eps: xs, sub: ysub, del: ydel };
  try {
    const P = await plotly();
    P.react('zchart', [
      { x: xs, y: ysub, name: 'substitute', mode: 'lines+markers', line: { color: '#7ea3c8' } },
      { x: xs, y: ydel, name: 'delete', mode: 'lines+markers', line: { color: '#d8b66a' } },
      { x: [0, 0.6], y: [4, 4], name: 'threshold z = 4', mode: 'lines', line: { color: '#d98c7a', dash: 'dash', width: 1 } },
    ], plotLayout({ xaxis: { title: { text: 'edit rate ε' }, gridcolor: 'rgba(228,231,235,.08)', color: '#9aa3ad' }, yaxis: { title: { text: 'mean z-score' }, gridcolor: 'rgba(228,231,235,.08)', color: '#9aa3ad', zerolinecolor: 'rgba(228,231,235,.22)' }, legend: { orientation: 'h', y: 1.12, bgcolor: 'rgba(0,0,0,0)' }, margin: { l: 52, r: 12, t: 30, b: 48 } }), PLOT_CONFIG);
  } catch { /* chart is optional */ }
}

// ---------- detect pasted text ----------
async function runDetect() {
  const text = $('#det-text').value;
  if (!text.trim()) return;
  $('#det').disabled = true;
  try {
    if (!tok) tok = await T.AutoTokenizer.from_pretrained(MODEL);
    const c = cfg();
    const ids = Array.from(tok(text).input_ids.data, Number);
    const d = detect(ids, c.key, c.gamma, { dedup: $('#dedup').checked });
    renderTokens($('#out-det'), ids, 1, d.flags);
    $('#out-det').firstChild.title = 'first token: no predecessor, not scored';
    renderMetrics($('#m-det'), d);
    window.__lab.detect = { z: d.z, T: d.T, G: d.G };
  } catch (e) { $('#out-det').textContent = 'Couldn’t tokenize: ' + (e.message || e); }
  finally { $('#det').disabled = false; }
}

try {
  ui = progressUI($('#load'), 'distilgpt2');
  ui.set('Model: distilgpt2 (82M parameters, 8-bit) · ≈85 MB download on first use, then cached', 0);
  $('#prompts').replaceChildren(...PROMPTS.map((p) => el('button', { class: 'chip', type: 'button', text: p, onclick: () => { $('#prompt').value = p; runGen(); } })));
  for (const [id, f] of [['gamma', (v) => (+v).toFixed(2)], ['delta', (v) => (+v).toFixed(1)], ['len', (v) => v], ['eps', (v) => (+v).toFixed(2)]]) {
    $('#' + id).addEventListener('input', () => { $(`#${id}-val`).textContent = f($('#' + id).value); });
  }
  $('#gen').addEventListener('click', runGen);
  $('#eps').addEventListener('input', () => renderAttack(false));
  $('#att').addEventListener('change', () => renderAttack(false));
  $('#det').addEventListener('click', runDetect);
  $('#dedup').addEventListener('change', () => { rescore(); renderAttack(true); if ($('#out-det').querySelector('.t')) runDetect(); });
  $('#det-copy').addEventListener('click', () => { if (window.__lab.last) { $('#det-text').value = window.__lab.last.text; runDetect(); } });
  window.__lab.detectIds = (ids, key, gamma) => detect(ids, key, gamma);
  window.__lab.runGen = runGen;
  ready();
} catch (e) { fail(e); }
