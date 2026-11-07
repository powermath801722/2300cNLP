import { el, $, ready, fail } from '/lab/shared/lab.js';
import { pipe, progressUI, transformers as T } from '/lab/shared/hf.js';

const LANGS = { fr: 'French', de: 'German', es: 'Spanish' };
const fwdModel = (l) => `Xenova/opus-mt-en-${l}`;
const revModel = (l) => `Xenova/opus-mt-${l}-en`;

// Reference translations written for this lab (not the model's output).
const SAMPLES = [
  { label: 'River', src: 'The river carries sediment to the sea.', ref: { fr: 'La rivière transporte des sédiments jusqu’à la mer.', de: 'Der Fluss trägt Sedimente ins Meer.', es: 'El río transporta sedimentos hasta el mar.' }, note: 'A plain sentence. Compare greedy with beam 4: they often agree on short, easy input.' },
  { label: 'Language models', src: 'Language models learn patterns from large amounts of text.', ref: { fr: 'Les modèles de langage apprennent des régularités à partir de grandes quantités de texte.', de: 'Sprachmodelle lernen Muster aus großen Textmengen.', es: 'Los modelos de lenguaje aprenden patrones a partir de grandes cantidades de texto.' }, note: 'Watch BLEU: a correct translation with different word choices than the reference still loses points.' },
  { label: 'Idiom', src: 'It’s raining cats and dogs, so we will stay at home.', ref: { fr: 'Il pleut des cordes, alors nous allons rester à la maison.', de: 'Es regnet in Strömen, also bleiben wir zu Hause.', es: 'Está lloviendo a cántaros, así que nos quedaremos en casa.' }, note: 'Idioms: does the model translate the meaning (heavy rain) or the words (cats and dogs)?' },
  { label: 'Which bank?', src: 'She sat on the bank and watched the boats go by.', ref: { fr: 'Elle s’est assise sur la berge et a regardé passer les bateaux.', de: 'Sie saß am Ufer und sah den Booten zu.', es: 'Se sentó en la orilla y vio pasar los barcos.' }, note: 'Lexical ambiguity: river bank or financial bank? The context word “boats” should decide.' },
  { label: 'Gender', src: 'The doctor told the nurse that she would be late.', ref: null, note: 'French, German and Spanish mark gender on nouns. Who is “she”? Systems tend to follow stereotypes (Stanovsky et al., 2019). No single reference is right here, so BLEU is skipped.' },
  { label: 'Long sentence', src: 'When the tide turns, the salt water pushes inland along the bottom of the estuary while the lighter fresh water continues to flow out toward the sea above it.', ref: null, note: 'Longer input: try α = 0 with beam 5 to see the short-output bias that length normalization corrects.' },
];

const ui = progressUI($('#load'), 'MarianMT');
ui.set('Model not loaded yet · ≈107 MB per language direction on first run, then cached', 0);
let busy = false, last = null;

const load = (id, label) => pipe('translation', id, { ui, label, readyText: `Model ready · ${label} (cached for next time)` });

const i64 = (arr, dims) => new T.Tensor('int64', BigInt64Array.from(arr.map((x) => BigInt(x))), dims);

/** Beam search (beam=1 is greedy) over raw decoder logits. No KV cache: each step re-runs the batch, which is fine for sentences. */
export async function decode(pipeObj, text, { beams = 4, alpha = 1, onStep } = {}) {
  const { model, tokenizer: tok } = pipeObj;
  const cfg = model.config;
  const START = cfg.decoder_start_token_id, EOS = cfg.eos_token_id, PAD = cfg.pad_token_id;
  const enc = tok(text, { truncation: true, max_length: 256 });
  const src = Array.from(enc.input_ids.data, Number), n = src.length;
  const maxNew = Math.min(200, Math.round(n * 2 + 10));
  let live = [{ ids: [START], logp: 0, probs: [] }];
  const done = [];
  const norm = (h) => h.logp / Math.pow(Math.max(1, h.ids.length - 1), alpha);
  for (let step = 0; step < maxNew && live.length; step++) {
    const B = live.length, t = live[0].ids.length;
    const out = await model({
      input_ids: i64(Array(B).fill(src).flat(), [B, n]),
      attention_mask: i64(Array(B * n).fill(1), [B, n]),
      decoder_input_ids: i64(live.flatMap((h) => h.ids), [B, t]),
    });
    const V = out.logits.dims[2], L = out.logits.data;
    const cand = [];
    for (let b = 0; b < B; b++) {
      const off = (b * t + (t - 1)) * V;
      let m = -Infinity;
      for (let v = 0; v < V; v++) if (v !== PAD && L[off + v] > m) m = L[off + v];
      let s = 0;
      for (let v = 0; v < V; v++) if (v !== PAD) s += Math.exp(L[off + v] - m);
      const lse = m + Math.log(s);
      // top (2·beams) tokens for this hypothesis
      const k = Math.max(2, beams * 2), top = [];
      for (let v = 0; v < V; v++) {
        if (v === PAD) continue;
        const x = L[off + v];
        if (top.length < k) { top.push([x, v]); if (top.length === k) top.sort((a, c) => a[0] - c[0]); }
        else if (x > top[0][0]) { top[0] = [x, v]; top.sort((a, c) => a[0] - c[0]); }
      }
      for (const [x, v] of top) cand.push({ parent: live[b], tok: v, lp: x - lse });
    }
    cand.sort((a, c) => (c.parent.logp + c.lp) - (a.parent.logp + a.lp));
    const next = [];
    for (const c of cand) {
      const h = { ids: [...c.parent.ids, c.tok], logp: c.parent.logp + c.lp, probs: [...c.parent.probs, Math.exp(c.lp)] };
      if (c.tok === EOS) done.push(h); else next.push(h);
      if (next.length >= beams) break;
    }
    live = done.length >= beams ? [] : next;
    onStep && onStep(step + 1);
    if (beams === 1 && done.length) break;
  }
  const finals = (done.length ? done : live).map((h) => ({ ...h, score: norm(h) })).sort((a, c) => c.score - a.score).slice(0, beams);
  const toks = (ids) => tok.model.convert_ids_to_tokens(ids);
  return {
    srcTokens: toks(src),
    hyps: finals.map((h) => ({ text: tok.decode(h.ids, { skip_special_tokens: true }).trim(), ids: h.ids.slice(1), tokens: toks(h.ids.slice(1)), probs: h.probs, logp: h.logp, score: h.score })),
  };
}

// ── BLEU (sentence, BLEU-4, add-one smoothing for n ≥ 2, simple 13a-like tokenization) ──
const tokenize = (s) => s.normalize('NFC').replace(/[’]/g, "'").replace(/([.,!?;:()"«»¡¿])/g, ' $1 ').trim().split(/\s+/).filter(Boolean);
export function bleu(hyp, ref) {
  const h = tokenize(hyp), r = tokenize(ref);
  if (!h.length || !r.length) return null;
  const prec = [];
  for (let n = 1; n <= 4; n++) {
    const grams = (a) => { const m = new Map(); for (let i = 0; i + n <= a.length; i++) { const g = a.slice(i, i + n).join(' '); m.set(g, (m.get(g) || 0) + 1); } return m; };
    const hg = grams(h), rg = grams(r);
    let match = 0, total = 0;
    for (const [g, c] of hg) { total += c; match += Math.min(c, rg.get(g) || 0); }
    prec.push(n === 1 ? (total ? match / total : 0) : (match + 1) / (total + 1));
  }
  const bp = h.length >= r.length ? 1 : Math.exp(1 - r.length / h.length);
  const score = prec[0] === 0 ? 0 : bp * Math.exp(prec.reduce((a, p) => a + Math.log(p), 0) / 4);
  return { score: score * 100, prec, bp, hl: h.length, rl: r.length };
}

function wordDiff(a, b) {
  // LCS over words; returns b's words marked as kept or inserted, plus deleted words of a.
  const A = a.split(/\s+/), Bw = b.split(/\s+/);
  const key = (w) => w.toLowerCase().replace(/[^\p{L}\p{N}]/gu, '');
  const dp = Array.from({ length: A.length + 1 }, () => new Array(Bw.length + 1).fill(0));
  for (let i = A.length - 1; i >= 0; i--) for (let j = Bw.length - 1; j >= 0; j--)
    dp[i][j] = key(A[i]) === key(Bw[j]) ? dp[i + 1][j + 1] + 1 : Math.max(dp[i + 1][j], dp[i][j + 1]);
  const out = []; let i = 0, j = 0;
  while (i < A.length && j < Bw.length) {
    if (key(A[i]) === key(Bw[j])) { out.push(Bw[j], ' '); i++; j++; }
    else if (dp[i + 1][j] >= dp[i][j + 1]) { out.push(el('del', { text: A[i] }), ' '); i++; }
    else { out.push(el('ins', { text: Bw[j] }), ' '); j++; }
  }
  while (i < A.length) out.push(el('del', { text: A[i++] }), ' ');
  while (j < Bw.length) out.push(el('ins', { text: Bw[j++] }), ' ');
  return out;
}

function tokEls(tokens, probs) {
  return tokens.map((t, i) => {
    const p = probs ? probs[i] : null;
    const style = p != null ? { background: `color-mix(in srgb, var(--accent) ${Math.round(p * 35)}%, transparent)`, borderColor: p < 0.5 ? '#d8b66a' : 'var(--divider)' } : {};
    return el('span', { class: 'tok', style, title: p != null ? `${t}: p = ${p.toFixed(3)}` : t }, [t, p != null ? el('small', { text: p.toFixed(2) }) : null]);
  });
}

function scoreBleu() {
  const ref = $('#ref').value.trim();
  const out = last && last.hyps[0] ? last.hyps[0].text : '';
  const b = ref && out ? bleu(out, ref) : null;
  $('#bleu').textContent = b ? b.score.toFixed(1) : '—';
  $('#bleu-detail').replaceChildren(...(b ? [
    ...b.prec.map((p, i) => el('span', {}, [`p${i + 1} `, el('b', { text: p.toFixed(2) })])),
    el('span', {}, ['brevity penalty ', el('b', { text: b.bp.toFixed(2) })]),
    el('span', {}, ['length ', el('b', { text: `${b.hl} / ${b.rl}` })]),
  ] : [el('span', { text: ref ? 'Translate first.' : 'Add a reference to compute BLEU.' })]));
  if (last) last.bleu = b ? b.score : null;
  return b;
}

async function translate() {
  if (busy) return;
  const text = $('#src').value.trim();
  if (!text) return void ($('#status').textContent = 'Type an English sentence first.');
  const lang = $('#dir').value, beams = Number($('#beams').value), alpha = Math.max(0, Math.min(2, Number($('#alpha').value) || 0));
  busy = true; $('#run').disabled = true; $('#back').disabled = true; $('#status').textContent = 'Running…';
  try {
    const p = await load(fwdModel(lang), `MarianMT en→${lang}`);
    const t0 = performance.now();
    const r = await decode(p, text, { beams, alpha, onStep: (s) => { $('#status').textContent = `Decoding… step ${s}${beams > 1 ? ` · ${beams} beams` : ' · greedy'}`; } });
    last = { ...r, lang, src: text };
    const best = r.hyps[0];
    $('#out').textContent = best.text;
    $('#srctoks').replaceChildren(...tokEls(r.srcTokens));
    $('#outtoks').replaceChildren(...tokEls(best.tokens, best.probs));
    $('#hyps tbody').replaceChildren(...r.hyps.map((h, i) => el('tr', {}, [
      el('td', { text: String(i + 1) }), el('td', { class: 'hyp', text: h.text }),
      el('td', { text: h.logp.toFixed(2) }), el('td', { text: String(h.tokens.length) }), el('td', { text: h.score.toFixed(3) }),
    ])));
    scoreBleu();
    $('#status').textContent = `${beams > 1 ? `Beam ${beams}` : 'Greedy'} · α ${alpha} · ${best.tokens.length} pieces · ${((performance.now() - t0) / 1000).toFixed(1)} s`;
    $('#back').disabled = false; $('#backbox').hidden = true;
    window.__lab.lastMT = { text: best.text, hyps: r.hyps.map((h) => h.text), bleu: last.bleu, beams };
  } catch (e) {
    $('#status').textContent = 'Something went wrong: ' + (e.message || e);
    window.__lab.runError = String(e.message || e);
  } finally { busy = false; $('#run').disabled = false; }
}

async function backTranslate() {
  if (busy || !last) return;
  busy = true; $('#run').disabled = true; $('#back').disabled = true;
  try {
    const p = await load(revModel(last.lang), `MarianMT ${last.lang}→en`);
    const r = await decode(p, last.hyps[0].text, { beams: Number($('#beams').value), alpha: Number($('#alpha').value) || 1, onStep: (s) => { $('#status').textContent = `Back-translating… step ${s}`; } });
    $('#backbox').hidden = false;
    $('#backout').replaceChildren(...wordDiff(last.src, r.hyps[0].text));
    $('#status').textContent = 'Round trip complete.';
    window.__lab.lastBack = r.hyps[0].text;
  } catch (e) {
    $('#status').textContent = 'Something went wrong: ' + (e.message || e);
    window.__lab.runError = String(e.message || e);
  } finally { busy = false; $('#run').disabled = false; $('#back').disabled = false; }
}

let sample = SAMPLES[0];
function useSample(s) {
  sample = s;
  $('#src').value = s.src;
  $('#ref').value = s.ref ? s.ref[$('#dir').value] : '';
  $('#sample-note').textContent = s.note;
}

try {
  $('#samples').replaceChildren(...SAMPLES.map((s) => el('button', { class: 'chip', type: 'button', text: s.label, onclick: () => { useSample(s); translate(); } })));
  useSample(SAMPLES[0]); $('#sample-note').textContent = '';
  $('#dir').addEventListener('change', () => { if (sample && $('#src').value.trim() === sample.src) $('#ref').value = sample.ref ? sample.ref[$('#dir').value] : ''; last = null; $('#back').disabled = true; });
  $('#run').addEventListener('click', translate);
  $('#back').addEventListener('click', backTranslate);
  $('#ref').addEventListener('input', scoreBleu);
  ready();
} catch (e) { fail(e); }
