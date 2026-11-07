import { el, $, ready, fail, plotly, plotLayout, PLOT_CONFIG } from '/lab/shared/lab.js';
import { progressUI, transformers as T } from '/lab/shared/hf.js';
import { alignTokens } from '/lab/ner/align.js';

const MODEL = 'Xenova/distilbert-base-cased-distilled-squad';
const MAX_LEN = 384, MAX_SPAN = 30, TOP_K = 5;

// Passages written for this lab. gold: [] marks a question the passage can't answer.
const PASSAGES = [
  {
    title: 'The Analytical Engine',
    text: 'Charles Babbage first described the Analytical Engine in 1837 as a general-purpose mechanical computer. It was to be programmed with punched cards, an idea borrowed from the Jacquard loom, and it separated a "store" that held numbers from a "mill" that performed arithmetic. In 1843 Ada Lovelace published an English translation of an article about the machine by the Italian engineer Luigi Menabrea, adding notes of her own that were about three times longer than the original. Note G described an algorithm for computing Bernoulli numbers, often called the first published computer program. The Analytical Engine was never completed during Babbage\'s lifetime.',
    qs: [
      { q: 'Who first described the Analytical Engine?', gold: ['Charles Babbage'] },
      { q: 'When did Lovelace publish her translation?', gold: ['1843', 'In 1843'] },
      { q: 'What did Note G compute?', gold: ['Bernoulli numbers'] },
      { q: 'Where did the idea of punched cards come from?', gold: ['the Jacquard loom', 'Jacquard loom'] },
      { q: 'Which part of the machine performed arithmetic?', gold: ['the mill', 'mill'] },
      { q: 'How much did the Analytical Engine cost to build?', gold: [] },
    ],
  },
  {
    title: 'The Rosetta Stone',
    text: 'The Rosetta Stone is a granodiorite stele inscribed with a decree issued in Memphis, Egypt, in 196 BC. The decree appears three times, in Egyptian hieroglyphs, in Demotic script and in Ancient Greek. French soldiers found the stone in 1799 near the town of Rashid, also known as Rosetta, in the Nile Delta. Because the Greek text could be read, scholars used it as a key to the hieroglyphs, and Jean-François Champollion announced his decipherment in 1822. The stone has been on display at the British Museum since 1802.',
    qs: [
      { q: 'Who deciphered the hieroglyphs?', gold: ['Jean-François Champollion', 'Champollion'] },
      { q: 'What is the Rosetta Stone made of?', gold: ['granodiorite'] },
      { q: 'When was the stone found?', gold: ['1799', 'in 1799'] },
      { q: 'Where has the stone been displayed since 1802?', gold: ['the British Museum', 'British Museum'] },
      { q: 'Which script let scholars read the other two?', gold: ['Ancient Greek', 'Greek', 'the Greek text'] },
      { q: 'Who carved the inscription on the stone?', gold: [] },
    ],
  },
  {
    title: 'Estuaries',
    text: 'An estuary is a partially enclosed coastal body of water where fresh water from rivers mixes with salt water from the sea. Because the two differ in density, the lighter fresh water often flows seaward over a wedge of denser salt water that pushes inland along the bottom. Estuaries trap sediment and nutrients, which makes them among the most productive ecosystems on Earth and important nurseries for young fish and shellfish. The layer in which salinity changes most rapidly with depth is called the halocline.',
    qs: [
      { q: 'What mixes with salt water in an estuary?', gold: ['fresh water from rivers', 'fresh water'] },
      { q: 'What do estuaries trap?', gold: ['sediment and nutrients'] },
      { q: 'What is the layer where salinity changes fastest with depth called?', gold: ['the halocline', 'halocline'] },
      { q: 'Why does fresh water flow over salt water?', gold: ['the two differ in density', 'Because the two differ in density', 'density'] },
      { q: 'How deep is the average estuary?', gold: [] },
    ],
  },
];

const ui = progressUI($('#load'), 'DistilBERT-SQuAD (≈66 MB)');
ui.set('Model not loaded yet · ≈66 MB download on first run, then cached', 0);
let modelP = null, busy = false, current = null, lastGold = null;

function loadQA() {
  modelP ||= Promise.all([
    T.AutoTokenizer.from_pretrained(MODEL, { progress_callback: (p) => ui.onProgress(p) }),
    T.AutoModelForQuestionAnswering.from_pretrained(MODEL, { dtype: 'q8', device: 'wasm', progress_callback: (p) => ui.onProgress(p) }),
  ]).then(([tok, model]) => { ui.done('Model ready · DistilBERT-SQuAD (cached for next time)'); return { tok, model }; })
    .catch((e) => { modelP = null; ui.error(e); throw e; });
  return modelP;
}

function softmaxRange(arr, a, b) {
  let m = -Infinity; for (let i = a; i <= b; i++) m = Math.max(m, arr[i]);
  const out = new Map(); let s = 0;
  for (let i = a; i <= b; i++) { const v = Math.exp(arr[i] - m); out.set(i, v); s += v; }
  for (const [k, v] of out) out.set(k, v / s);
  return out;
}

// SQuAD evaluation (normalize_answer, exact match, token F1).
const norm = (s) => s.toLowerCase().replace(/[^\p{L}\p{N}\s]/gu, ' ').replace(/\b(a|an|the)\b/g, ' ').replace(/\s+/g, ' ').trim();
const em = (p, g) => Number(norm(p) === norm(g));
function f1(p, g) {
  const pt = norm(p).split(' ').filter(Boolean), gt = norm(g).split(' ').filter(Boolean);
  if (!pt.length || !gt.length) return Number(pt.length === gt.length);
  const counts = new Map(); for (const t of gt) counts.set(t, (counts.get(t) || 0) + 1);
  let same = 0; for (const t of pt) if (counts.get(t) > 0) { same++; counts.set(t, counts.get(t) - 1); }
  if (!same) return 0;
  const pr = same / pt.length, rc = same / gt.length;
  return 2 * pr * rc / (pr + rc);
}

async function answer() {
  if (busy) return;
  const context = $('#context').value.trim(), question = $('#question').value.trim();
  if (!context || !question) { $('#status').textContent = 'Add both a passage and a question.'; return; }
  busy = true; $('#run').disabled = true; $('#status').textContent = 'Running…';
  try {
    const { tok, model } = await loadQA();
    $('#status').textContent = 'Running…';
    const t0 = performance.now();
    const enc = tok(question, { text_pair: context, truncation: true, max_length: MAX_LEN });
    const ids = Array.from(enc.input_ids.data, Number);
    const tokens = tok.model.convert_ids_to_tokens(ids);
    const out = await model(enc);
    const sl = Array.from(out.start_logits.data), elg = Array.from(out.end_logits.data);
    const firstSep = tokens.indexOf('[SEP]'), lastSep = tokens.lastIndexOf('[SEP]');
    const c0 = firstSep + 1, c1 = lastSep - 1;
    const offs = new Array(tokens.length).fill(null);
    alignTokens(context, tokens.slice(c0, c1 + 1)).forEach((o, i) => { offs[c0 + i] = o; });
    const ps = softmaxRange(sl, c0, c1), pe = softmaxRange(elg, c0, c1);
    const isCont = (i) => tokens[i] && tokens[i].startsWith('##');
    const cands = [];
    for (let s = c0; s <= c1; s++) {
      if (!offs[s] || isCont(s)) continue;
      for (let e = s; e <= Math.min(c1, s + MAX_SPAN - 1); e++) {
        if (!offs[e] || (e + 1 <= c1 && isCont(e + 1))) continue;
        cands.push({ s, e, score: sl[s] + elg[e], prob: ps.get(s) * pe.get(e), text: context.slice(offs[s].start, offs[e].end), start: offs[s].start, end: offs[e].end });
      }
    }
    cands.sort((a, b) => b.score - a.score);
    const top = cands.slice(0, TOP_K);
    const truncated = offs[c1] && offs[c1].end < context.length - 2;
    current = { context, top, tokens: tokens.slice(c0, c1 + 1), ps: [...ps.values()], pe: [...pe.values()], sel: 0, truncated };
    render(performance.now() - t0);
    window.__lab.lastAnswer = { answer: top[0] ? top[0].text : '', prob: top[0] ? top[0].prob : 0, abstained: !top[0] || top[0].prob < thr() };
  } catch (e) {
    $('#status').textContent = 'Something went wrong: ' + (e.message || e);
    window.__lab.runError = String(e.message || e);
  } finally { busy = false; $('#run').disabled = false; }
}

const thr = () => Number($('#thr').value);

function render(ms) {
  if (!current) return;
  const { top, context, sel } = current;
  const best = top[0];
  const abstain = !best || best.prob < thr();
  const a = $('#answer');
  a.className = 'answer' + (abstain ? ' none' : '');
  a.textContent = abstain ? 'No confident answer' : best.text;
  $('#meter').replaceChildren(
    el('span', {}, ['Best span probability ', el('b', { text: best ? best.prob.toFixed(3) : '—' })]),
    el('span', {}, ['Threshold ', el('b', { text: thr().toFixed(2) })]),
    abstain && best ? el('span', {}, ['Best guess ', el('b', { text: `“${best.text}”` })]) : null,
    ms != null ? el('span', { text: `${Math.round(ms)} ms` }) : null,
  );
  if (ms != null) $('#status').textContent = current.truncated ? 'Note: the passage was cut to fit 384 tokens; text after the cut can’t be an answer.' : `Scored ${current.tokens.length} passage tokens.`;
  // gold / EM / F1
  const g = $('#gold');
  if (lastGold && lastGold.q === $('#question').value.trim()) {
    g.hidden = false;
    const pred = abstain ? '' : best.text;
    if (!lastGold.gold.length) {
      g.textContent = abstain
        ? 'Gold: unanswerable. The model abstained, which a SQuAD 2.0 evaluation scores as correct (EM 1, F1 1).'
        : `Gold: unanswerable. The model answered anyway, so EM 0 and F1 0. Raise the threshold to see where it would abstain.`;
    } else {
      const E = Math.max(...lastGold.gold.map((x) => em(pred, x))), F = Math.max(...lastGold.gold.map((x) => f1(pred, x)));
      g.textContent = `Gold: ${lastGold.gold.map((x) => `“${x}”`).join(' or ')} · Exact match ${E} · F1 ${F.toFixed(2)}`;
    }
  } else g.hidden = true;
  // candidates
  $('#cands tbody').replaceChildren(...(top.length ? top.map((c, i) => el('tr', { class: i === sel ? 'sel' : '' }, [
    el('td', { text: String(i + 1) }),
    el('td', {}, el('button', { type: 'button', text: c.text, title: 'Show in passage', onclick: () => { current.sel = i; render(); } })),
    el('td', { text: c.score.toFixed(2) }),
    el('td', { text: c.prob.toFixed(3) }),
  ])) : [el('tr', {}, el('td', { colspan: 4, class: 'note', text: 'No valid spans.' }))]));
  // passage with highlight
  const pv = $('#passage-view');
  const shown = top[sel];
  if (shown) {
    pv.replaceChildren(context.slice(0, shown.start), el('mark', { class: sel === 0 && abstain ? 'alt' : '', text: context.slice(shown.start, shown.end) }), context.slice(shown.end));
  } else pv.textContent = context;
  drawChart();
}

async function drawChart() {
  const P = await plotly().catch(() => null);
  if (!P || !current) return;
  const x = current.tokens.map((_, i) => i);
  const hover = current.tokens.map((t) => t);
  const narrow = $('#chart').clientWidth < 560;
  P.react('chart', [
    { type: 'bar', name: 'P(start)', x, y: current.ps, text: hover, hovertemplate: '%{text}<br>P(start) %{y:.3f}<extra></extra>', marker: { color: '#7ea3c8' } },
    { type: 'bar', name: 'P(end)', x, y: current.pe, text: hover, hovertemplate: '%{text}<br>P(end) %{y:.3f}<extra></extra>', marker: { color: '#d8b66a' } },
  ], plotLayout({
    barmode: 'group', bargap: 0.15, margin: { l: narrow ? 40 : 48, r: 12, t: 8, b: 36 },
    xaxis: { title: { text: 'passage token', font: { size: 12, color: '#9aa3ad' } }, gridcolor: 'rgba(0,0,0,0)', color: '#9aa3ad', tickfont: { size: 10 } },
    yaxis: { title: { text: 'probability', font: { size: 12, color: '#9aa3ad' } }, range: [0, 1], gridcolor: 'rgba(228,231,235,.08)', color: '#9aa3ad' },
    legend: { orientation: 'h', x: 0, y: 1.12, bgcolor: 'rgba(0,0,0,0)' }, showlegend: true,
  }), PLOT_CONFIG);
}

function selectPassage(i) {
  const p = PASSAGES[i];
  $('#context').value = p.text;
  $('#qchips').replaceChildren(...p.qs.map((q) => el('button', {
    type: 'button', class: 'chip' + (q.gold.length ? '' : ' unans'), text: q.q,
    title: q.gold.length ? 'Answerable' : 'Not answerable from the passage',
    onclick: () => { $('#question').value = q.q; lastGold = q; answer(); },
  })));
  $('#question').value = p.qs[0].q; lastGold = p.qs[0];
}

try {
  $('#preset').replaceChildren(...PASSAGES.map((p, i) => el('option', { value: i, text: p.title })), el('option', { value: 'custom', text: 'Your own passage' }));
  $('#preset').addEventListener('change', (e) => {
    if (e.target.value === 'custom') { $('#context').value = ''; $('#qchips').replaceChildren(); $('#question').value = ''; lastGold = null; $('#context').focus(); }
    else selectPassage(Number(e.target.value));
  });
  $('#qform').addEventListener('submit', (e) => { e.preventDefault(); const q = $('#question').value.trim(); if (!lastGold || lastGold.q !== q) lastGold = PASSAGES.flatMap((p) => p.qs).find((x) => x.q === q) || null; answer(); });
  $('#thr').addEventListener('input', () => { $('#thrv').textContent = thr().toFixed(2); if (current) render(); if (window.__lab.lastAnswer && current?.top[0]) window.__lab.lastAnswer.abstained = current.top[0].prob < thr(); });
  selectPassage(0);
  ready();
} catch (e) { fail(e); }
