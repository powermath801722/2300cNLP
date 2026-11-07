import { el, $, ready, fail } from '/lab/shared/lab.js';
import { progressUI, transformers as T } from '/lab/shared/hf.js';

const MODEL = 'Xenova/nli-deberta-v3-xsmall';
const SAMPLES = [
  { label: 'News: rates', text: 'The central bank raised interest rates by half a percentage point on Tuesday, citing stubborn inflation and a tight labour market.', labels: 'economy, sports, technology, health, politics', multi: false, note: 'A topical label set: the classic zero-shot use case.' },
  { label: 'News: match', text: 'Despite a slow start, the visitors scored twice in the final ten minutes to lift the cup in front of a sold-out stadium.', labels: 'economy, sports, technology, health, politics', multi: false, note: 'Same labels, different text. No word in the text is literally “sports”.' },
  { label: 'Support ticket', text: 'My order arrived two weeks late and the box was crushed. I have emailed three times and nobody has replied.', labels: 'complaint, praise, question, shipping, billing', multi: true, note: 'Multi-label: a ticket can be a complaint and about shipping at the same time.' },
  { label: 'Mixed review', text: 'The new phone has a gorgeous screen, but the battery barely lasts until lunch.', labels: 'positive, negative, battery, display, price', multi: true, note: 'Aspects and sentiment together. In single-label mode the labels would have to compete for one probability mass.' },
  { label: 'Template A', text: 'I can’t believe how good that soup was. I’m going back tomorrow.', labels: 'positive, negative', multi: false, tmpl: 'This example is {}.', note: 'Default template. Compare with “Template B”, same text and labels.' },
  { label: 'Template B', text: 'I can’t believe how good that soup was. I’m going back tomorrow.', labels: 'positive, negative', multi: false, tmpl: 'The sentiment of this review is {}.', note: 'A more specific template. Prompt wording changes the numbers.' },
  { label: 'Out of its depth', text: 'Patient presents with dyspnea and bilateral crackles; BNP elevated; started on IV furosemide.', labels: 'cardiology, dermatology, orthopedics, oncology', multi: false, note: 'Specialist jargon: labels and text are far from SNLI/MultiNLI, so expect lower confidence or mistakes.' },
];

const ui = progressUI($('#load'), 'DeBERTa-v3-xsmall NLI (≈87 MB)');
ui.set('Model not loaded yet · ≈87 MB download on first run, then cached', 0);
let modelP = null, busy = false;

function load() {
  modelP ||= Promise.all([
    T.AutoTokenizer.from_pretrained(MODEL, { progress_callback: (p) => ui.onProgress(p) }),
    T.AutoModelForSequenceClassification.from_pretrained(MODEL, { dtype: 'q8', device: 'wasm', progress_callback: (p) => ui.onProgress(p) }),
  ]).then(([tok, model]) => {
    ui.done('Model ready · DeBERTa-v3-xsmall NLI (cached for next time)');
    const id2label = model.config.id2label;
    const idx = (name) => Number(Object.keys(id2label).find((k) => id2label[k].toLowerCase() === name));
    return { tok, model, E: idx('entailment'), N: idx('neutral'), C: idx('contradiction') };
  }).catch((e) => { modelP = null; ui.error(e); throw e; });
  return modelP;
}

const sm = (xs) => { const m = Math.max(...xs); const e = xs.map((x) => Math.exp(x - m)); const s = e.reduce((a, b) => a + b, 0); return e.map((x) => x / s); };

export async function classify(text, labels, tmpl, multi) {
  const { tok, model, E, N, C } = await load();
  const hyps = labels.map((l) => tmpl.replace('{}', l));
  const enc = tok(new Array(hyps.length).fill(text), { text_pair: hyps, padding: true, truncation: true });
  const out = await model(enc);
  const L = Array.from(out.logits.data), k = out.logits.dims[1];
  const rows = hyps.map((h, i) => {
    const lg = L.slice(i * k, i * k + k);
    const p3 = sm(lg);
    return { label: labels[i], hyp: h, logits: lg, pE: p3[E], pN: p3[N], pC: p3[C], eLogit: lg[E], multi: sm([lg[C], lg[E]])[1] };
  });
  const single = sm(rows.map((r) => r.eLogit));
  rows.forEach((r, i) => { r.score = multi ? r.multi : single[i]; });
  return rows;
}

async function run() {
  if (busy) return;
  const text = $('#text').value.trim();
  const labels = [...new Set($('#labels').value.split(',').map((s) => s.trim()).filter(Boolean))].slice(0, 12);
  let tmpl = $('#tmpl').value.trim();
  const multi = $('#multi').checked;
  if (!text) return void ($('#status').textContent = 'Add some text to classify.');
  if (labels.length < (multi ? 1 : 2)) return void ($('#status').textContent = multi ? 'Add at least one label.' : 'Add at least two labels (single-label scores are relative to each other).');
  if (!tmpl.includes('{}')) { tmpl = tmpl ? `${tmpl} {}` : 'This example is {}.'; $('#tmpl').value = tmpl; }
  busy = true; $('#run').disabled = true; $('#status').textContent = `Running ${labels.length} NLI passes…`;
  try {
    const t0 = performance.now();
    const rows = await classify(text, labels, tmpl, multi);
    render(rows, multi, text);
    $('#status').textContent = `${labels.length} premise–hypothesis pairs · ${Math.round(performance.now() - t0)} ms`;
    window.__lab.lastZS = rows.slice().sort((a, b) => b.score - a.score).map((r) => ({ label: r.label, score: r.score }));
  } catch (e) {
    $('#status').textContent = 'Something went wrong: ' + (e.message || e);
    window.__lab.runError = String(e.message || e);
  } finally { busy = false; $('#run').disabled = false; }
}

function render(rows, multi, text) {
  $('#mode-note').textContent = multi
    ? 'Multi-label: each score is P(entailment) vs P(contradiction) for that label alone; scores don’t sum to 1.'
    : 'Single-label: softmax of the entailment logits across labels; scores sum to 1.';
  const sorted = rows.slice().sort((a, b) => b.score - a.score);
  $('#bars').replaceChildren(...sorted.map((r) => el('li', { 'aria-label': `${r.label}: ${(r.score * 100).toFixed(1)} percent` }, [
    el('span', { class: 'lab', text: r.label, title: r.label }),
    el('span', { class: 'track' }, el('i', { style: { width: (r.score * 100).toFixed(1) + '%' } })),
    el('span', { class: 'v', text: (r.score * 100).toFixed(1) + '%' }),
  ])));
  $('#pairs tbody').replaceChildren(...rows.map((r) => el('tr', {}, [
    el('td', { class: 'hyp' }, [el('span', { class: 'note', text: 'Premise: your text · Hypothesis: ' }), el('br'), el('b', { text: r.hyp })]),
    el('td', { text: r.pE.toFixed(2) }), el('td', { text: r.pN.toFixed(2) }), el('td', { text: r.pC.toFixed(2) }),
    el('td', {}, el('span', { class: 'mini', 'aria-hidden': 'true' }, [
      el('i', { style: { width: r.pE * 100 + '%', background: '#8fbf9a' } }),
      el('i', { style: { width: r.pN * 100 + '%', background: '#9aa3ad' } }),
      el('i', { style: { width: r.pC * 100 + '%', background: '#d98c7a' } }),
    ])),
  ])));
}

function useSample(s) {
  $('#text').value = s.text; $('#labels').value = s.labels; $('#multi').checked = s.multi;
  $('#tmpl').value = s.tmpl || 'This example is {}.';
  $('#sample-note').textContent = s.note;
}

try {
  $('#samples').replaceChildren(...SAMPLES.map((s) => el('button', { class: 'chip', type: 'button', text: s.label, onclick: () => { useSample(s); run(); } })));
  useSample(SAMPLES[0]); $('#sample-note').textContent = '';
  $('#run').addEventListener('click', run);
  for (const id of ['#labels', '#tmpl']) $(id).addEventListener('keydown', (e) => { if (e.key === 'Enter') run(); });
  ready();
} catch (e) { fail(e); }
