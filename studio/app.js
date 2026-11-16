// Braidwater Text Studio — UI. Analysis runs in /studio/worker.js (classic models) and on demand via
// /studio/lib/ner.js and /studio/lib/signals.js (transformer models).
import { el, $, $$, ready, fail, plotly, plotLayout, PLOT_CONFIG, REDUCED } from '/lab/shared/lab.js';
import { SAMPLES } from './samples.js';
import { fleschLabel } from './lib/readability.js';

const NAMES = { vader: 'Lexicon + rules (VADER)', nb: 'Naive Bayes', lr: 'Logistic regression', tf: 'DistilBERT (SST-2)' };
const LABEL = { pos: 'Positive', neg: 'Negative', neu: 'Neutral', mixed: 'Mixed' };
const POSNAMES = { noun: 'Nouns', verb: 'Verbs', adj: 'Adjectives', adv: 'Adverbs', pron: 'Pronouns', det: 'Determiners', adp: 'Prepositions & particles', conj: 'Conjunctions', num: 'Numbers', other: 'Other' };
const POSCOLORS = { noun: '#7ea3c8', verb: '#d8b66a', adj: '#8fbf9a', adv: '#d98c7a', pron: '#b49bd6', det: '#6cc3c3', adp: '#9cb4d8', conj: '#e0a35c', num: '#c3d07a', other: '#8a94a0' };
const ENT = { PER: ['Person', '#d8b66a'], ORG: ['Organization', '#7ea3c8'], LOC: ['Location', '#8fbf9a'], MISC: ['Misc.', '#b49bd6'] };
const fmt = (n, d = 0) => Number(n).toLocaleString('en-US', { minimumFractionDigits: d, maximumFractionDigits: d });
const pct = (x, d = 0) => (x * 100).toFixed(d) + '%';
const wordsOf = (t) => (t.trim() ? t.trim().split(/\s+/).length : 0);

const state = { text: '', report: null, active: -1, lens: 'none', tf: null, ents: null, sig: null, ignored: new Set(), sampleId: null, summaryK: null };

// ───────────────────────── worker plumbing (falls back to the main thread) ─────────────────────────
let worker = null, seq = 0;
const handlers = new Map();
try {
  worker = new Worker('/studio/worker.js', { type: 'module' });
  worker.onmessage = (e) => { const h = handlers.get(e.data.id); if (h) h(e.data); };
  worker.onerror = () => { worker = null; };
} catch { worker = null; }

let mainLib = null;
const lib = () => (mainLib ||= import('./lib/analyze.js'));

function call(msg, onMsg) {
  if (!worker) return null;
  const id = ++seq;
  handlers.set(id, (d) => { if (onMsg(d)) handlers.delete(id); });
  worker.postMessage({ id, ...msg });
  return id;
}

function warm() {
  return new Promise((resolve, reject) => {
    if (!call({ type: 'warm' }, (d) => { if (d.type === 'error') reject(new Error(d.message)); else resolve(); return true; }))
      lib().then((m) => m.loadClassic()).then(resolve, reject);
  });
}

/** Run the classic pipeline; onReport fires as soon as the report exists, the promise resolves once spelling is in. */
function runPipeline(text, summaryLength, onReport) {
  return new Promise((resolve, reject) => {
    let rep = null;
    const ok = call({ type: 'analyze', text, summaryLength }, (d) => {
      if (d.type === 'error') { reject(new Error(d.message)); return true; }
      if (d.type === 'report') { rep = d.report; onReport(rep); return false; }
      if (d.type === 'spelling') { rep.style.spelling = d.spelling; rep.timings.spellingMs = d.ms; resolve(rep); return true; }
      return false;
    });
    if (!ok) lib().then(async (m) => {
      const r = await m.analyzeText(text, { summaryLength, spelling: false }); onReport(r);
      const M = await m.loadClassic(); const t0 = performance.now();
      r.style.spelling = M.spell.checkText(text); r.timings.spellingMs = +(performance.now() - t0).toFixed(1);
      resolve(r);
    }).catch(reject);
  });
}

function summaryCall(text, k) {
  return new Promise((resolve, reject) => {
    if (!call({ type: 'summary', text, k }, (d) => { if (d.type === 'error') reject(new Error(d.message)); else resolve(d); return true; }))
      lib().then((m) => resolve(m.summaryFor(text, k))).catch(reject);
  });
}

// ───────────────────────── input ─────────────────────────
function setMsg(t) { const m = $('#msg'); m.textContent = t || ''; m.hidden = !t; }
function updateCounter() { const n = wordsOf($('#txt').value); $('#counter').textContent = `${fmt(n)} word${n === 1 ? '' : 's'}`; }

function stripMarkdown(md) {
  return md.replace(/```[\s\S]*?```/g, '').replace(/`([^`]+)`/g, '$1').replace(/!\[[^\]]*\]\([^)]*\)/g, '')
    .replace(/\[([^\]]+)\]\([^)]*\)/g, '$1').replace(/^\s{0,3}#{1,6}\s+/gm, '').replace(/^\s{0,3}>\s?/gm, '')
    .replace(/^\s*[-*+]\s+/gm, '').replace(/(\*\*|__)(.*?)\1/g, '$2').replace(/(\*|_)(.*?)\1/g, '$2').replace(/^\s*---+\s*$/gm, '');
}

async function loadSample(s) {
  let text = s.text;
  if (s.fromLab) {
    const r = await fetch(s.fromLab); const d = await r.json();
    text = d.samples.find((x) => x.id === s.id).text;
  }
  $('#txt').value = text; state.sampleId = s.id; updateCounter();
  return analyze();
}

// ───────────────────────── analysis ─────────────────────────
let busy = null;
/** Analyze the textarea. If a run is in progress, wait for it, then run again on the newest text. */
async function analyze() {
  if (busy) { await busy.catch(() => {}); if ($('#txt').value.replace(/\r/g, '') === state.text && state.report) return state.report; return analyze(); }
  const p = analyzeNow(); busy = p;
  try { return await p; } finally { if (busy === p) busy = null; }
}
async function analyzeNow() {
  const text = $('#txt').value.replace(/\r/g, '');
  setMsg('');
  if (wordsOf(text) < 3) { setMsg('Paste or type at least a sentence or two first.'); return null; } $('#go').disabled = true; $('#go').textContent = 'Analyzing…';
  state.text = text; state.tf = null; state.ents = null; state.sig = null; state.active = -1;
  if (state.lens === 'entities') state.lens = 'none';
  $('#entities').replaceChildren(el('p', { class: 'empty-note', text: 'Not run yet for this text.' }));
  $('#signals').replaceChildren(el('p', { class: 'empty-note', text: 'Not run yet for this text.' }));
  try {
    const rep = await runPipeline(text, state.summaryK, (r) => { state.report = r; renderAll(); });
    state.report = rep; renderStyle(); renderTiming();
    ['#ner-go', '#sig-go', '#dl-md', '#dl-json', '#copy-md'].forEach((s) => { $(s).disabled = false; });
    return rep;
  } catch (e) {
    setMsg('Analysis failed: ' + e.message); return null;
  } finally {
    $('#go').disabled = false; $('#go').textContent = 'Analyze';
  }
}

function renderAll() {
  renderOverview(); renderSentiment(); renderStyle(); renderIdeas(); renderTopics(); renderTiming();
}
function renderTiming() {
  const r = state.report; if (!r) return;
  const t = r.timings;
  $('#status').replaceChildren(el('span', { class: 'timing', text: `Analyzed ${fmt(r.overview.words)} words in ${fmt(t.totalMs)} ms` + (t.spellingMs != null ? ` · spelling ${fmt(t.spellingMs)} ms` : ' · checking spelling…') + (worker ? ' · in a background thread' : '') }));
}

// ───────────────────────── sentence highlighting ─────────────────────────
function sentBtn(i, extra) {
  const s = state.report.sentences[i];
  return el('button', { type: 'button', 'data-s': i, class: i === state.active ? 'active' : '', onclick: () => setActive(i, true) }, [s.text, extra ? el('small', { text: extra }) : null]);
}
function setActive(i, scrollDoc) {
  state.active = state.active === i ? -1 : i;
  $$('[data-s]').forEach((n) => n.classList.toggle('active', +n.dataset.s === state.active));
  renderInspector();
  if (state.active >= 0 && scrollDoc) {
    const d = $('#doc'), n = d.querySelector(`.sent[data-s="${state.active}"]`);
    if (n) d.scrollTop = Math.max(0, n.offsetTop - d.offsetTop - d.clientHeight / 3);
  }
}

let posLib = null;
async function renderInspector() {
  const box = $('#inspector'), i = state.active, r = state.report;
  if (i < 0 || !r) { box.hidden = true; return; }
  const s = r.sentences[i], sv = s.sentiment;
  const pas = r.style.passives.filter((p) => p.sentence === i).map((p) => p.phrase);
  const verdict = (k, lab, val) => el('span', {}, [`${NAMES[k].split(' (')[0]}: `, el('span', { class: 'lab ' + lab, text: LABEL[lab] }), ` ${val}  `]);
  box.replaceChildren(
    el('q', { text: s.text }),
    el('div', {}, [`Sentence ${i + 1} of ${r.sentences.length} · ${s.words} words` + (pas.length ? ` · passive: “${pas.join('”, “')}”` : '') + (r.keyIdeas.summary.includes(i) ? ' · in the summary' : '')]),
    el('div', { style: { marginTop: '6px', display: 'flex', flexWrap: 'wrap', gap: '6px 12px' } }, [
      verdict('vader', sv.vader.label, sv.vader.compound.toFixed(2)), verdict('nb', sv.nb.label, pct(sv.nb.prob)), verdict('lr', sv.lr.label, pct(sv.lr.prob)),
      state.tf ? verdict('tf', state.tf[i].label, pct(state.tf[i].prob)) : null,
    ]),
    el('div', { class: 'tags', 'aria-label': 'Part-of-speech tags' }, el('span', { text: 'tagging…' })),
  );
  box.hidden = false;
  try {
    posLib ||= import('./lib/pos.js').then((m) => m.loadPOS());
    const P = await posLib;
    if (state.active !== i) return;
    const { words, tags } = P.tagSentence(s.text.slice(0, 1500));
    box.querySelector('.tags').replaceChildren(...words.map((w, k) => el('span', { title: tags[k] }, [el('b', { text: w }), tags[k]])));
  } catch { box.querySelector('.tags').replaceChildren(); }
}

// ───────────────────────── Overview + document lenses ─────────────────────────
const LENSES = [
  ['none', 'Plain'], ['sentiment', 'Sentiment'], ['summary', 'Summary'], ['long', 'Long sentences'], ['passive', 'Passive voice'], ['entities', 'Entities'],
];
function lensColor(i) {
  const r = state.report;
  switch (state.lens) {
    case 'sentiment': { const c = r.sentences[i].sentiment.vader.compound; if (Math.abs(c) < 0.05) return ''; const a = (0.12 + Math.min(Math.abs(c), 1) * 0.4).toFixed(2); return c > 0 ? `rgba(143,191,154,${a})` : `rgba(217,140,122,${a})`; }
    case 'summary': return r.keyIdeas.summary.includes(i) ? 'rgba(126,163,200,.32)' : '';
    case 'long': return r.style.lengths.long.includes(i) ? 'rgba(216,182,106,.30)' : '';
    case 'passive': return r.style.passives.some((p) => p.sentence === i) ? 'rgba(180,155,214,.32)' : '';
    default: return '';
  }
}
function legendFor() {
  const sw = (c, t) => el('span', {}, [el('i', { style: { background: c } }), t]);
  switch (state.lens) {
    case 'sentiment': return [sw('rgba(143,191,154,.5)', 'positive'), sw('rgba(217,140,122,.5)', 'negative'), 'darker = stronger (VADER)'];
    case 'summary': return [sw('rgba(126,163,200,.45)', `TextRank summary (${state.report.keyIdeas.summary.length} sentences)`)];
    case 'long': return [sw('rgba(216,182,106,.45)', 'over 30 words')];
    case 'passive': return [sw('rgba(180,155,214,.45)', 'contains a likely passive')];
    case 'entities': return state.ents ? Object.entries(ENT).map(([k, [n, c]]) => sw(c, n)) : ['Run “Find entities” below to use this lens.'];
    default: return [];
  }
}
function entityRegex() {
  if (!state.ents || !state.ents.length) return null;
  const uniq = [...new Map(state.ents.filter((e) => e.text && e.text.length > 1).map((e) => [e.text, e.type])).entries()].sort((a, b) => b[0].length - a[0].length);
  if (!uniq.length) return null;
  const esc = (s) => s.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
  return { re: new RegExp(`(?<![\\p{L}\\p{N}])(${uniq.map(([t]) => esc(t)).join('|')})(?![\\p{L}\\p{N}])`, 'gu'), type: new Map(uniq) };
}
function sentenceContent(text, ER) {
  if (state.lens !== 'entities' || !ER) return [text];
  const out = []; let last = 0;
  for (const m of text.matchAll(ER.re)) {
    if (m.index > last) out.push(text.slice(last, m.index));
    const [name, color] = ENT[ER.type.get(m[0])] || ['', '#888'];
    out.push(el('span', { class: 'ent', style: { borderColor: color }, title: name, text: m[0] }));
    last = m.index + m[0].length;
  }
  if (last < text.length) out.push(text.slice(last));
  return out;
}
function renderDoc() {
  const r = state.report, doc = $('#doc');
  const ER = entityRegex();
  const paras = new Map();
  for (const h of r.headings) (paras.get(h.para) || paras.set(h.para, { h: [], s: [] }).get(h.para)).h.push(h.text);
  for (const s of r.sentences) (paras.get(s.para) || paras.set(s.para, { h: [], s: [] }).get(s.para)).s.push(s.index);
  doc.replaceChildren(...[...paras.entries()].sort((a, b) => a[0] - b[0]).map(([, p]) => el('p', {}, [
    ...p.h.map((t) => el('span', { class: 'hd', text: t })),
    ...p.s.flatMap((i) => [el('button', { type: 'button', class: 'sent' + (i === state.active ? ' active' : ''), 'data-s': i, style: { background: lensColor(i) },
      'aria-label': `Sentence ${i + 1}`, onclick: () => setActive(i, false) }, sentenceContent(r.sentences[i].text, ER)), ' ']),
  ])));
  $('#lenses').replaceChildren(...LENSES.map(([k, t]) => el('button', { class: 'chip', type: 'button', 'aria-pressed': String(state.lens === k), text: t,
    onclick: () => { state.lens = k; renderDoc(); } })));
  $('#legend').replaceChildren(...legendFor());
}
function renderOverview() {
  const o = state.report.overview;
  const fkgl = Math.max(0, o.fkgl), fre = Math.max(0, Math.min(121, o.fre));
  const mins = (m) => (m < 1 ? `${Math.max(1, Math.round(m * 60))} s` : `${m < 10 ? m.toFixed(1) : Math.round(m)} min`);
  const stat = (b, s, sub) => el('div', {}, [el('b', { text: b }), el('span', { text: s }), sub ? el('small', { text: sub }) : null]);
  $('#overview').replaceChildren(
    el('div', { class: 'statgrid' }, [
      stat(fmt(o.words), 'words', `${fmt(o.unique)} unique`),
      stat(fmt(o.sentences), 'sentences', `${o.avgSentence.toFixed(1)} words each on average`),
      stat(fmt(o.paragraphs), o.paragraphs === 1 ? 'paragraph' : 'paragraphs', `${(o.sentences / Math.max(1, o.paragraphs)).toFixed(1)} sentences each`),
      stat(mins(o.readingMinutes), 'reading time', `${mins(o.speakingMinutes)} aloud`),
      stat(fre.toFixed(0), 'Flesch reading ease', fleschLabel(o.fre)),
      stat(fkgl.toFixed(1), 'Flesch–Kincaid grade', fkgl >= 13 ? 'college level' : fkgl >= 9 ? 'high school' : 'middle school or below'),
      stat(o.mattr.toFixed(2), 'lexical diversity (MATTR)', `type–token ratio ${o.ttr.toFixed(2)}`),
      stat(o.avgSyllables.toFixed(2), 'syllables per word', `${fmt(o.polysyllabic)} words of 3+ syllables`),
    ]),
    el('div', {}, [el('div', { class: 'field-label', style: { margin: '6px 0 8px' }, text: 'Longest sentences' }),
      el('ol', { class: 'slist' }, o.longestSentences.map((i) => el('li', {}, sentBtn(i, `${state.report.sentences[i].words} words`))))]),
  );
  $('#docwrap').hidden = false;
  renderDoc(); renderInspector();
}

// ───────────────────────── Sentiment ─────────────────────────
function verdictCard(key, label, prob, detail) {
  return el('div', { class: 'verdict', 'data-method': key }, [
    el('h3', { text: NAMES[key] }),
    el('div', {}, [el('span', { class: 'lab ' + label, 'data-label': label, text: LABEL[label] })]),
    el('div', { class: 'meter', role: 'img', 'aria-label': `Positive ${pct(prob)}` }, el('i', { style: { width: pct(prob, 1) } })),
    el('span', { class: 'note', text: detail }),
  ]);
}
function renderSentiment() {
  const r = state.report, s = r.sentiment, o = s.overall;
  const cards = [
    verdictCard('vader', o.vader.label, (o.vader.compound + 1) / 2, `mean compound ${o.vader.compound.toFixed(3)} over ${r.sentences.length} sentences`),
    verdictCard('nb', o.nb.label, o.nb.prob, `P(positive) ${pct(o.nb.prob, 1)} for the whole text`),
    verdictCard('lr', o.lr.label, o.lr.prob, `P(positive) ${pct(o.lr.prob, 1)} for the whole text`),
  ];
  if (state.tf) {
    const m = state.tf.reduce((a, x) => a + x.prob, 0) / state.tf.length;
    cards.push(verdictCard('tf', m >= 0.5 ? 'pos' : 'neg', m, `mean P(positive) ${pct(m, 1)} over sentences`));
  }
  const tfBtn = state.tf ? null : el('button', { class: 'btn btn-secondary', type: 'button', id: 'tf-go', text: 'Add DistilBERT (≈67 MB on first use)', onclick: runTf });
  const posL = s.mostPositive, negL = s.mostNegative;
  $('#sentiment').replaceChildren(
    el('p', { style: { margin: 0, fontSize: '17px' } }, ['Overall: ', el('span', { class: 'lab ' + o.label, id: 'overall', 'data-label': o.label, text: LABEL[o.label] }), ` · ${o.agreement} classic models agree`]),
    el('div', { class: 'verdicts' }, cards),
    el('div', { class: 'row' }, [tfBtn, el('div', { id: 'tf-load' })]),
    el('div', { class: 'field-label', text: 'Sentiment arc (click a point to find the sentence)' }),
    el('div', { class: 'plot', id: 'arc', role: 'img', 'aria-label': 'Line chart of sentiment by sentence' }),
    el('div', { class: 'grid2' }, [
      el('div', {}, [el('div', { class: 'field-label', style: { marginBottom: '8px' }, text: 'Most positive' }), posL.length ? el('ol', { class: 'slist' }, posL.map((i) => el('li', {}, sentBtn(i, `VADER ${r.sentences[i].sentiment.vader.compound.toFixed(2)}`)))) : el('p', { class: 'empty-note', text: 'No clearly positive sentences.' })]),
      el('div', {}, [el('div', { class: 'field-label', style: { marginBottom: '8px' }, text: 'Most negative' }), negL.length ? el('ol', { class: 'slist' }, negL.map((i) => el('li', {}, sentBtn(i, `VADER ${r.sentences[i].sentiment.vader.compound.toFixed(2)}`)))) : el('p', { class: 'empty-note', text: 'No clearly negative sentences.' })]),
    ]),
  );
  drawArc();
}
async function drawArc() {
  let P; try { P = await plotly(); } catch { return; }
  const r = state.report, x = r.sentences.map((s) => s.index + 1);
  const traces = [{ type: 'scatter', mode: 'lines+markers', name: 'VADER compound', x, y: r.sentiment.arc, line: { color: '#7ea3c8', width: 2, shape: 'spline', smoothing: 0.6 },
    marker: { size: 7, color: r.sentiment.arc.map((v) => (v > 0.05 ? '#8fbf9a' : v < -0.05 ? '#d98c7a' : '#8a94a0')) },
    text: r.sentences.map((s) => s.text.length > 90 ? s.text.slice(0, 88) + '…' : s.text), hovertemplate: 'Sentence %{x}: %{y:.2f}<br>%{text}<extra></extra>' }];
  if (state.tf) traces.push({ type: 'scatter', mode: 'lines', name: 'DistilBERT (2·P−1)', x, y: state.tf.map((t) => 2 * t.prob - 1), line: { color: '#d8b66a', width: 1.5, dash: 'dot' }, hovertemplate: 'Sentence %{x}: %{y:.2f}<extra>DistilBERT</extra>' });
  const g = $('#arc'); if (!g) return;
  await P.react(g, traces, plotLayout({ margin: { l: 44, r: 12, t: 12, b: 40 }, showlegend: !!state.tf, legend: { orientation: 'h', y: 1.12 },
    xaxis: { title: { text: 'Sentence' }, gridcolor: 'rgba(228,231,235,.08)', color: '#9aa3ad', zeroline: false },
    yaxis: { range: [-1.05, 1.05], gridcolor: 'rgba(228,231,235,.08)', zerolinecolor: 'rgba(228,231,235,.3)', color: '#9aa3ad' }, transition: REDUCED ? undefined : { duration: 0 } }), PLOT_CONFIG);
  if (!g._bound) { g._bound = true; g.on('plotly_click', (e) => setActive(e.points[0].x - 1, true)); }
}
async function runTf() {
  const btn = $('#tf-go'); if (btn) btn.disabled = true;
  try {
    const { pipe, progressUI } = await import('/lab/shared/hf.js');
    const { DISTILBERT } = await import('./lib/sentiment.js');
    const ui = progressUI($('#tf-load'), 'DistilBERT (≈67 MB)');
    const clf = await pipe('sentiment-analysis', DISTILBERT, { ui, label: 'DistilBERT', readyText: 'DistilBERT ready' });
    const sents = state.report.sentences.map((s) => s.text), out = [];
    for (let i = 0; i < sents.length; i += 8) {
      const res = await clf(sents.slice(i, i + 8));
      for (const r of [].concat(res)) { const p = r.label === 'POSITIVE' ? r.score : 1 - r.score; out.push({ label: p >= 0.5 ? 'pos' : 'neg', prob: +p.toFixed(4) }); }
      ui.set(`DistilBERT · ${Math.min(i + 8, sents.length)} / ${sents.length} sentences`, Math.min(1, (i + 8) / sents.length));
    }
    state.tf = out; ui.done(`DistilBERT · ${sents.length} sentences scored`);
    renderSentiment(); renderInspector();
  } catch (e) {
    if (btn) { btn.disabled = false; btn.textContent = 'Retry DistilBERT'; }
    setMsg('DistilBERT failed to load: ' + e.message);
  }
}

// ───────────────────────── Grammar & style ─────────────────────────
function bars(rows) {
  return el('div', { class: 'bars' }, rows.map(([name, p, color, val]) => el('div', {}, [
    el('span', { text: name }), el('div', { class: 'b', role: 'img', 'aria-label': `${name} ${val}` }, el('i', { style: { width: (Math.min(1, p) * 100).toFixed(1) + '%', background: color } })), el('span', { class: 'v', text: val }),
  ])));
}
function context(text, a, b) {
  const s = Math.max(0, a - 40), e = Math.min(text.length, b + 40);
  return (s > 0 ? '…' : '') + text.slice(s, a).replace(/\s+/g, ' ') + '[' + text.slice(a, b) + ']' + text.slice(b, e).replace(/\s+/g, ' ') + (e < text.length ? '…' : '');
}
function applyFix(item, w) {
  const ta = $('#txt'); let t = ta.value.replace(/\r/g, '');
  if (t === state.text && t.slice(item.start, item.end) === item.word) t = t.slice(0, item.start) + w + t.slice(item.end);
  else { const re = new RegExp(`\\b${item.word}\\b`); if (!re.test(t)) return; t = t.replace(re, w); }
  ta.value = t; updateCounter();
  toast(`Replaced “${item.word}” with “${w}”. Re-analyzing…`);
  return analyze();
}
function renderStyle() {
  const r = state.report, st = r.style, rt = st.pos.ratios;
  const order = Object.keys(POSNAMES).filter((k) => st.pos.counts[k] > 0);
  const nounVerb = st.pos.counts.verb ? st.pos.counts.noun / st.pos.counts.verb : 0;
  const host = $('#style');
  const spell = st.spelling;
  let spellEl;
  if (spell == null) spellEl = el('div', { class: 'loading' }, [el('span', { class: 'spin', 'aria-hidden': 'true' }), 'Checking spelling…']);
  else {
    const items = spell.filter((x) => !state.ignored.has(x.word.toLowerCase()));
    spellEl = items.length ? el('div', { class: 'spell', id: 'spelllist' }, items.map((x) => el('div', { 'data-word': x.word }, [
      el('span', { class: 'bad', text: x.word }),
      ...(x.suggestions.length ? x.suggestions.slice(0, 3).map((s) => el('button', { class: 'fix', type: 'button', 'aria-label': `Replace ${x.word} with ${s.w}`, onclick: () => applyFix(x, s.w) }, [s.w, el('small', { text: pct(s.post) })]))
        : [el('span', { class: 'note', text: 'no dictionary word within two edits' })]),
      el('button', { class: 'ign', type: 'button', text: 'Ignore', 'aria-label': `Ignore ${x.word}`, onclick: () => { state.ignored.add(x.word.toLowerCase()); renderStyle(); } }),
      el('span', { class: 'ctx', text: context(state.text, x.start, x.end) }),
    ]))) : el('p', { class: 'empty-note', text: spell.length ? 'All flagged words are ignored.' : 'No words outside the 60,000-word dictionary. (Real-word errors like “their/there” are not detected.)' });
  }
  host.replaceChildren(
    el('div', { class: 'grid2' }, [
      el('div', { class: 'stack' }, [el('div', { class: 'field-label', text: 'Parts of speech' }),
        bars(order.map((k) => [POSNAMES[k], rt[k] / Math.max(...order.map((x) => rt[x])), POSCOLORS[k], pct(rt[k], 1)])),
        el('p', { class: 'note', text: `${fmt(st.pos.total)} words tagged · ${nounVerb.toFixed(2)} nouns per verb · ${pct(rt.adj + rt.adv, 1)} adjectives and adverbs. Nominal, noun-heavy prose (more than about 2 nouns per verb) tends to read as formal or dense.` })]),
      el('div', { class: 'stack' }, [el('div', { class: 'field-label', text: `Likely passive voice · ${st.passives.length}` }),
        st.passives.length ? el('ol', { class: 'slist' }, st.passives.slice(0, 8).map((p) => el('li', {}, sentBtn(p.sentence, `“${p.phrase}”${p.byAgent ? ' · with a by-phrase' : ''}`)))) : el('p', { class: 'empty-note', text: 'No passive constructions found.' }),
        st.passives.length > 8 ? el('p', { class: 'note', text: `…and ${st.passives.length - 8} more (use the Passive voice lens above).` }) : null]),
    ]),
    el('div', { class: 'field-label', text: `Sentence lengths · mean ${st.lengths.mean.toFixed(1)} words, ${st.lengths.long.length} over 30` }),
    el('div', { class: 'plot', id: 'lens-chart', style: { height: '220px' }, role: 'img', 'aria-label': 'Bar chart of sentence lengths' }),
    el('div', { class: 'field-label', text: `Possible misspellings${spell ? ' · ' + spell.filter((x) => !state.ignored.has(x.word.toLowerCase())).length : ''}` }),
    spellEl,
  );
  drawLengths();
}
async function drawLengths() {
  let P; try { P = await plotly(); } catch { return; }
  const st = state.report.style, g = $('#lens-chart'); if (!g) return;
  await P.react(g, [{ type: 'bar', x: st.lengths.lengths.map((_, i) => i + 1), y: st.lengths.lengths, marker: { color: st.lengths.lengths.map((l) => (l > 30 ? '#d8b66a' : '#7ea3c8')) }, hovertemplate: 'Sentence %{x}: %{y} words<extra></extra>' }],
    plotLayout({ margin: { l: 40, r: 12, t: 8, b: 36 }, bargap: 0.15, xaxis: { title: { text: 'Sentence' }, color: '#9aa3ad', gridcolor: 'rgba(0,0,0,0)' }, yaxis: { color: '#9aa3ad', gridcolor: 'rgba(228,231,235,.08)' },
      shapes: [{ type: 'line', xref: 'paper', x0: 0, x1: 1, y0: 30, y1: 30, line: { color: '#d8b66a', dash: 'dot', width: 1 } }] }), PLOT_CONFIG);
  if (!g._bound) { g._bound = true; g.on('plotly_click', (e) => setActive(e.points[0].x - 1, true)); }
}

// ───────────────────────── Key ideas ─────────────────────────
function renderIdeas() {
  const r = state.report, ki = r.keyIdeas, n = r.sentences.length;
  const max = Math.max(1, Math.min(10, n));
  const both = ki.summary.filter((i) => ki.lead.includes(i)).length;
  $('#ideas').replaceChildren(
    el('div', { class: 'row' }, [el('label', { class: 'field-label', for: 'klen', text: 'Summary length' }),
      el('input', { type: 'range', id: 'klen', min: 1, max, value: Math.min(ki.summaryLength, max), 'aria-valuetext': `${ki.summaryLength} sentences`, oninput: (e) => { $('#kval').textContent = e.target.value; }, onchange: (e) => resummarize(+e.target.value) }),
      el('span', { id: 'kval', text: String(ki.summaryLength) }), el('span', { class: 'note', text: `of ${n} sentences` })]),
    el('ol', { class: 'slist', id: 'summary' }, ki.summary.map((i) => el('li', {}, sentBtn(i, `sentence ${i + 1}`)))),
    el('p', { class: 'note', text: `${both} of ${ki.summary.length} summary sentences are also among the first ${ki.summary.length} (the “lead” baseline that is hard to beat on news).` }),
    el('div', { class: 'field-label', text: 'Keywords and keyphrases' }),
    el('div', { class: 'kw', id: 'keywords' }, ki.keywords.map((k) => el('span', { class: k.phrase ? 'phrase' : '' }, [k.text, el('small', { text: k.score.toFixed(2) })]))),
  );
}
async function resummarize(k) {
  state.summaryK = k;
  const r = await summaryCall(state.text, k);
  state.report.keyIdeas.summary = r.picked; state.report.keyIdeas.lead = r.lead; state.report.keyIdeas.summaryLength = r.picked.length;
  renderIdeas(); if (state.lens === 'summary') renderDoc();
  $('#klen') && $('#klen').focus();
}

// ───────────────────────── Topics ─────────────────────────
function renderTopics() {
  const t = state.report.topics;
  if (!t.mixture.length) { $('#topics').replaceChildren(el('p', { class: 'empty-note', text: 'None of these words are in the topic model’s 4,000-word news vocabulary.' })); return; }
  const top = t.mixture.slice(0, 5), mx = top[0].p;
  $('#topics').replaceChildren(
    bars(top.map((m) => [m.label, m.p / mx, m.color, pct(m.p, 1)])),
    el('p', { class: 'note', text: `${fmt(t.known)} of ${fmt(t.total)} content words (${pct(t.coverage)}) are in the model’s vocabulary. Corpus: ${t.corpus}.` }),
    t.coverage < 0.3 ? el('p', { class: 'callout', text: 'Low coverage: most of your words are outside the newswire vocabulary, so these topics are a rough fit at best.' }) : null,
  );
}

// ───────────────────────── Entities ─────────────────────────
async function runNER() {
  const btn = $('#ner-go'); btn.disabled = true;
  const text = state.text;
  try {
    const [{ loadNER }, { progressUI }] = await Promise.all([import('./lib/ner.js'), import('/lab/shared/hf.js')]);
    const ui = progressUI($('#ner-load'), 'BERT-base NER (≈109 MB)');
    const N = await loadNER({ ui });
    $('#entities').replaceChildren(el('div', { class: 'loading' }, [el('span', { class: 'spin' }), 'Finding entities…']));
    const ents = await N.entities(text, { onProgress: (f) => ui.set(`Finding entities · ${pct(f)}`, f) });
    if (text !== state.text) return;
    state.ents = ents; ui.done(`Entities · ${ents.length} mentions`);
    renderEntities(); state.lens = 'entities'; renderDoc();
  } catch (e) {
    setMsg('Entity model failed: ' + e.message);
  } finally { btn.disabled = false; btn.textContent = 'Run again'; }
}
function aggregateEntities() {
  const by = {};
  for (const e of state.ents || []) {
    const k = e.text.replace(/\s+/g, ' ').trim(); if (!k) continue;
    const b = (by[e.type] ||= new Map()); const cur = b.get(k) || { text: k, count: 0, score: 0 };
    cur.count++; cur.score += e.score; b.set(k, cur);
  }
  return Object.fromEntries(Object.keys(ENT).map((t) => [t, [...(by[t] || new Map()).values()].sort((a, b) => b.count - a.count || b.score - a.score).map((x) => ({ text: x.text, count: x.count, score: +(x.score / x.count).toFixed(3) }))]));
}
function renderEntities() {
  const agg = aggregateEntities(), total = (state.ents || []).length;
  if (!total) { $('#entities').replaceChildren(el('p', { class: 'empty-note', text: 'No named entities found.' })); return; }
  $('#entities').replaceChildren(
    el('p', { class: 'note', text: `${total} mentions of ${Object.values(agg).reduce((a, l) => a + l.length, 0)} distinct names. The Entities lens in the Overview highlights them in your text.` }),
    el('div', { class: 'ents' }, Object.entries(ENT).map(([k, [name, color]]) => el('div', { 'data-type': k }, [
      el('h3', {}, [el('i', { style: { background: color } }), `${name} · ${agg[k].length}`]),
      agg[k].length ? el('ul', {}, agg[k].slice(0, 12).map((x) => el('li', { title: `average confidence ${pct(x.score)}` }, [el('span', { text: x.text }), el('span', { text: '×' + x.count })]))) : el('p', { class: 'empty-note', text: 'none' }),
    ]))),
  );
}

// ───────────────────────── AI-text signals ─────────────────────────
async function runSignals() {
  const btn = $('#sig-go'); btn.disabled = true;
  const text = state.text;
  try {
    const [{ loadSignals, BUCKETS }, { progressUI }] = await Promise.all([import('./lib/signals.js'), import('/lab/shared/hf.js')]);
    const ui = progressUI($('#sig-load'), 'distilgpt2 (≈85 MB)');
    const S = await loadSignals({ ui });
    $('#signals').replaceChildren(el('div', { class: 'loading' }, [el('span', { class: 'spin' }), 'Scoring tokens…']));
    const res = await S.analyze(text, { onProgress: (f) => ui.set(`Scoring tokens · ${pct(f)}`, f) });
    if (text !== state.text) return;
    state.sig = res; ui.done(`distilgpt2 · ${res.analyzed} tokens scored`);
    renderSignals(BUCKETS);
  } catch (e) {
    setMsg('Language model failed: ' + e.message);
  } finally { btn.disabled = false; btn.textContent = 'Run again'; }
}
function renderSignals(BUCKETS) {
  const s = state.sig, n = s.analyzed;
  const stat = (b, l, sub) => el('div', {}, [el('b', { text: b }), el('span', { text: l }), sub ? el('small', { text: sub }) : null]);
  const shade = (t) => { const c = BUCKETS[t.bucket].color; return c + (t.bucket === 0 ? '40' : '66'); };
  $('#signals').replaceChildren(
    el('div', { class: 'statgrid' }, [
      stat(s.ppl.toFixed(1), 'perplexity', 'lower = more predictable'),
      stat(s.meanBits.toFixed(2), 'bits per token', 'mean surprisal'),
      stat(pct(s.top10Share), 'tokens in the top 10', 'of the model’s guesses'),
      stat(`${fmt(n)}`, 'tokens analyzed', s.totalTokens > n ? `of ${fmt(s.totalTokens)} (first ${fmt(n)})` : 'the whole text'),
    ]),
    bars(BUCKETS.map((b, i) => [`Rank ${b.name}`, s.buckets[i] / n, b.color, pct(s.buckets[i] / n, 1)])),
    el('p', { class: 'note', text: 'For reference, on the lab’s Surprisal samples distilgpt2 put 49% of the tokens of a Thoreau passage in its top 10 (perplexity 90), and 92% of a passage it generated itself (perplexity 15). Short texts give noisy numbers.' }),
    el('div', { class: 'legend' }, BUCKETS.map((b) => el('span', {}, [el('i', { style: { background: b.color } }), b.name]))),
    el('div', { class: 'toks', id: 'toks', 'aria-label': 'Tokens colored by rank' }, s.tokens.map((t) => el('span', { style: { background: shade(t) }, title: `rank ${t.rank} · ${t.bits.toFixed(1)} bits · p ${t.p.toFixed(3)}`, text: t.text }))),
  );
}

// ───────────────────────── Export ─────────────────────────
function exportObject() {
  const r = state.report;
  return {
    tool: 'Braidwater Text Studio', url: location.origin + '/studio/', ...r,
    entities: state.ents ? { model: 'Xenova/bert-base-NER', mentions: state.ents.map(({ type, text, start, end, score }) => ({ type, text, start, end, score: +score.toFixed(4) })), byType: aggregateEntities() } : null,
    distilbert: state.tf ? { model: 'Xenova/distilbert-base-uncased-finetuned-sst-2-english', sentences: state.tf } : null,
    aiSignals: state.sig ? { model: state.sig.model, caveat: 'Statistical signals, not proof of authorship.', perplexity: +state.sig.ppl.toFixed(3), meanBits: +state.sig.meanBits.toFixed(4), top10Share: +state.sig.top10Share.toFixed(4), buckets: state.sig.buckets, analyzedTokens: state.sig.analyzed, totalTokens: state.sig.totalTokens } : null,
  };
}
function markdown() {
  const r = state.report, o = r.overview, s = r.sentiment.overall, st = r.style, S = (i) => r.sentences[i].text;
  const L = [];
  L.push('# Text Studio report', '', `_Braidwater Text Studio · ${new Date(r.generatedAt).toLocaleString('en-US')} · ${location.origin}/studio/_`, '');
  L.push('## Overview', '', '| Measure | Value |', '|---|---|',
    `| Words | ${fmt(o.words)} (${fmt(o.unique)} unique) |`, `| Sentences | ${o.sentences} in ${o.paragraphs} paragraph(s) |`,
    `| Reading time | ${o.readingMinutes.toFixed(1)} min (238 wpm) |`, `| Flesch reading ease | ${Math.max(0, o.fre).toFixed(0)} (${fleschLabel(o.fre)}) |`,
    `| Flesch–Kincaid grade | ${Math.max(0, o.fkgl).toFixed(1)} |`, `| Words per sentence | ${o.avgSentence.toFixed(1)} |`, `| Lexical diversity (MATTR-50) | ${o.mattr.toFixed(2)} |`, '');
  L.push('## Sentiment', '', `**Overall: ${LABEL[s.label]}** (${s.agreement} classic models agree)`, '',
    `- Lexicon + rules (VADER): ${LABEL[s.vader.label]}, mean compound ${s.vader.compound.toFixed(3)}`,
    `- Naive Bayes: ${LABEL[s.nb.label]}, P(positive) ${pct(s.nb.prob, 1)}`, `- Logistic regression: ${LABEL[s.lr.label]}, P(positive) ${pct(s.lr.prob, 1)}`);
  if (state.tf) { const m = state.tf.reduce((a, x) => a + x.prob, 0) / state.tf.length; L.push(`- DistilBERT (SST-2): mean P(positive) ${pct(m, 1)} over sentences`); }
  if (r.sentiment.mostPositive.length) L.push('', '**Most positive:**', ...r.sentiment.mostPositive.map((i) => `> ${S(i)}`, ''));
  if (r.sentiment.mostNegative.length) L.push('', '**Most negative:**', ...r.sentiment.mostNegative.map((i) => `> ${S(i)}`, ''));
  L.push('', '## Summary (TextRank)', '', ...r.keyIdeas.summary.map((i) => `- ${S(i)}`), '', `**Keywords:** ${r.keyIdeas.keywords.map((k) => k.text).join(', ')}`, '');
  L.push('## Grammar & style', '', `Parts of speech: ${Object.entries(st.pos.ratios).filter(([, v]) => v > 0).sort((a, b) => b[1] - a[1]).map(([k, v]) => `${POSNAMES[k].toLowerCase()} ${pct(v, 1)}`).join(', ')}.`, '',
    `Sentences over 30 words: ${st.lengths.long.length} · Likely passives: ${st.passives.length}`, ...st.passives.slice(0, 10).map((p) => `- “${p.phrase}” — ${S(p.sentence)}`), '');
  if (st.spelling && st.spelling.length) L.push('Possible misspellings:', ...st.spelling.filter((x) => !state.ignored.has(x.word.toLowerCase())).map((x) => `- **${x.word}** → ${x.suggestions.slice(0, 3).map((q) => q.w).join(', ') || '(no suggestion)'}`), '');
  if (r.topics.mixture.length) L.push('## Topics (Reuters newswire LDA)', '', ...r.topics.mixture.slice(0, 5).map((m) => `- ${m.label}: ${pct(m.p, 1)}`), '', `_${pct(r.topics.coverage)} of content words are in the model’s vocabulary._`, '');
  if (state.ents) { const a = aggregateEntities(); L.push('## Entities', ''); for (const [k, [n]] of Object.entries(ENT)) if (a[k].length) L.push(`- **${n}:** ${a[k].slice(0, 15).map((x) => `${x.text} (${x.count})`).join(', ')}`); L.push(''); }
  if (state.sig) L.push('## AI-text signals', '', '> Statistical signals from distilgpt2, not proof of authorship.', '', `- Perplexity ${state.sig.ppl.toFixed(1)} · ${state.sig.meanBits.toFixed(2)} bits per token · ${pct(state.sig.top10Share)} of tokens in the top 10 (${state.sig.analyzed} tokens analyzed)`, '');
  return L.join('\n');
}
function download(name, text, type) {
  const url = URL.createObjectURL(new Blob([text], { type }));
  const a = el('a', { href: url, download: name }); document.body.append(a); a.click(); a.remove();
  setTimeout(() => URL.revokeObjectURL(url), 2000);
}
let toastT;
function toast(t) { $('#toast').textContent = t; clearTimeout(toastT); toastT = setTimeout(() => { $('#toast').textContent = ''; }, 4000); }
const stamp = () => new Date().toISOString().slice(0, 10);

// ───────────────────────── section nav ─────────────────────────
function navSpy() {
  const links = $$('#snav a'), map = new Map(links.map((a) => [a.getAttribute('href').slice(1), a]));
  const io = new IntersectionObserver((es) => {
    for (const e of es) if (e.isIntersecting) { links.forEach((a) => a.removeAttribute('aria-current')); const a = map.get(e.target.id); a.setAttribute('aria-current', 'true'); a.scrollIntoView({ block: 'nearest', inline: 'nearest' }); }
  }, { rootMargin: '-30% 0px -60% 0px' });
  $$('.sec').forEach((s) => io.observe(s));
}

// ───────────────────────── init ─────────────────────────
async function init() {
  $('#samples').append(...SAMPLES.map((s) => el('button', { class: 'sample', type: 'button', 'data-sample': s.id, title: s.note, onclick: () => loadSample(s) }, [s.label, el('small', { text: s.note })])));
  $('#txt').addEventListener('input', updateCounter);
  $('#txt').addEventListener('keydown', (e) => { if (e.key === 'Enter' && (e.ctrlKey || e.metaKey)) { e.preventDefault(); analyze(); } });
  $('#go').addEventListener('click', () => analyze());
  $('#clear').addEventListener('click', () => { $('#txt').value = ''; updateCounter(); $('#txt').focus(); });
  $('#file').addEventListener('change', async (e) => {
    const f = e.target.files[0]; if (!f) return;
    if (f.size > 2e6) { setMsg('That file is over 2 MB; please paste a shorter excerpt.'); return; }
    let t = await f.text(); if (/\.(md|markdown)$/i.test(f.name)) t = stripMarkdown(t);
    $('#txt').value = t.replace(/\r/g, ''); updateCounter(); e.target.value = ''; analyze();
  });
  $('#ner-go').addEventListener('click', runNER);
  $('#sig-go').addEventListener('click', runSignals);
  $('#dl-md').addEventListener('click', () => { download(`text-studio-report-${stamp()}.md`, markdown(), 'text/markdown'); toast('Markdown report downloaded.'); });
  $('#dl-json').addEventListener('click', () => { download(`text-studio-report-${stamp()}.json`, JSON.stringify(exportObject(), null, 2), 'application/json'); toast('JSON report downloaded.'); });
  $('#copy-md').addEventListener('click', async () => {
    try { await navigator.clipboard.writeText(markdown()); toast('Report copied to the clipboard as Markdown.'); }
    catch { toast('Couldn’t reach the clipboard; use Download Markdown instead.'); }
  });
  navSpy();
  try { await warm(); } catch (e) { fail('Couldn’t load the lab’s models: ' + e.message); return; }
  $('#go').disabled = false;
  $('#status').replaceChildren(el('span', { text: 'Ready · models loaded. Ctrl+Enter analyzes.' }));
  window.__lab = Object.assign(window.__lab || {}, {
    analyze: (text) => { $('#txt').value = text; updateCounter(); return analyze(); },
    report: () => state.report, exportObject, markdown,
  });
  ready();
  if ('serviceWorker' in navigator && location.protocol === 'https:') navigator.serviceWorker.register('/studio/sw.js', { scope: '/studio/' }).catch(() => {});
  if (!$('#txt').value.trim()) await loadSample(SAMPLES[0]);
}
init();
