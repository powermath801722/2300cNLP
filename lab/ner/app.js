import { el, $, ready, fail } from '/lab/shared/lab.js';
import { pipe, progressUI } from '/lab/shared/hf.js';
import { alignTokens } from './align.js';

const MODEL = 'Xenova/bert-base-NER';
const TYPES = {
  PER: { name: 'Person', color: '#d8b66a' },
  ORG: { name: 'Organization', color: '#7ea3c8' },
  LOC: { name: 'Location', color: '#8fbf9a' },
  MISC: { name: 'Miscellaneous', color: '#b49bd6' },
};
const SAMPLES = [
  { label: 'Lovelace & Babbage', text: 'Ada Lovelace published her notes on Charles Babbage\'s Analytical Engine in London in 1843, translating an article by the Italian engineer Luigi Menabrea.', note: 'A clean newswire-style sentence: the kind of text CoNLL-2003 was built from. Note “Italian” → MISC (nationalities are MISC in CoNLL).' },
  { label: 'Which Washington?', text: 'Washington crossed the Delaware in 1776. Two centuries later, Washington imposed new tariffs, the governor of Washington objected, and the University of Washington hosted the debate.', note: 'Ambiguity: the same string is a person, a metonym for the US government (often tagged LOC in CoNLL), a state, and part of an organization name. Context decides.' },
  { label: 'Long names', text: 'Researchers from the University of California, Berkeley and the Massachusetts Institute of Technology met delegates of the Food and Agriculture Organization of the United Nations in Rome.', note: 'Long multi-word names with internal commas and nested place names. Watch whether “Berkeley” stays inside the organization or splits off as a location.' },
  { label: 'Nested entities', text: 'The Bank of England and the New York Stock Exchange both reported to the Ministry of Finance of France.', note: 'Nesting: England, New York and France are locations inside organization names. BIO allows one label per token, so the model must choose the outer or the inner entity.' },
  { label: 'No capitals', text: 'yesterday marie curie and pierre curie visited paris and then took the train to warsaw with friends from the sorbonne.', note: 'Domain shift: CoNLL text is properly capitalized and this model is cased, so lowercase input removes its strongest cue. Compare with the capitalized version.' },
  { label: 'Capitalized again', text: 'Yesterday Marie Curie and Pierre Curie visited Paris and then took the train to Warsaw with friends from the Sorbonne.', note: 'The same sentence with capitalization restored.' },
  { label: 'MISC & events', text: 'The French and German delegations discussed the Treaty of Versailles before the Olympic Games opened in Antwerp.', note: 'MISC covers nationalities, events, treaties and other names that are not people, places or organizations.' },
];

const ui = progressUI($('#load'), 'BERT-base NER (≈109 MB)');
ui.set('Model not loaded yet · ≈109 MB download on first run, then cached', 0);
let busy = false;

function legend() {
  $('#legend').replaceChildren(...Object.entries(TYPES).map(([k, t]) =>
    el('span', {}, [el('i', { style: { background: t.color } }), `${k} · ${t.name}`])));
}

function samples() {
  $('#samples').replaceChildren(...SAMPLES.map((s) => el('button', {
    class: 'chip', type: 'button', text: s.label,
    onclick: () => { $('#text').value = s.text; $('#sample-note').textContent = s.note; run(); },
  })));
}

/** Group WordPiece predictions into words (first-piece tag) and words into BIO entities. */
export function group(text, toks) {
  const offs = alignTokens(text, toks.map((t) => t.word));
  const words = [];
  toks.forEach((t, i) => {
    const o = offs[i];
    const cont = t.word.startsWith('##');
    if (cont && words.length) {
      const w = words[words.length - 1];
      w.pieces.push(t);
      if (o) w.end = o.end;
    } else {
      words.push({ tag: t.entity, start: o ? o.start : null, end: o ? o.end : null, pieces: [t] });
    }
  });
  const ents = [];
  let cur = null;
  for (const w of words) {
    const m = /^([BI])-(\w+)$/.exec(w.tag);
    if (!m) { cur = null; continue; }
    const [, bi, type] = m;
    if (bi === 'B' || !cur || cur.type !== type) {
      cur = { type, words: [w] };
      ents.push(cur);
    } else cur.words.push(w);
  }
  return ents.map((e) => {
    const pieces = e.words.flatMap((w) => w.pieces);
    const start = e.words[0].start, end = e.words[e.words.length - 1].end;
    return {
      type: e.type, start, end,
      text: start != null && end != null ? text.slice(start, end) : e.words.map((w) => w.pieces.map((p) => p.word.replace(/^##/, '')).join('')).join(' '),
      score: pieces.reduce((a, p) => a + p.score, 0) / pieces.length,
      pieces: pieces.map((p) => p.word),
    };
  });
}

function renderInline(text, ents) {
  const box = $('#inline');
  const spans = ents.filter((e) => e.start != null && e.end != null).sort((a, b) => a.start - b.start);
  const frag = [];
  let cur = 0;
  for (const e of spans) {
    if (e.start < cur) continue;
    frag.push(document.createTextNode(text.slice(cur, e.start)));
    const c = TYPES[e.type]?.color || '#9aa3ad';
    frag.push(el('mark', {
      class: 'ent', title: `${e.type} · ${(e.score * 100).toFixed(1)}%`,
      style: { background: `color-mix(in srgb, ${c} 22%, transparent)`, borderBottomColor: c },
    }, [text.slice(e.start, e.end), el('sup', { text: e.type, style: { color: c } })]));
    cur = e.end;
  }
  frag.push(document.createTextNode(text.slice(cur)));
  box.replaceChildren(...frag);
}

function renderTable(ents) {
  const tb = $('#ents tbody');
  if (!ents.length) { tb.replaceChildren(el('tr', {}, el('td', { colspan: 4, class: 'note', text: 'No entities found.' }))); return; }
  tb.replaceChildren(...ents.map((e) => el('tr', {}, [
    el('td', { text: e.text }),
    el('td', {}, el('span', { text: e.type, style: { color: TYPES[e.type]?.color, fontWeight: 600 } })),
    el('td', { text: (e.score * 100).toFixed(1) + '%' }),
    el('td', {}, el('code', { text: e.pieces.join(' ') })),
  ])));
}

function renderTokens(toks) {
  $('#tokens').replaceChildren(...toks.map((t) => {
    const type = t.entity.slice(2);
    const c = TYPES[type]?.color;
    return el('span', {
      class: 'tok' + (t.entity === 'O' ? ' o' : ''),
      style: c ? { borderColor: c, background: `color-mix(in srgb, ${c} 10%, transparent)` } : {},
      title: `${t.word}: ${t.entity} (${(t.score * 100).toFixed(1)}%)`,
    }, [t.word, el('small', { text: `${t.entity} · ${t.score.toFixed(2)}` })]);
  }));
}

async function run() {
  if (busy) return;
  const text = $('#text').value.trim();
  if (!text) { $('#status').textContent = 'Type or paste some text first.'; return; }
  busy = true; $('#run').disabled = true; $('#status').textContent = 'Running…';
  try {
    const ner = await pipe('token-classification', MODEL, { ui, label: 'BERT-base NER', readyText: 'Model ready · BERT-base NER (cached for next time)' });
    $('#status').textContent = 'Running…';
    const t0 = performance.now();
    const toks = await ner(text, { ignore_labels: [] });
    const ents = group(text, toks);
    renderInline(text, ents); renderTable(ents); renderTokens(toks);
    $('#status').textContent = `${ents.length} entit${ents.length === 1 ? 'y' : 'ies'} · ${toks.length} WordPieces · ${Math.round(performance.now() - t0)} ms`;
    window.__lab.lastEntities = ents.map(({ type, text: t, score }) => ({ type, text: t, score }));
  } catch (e) {
    $('#status').textContent = 'Something went wrong: ' + (e.message || e);
    window.__lab.runError = String(e.message || e);
  } finally { busy = false; $('#run').disabled = false; }
}

try {
  legend(); samples();
  $('#run').addEventListener('click', run);
  $('#text').addEventListener('keydown', (e) => { if (e.key === 'Enter' && (e.metaKey || e.ctrlKey)) run(); });
  ready();
} catch (e) { fail(e); }
