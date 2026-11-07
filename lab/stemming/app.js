import { el, $, ready, fail, getJSON } from '/lab/shared/lab.js';
import { porter, measure } from './porter.js';
import { lemmatize } from './lemma.js';

const EXAMPLES = ['generalizations', 'hopping', 'hoping', 'relational', 'ponies', 'caresses', 'university', 'running', 'went', 'better', 'studies', 'meeting', 'agreed', 'electricity'];
const BATCH = `The runners were running faster than they had ever run before. Organizers organized the organization's annual meeting, and the meetings generally generated generous donations. Studies of universities show that university students study universal questions; the studied answers vary. Children's feet grew, mice ate the wolves' leaves, and the happiest geese flew better than ever.`;
const OVER = [['university', 'universe', 'universal', 'universities'], ['general', 'generous', 'generation', 'generator'], ['organization', 'organ', 'organic'], ['policy', 'police', 'polite'], ['news', 'new'], ['experiment', 'experience']];
const UNDER = [['absorb', 'absorption'], ['alumnus', 'alumni'], ['matrix', 'matrices'], ['go', 'went'], ['mouse', 'mice'], ['better', 'good'], ['ran', 'run']];

let LEX;

function mdecompose(w) {
  // label runs of consonants / vowels, Porter style
  const isC = (i) => { const c = w[i]; if ('aeiou'.includes(c)) return false; if (c === 'y') return i === 0 ? true : !isC(i - 1); return true; };
  const runs = [];
  for (let i = 0; i < w.length; i++) { const k = isC(i) ? 'c' : 'v'; if (runs.length && runs[runs.length - 1].k === k) runs[runs.length - 1].s += w[i]; else runs.push({ k, s: w[i] }); }
  return runs;
}

function analyze() {
  const w = $('#word').value.trim().toLowerCase().replace(/[^a-z]/g, '');
  if (!w) return;
  const { stem, trace } = porter(w);
  $('#stem').textContent = stem;
  const fired = trace.filter((t) => t.rule && t.ok && t.before !== t.after);
  $('#stemwhy').textContent = fired.length ? `${fired.length} rule${fired.length > 1 ? 's' : ''} fired: ${fired.map((t) => 'step ' + t.step).join(', ')}` : 'No rule changed the word.';
  const L = lemmatize(w, LEX, $('#pos').value);
  $('#lemma').textContent = L.lemma;
  $('#lemmawhy').textContent = `${L.pos}: ${L.how}`;
  const tb = $('#trace tbody'); tb.innerHTML = '';
  for (const t of trace) {
    const changed = t.rule && t.ok && t.before !== t.after;
    tb.append(el('tr', { class: changed ? '' : 'skip' }, [
      el('td', { text: t.step }),
      el('td', {}, t.rule ? el('code', { text: t.rule }) : '—'),
      el('td', { text: t.rule ? (t.cond || 'none') + (t.ok ? '' : ' ✗ fails') : 'no suffix matches' }),
      el('td', { class: changed ? 'fired' : '', text: changed ? `${t.before} → ${t.after}` : t.before }),
    ]));
  }
  const runs = mdecompose(stem), m = measure(stem);
  const md = $('#mdec'); md.innerHTML = '';
  md.append(document.createTextNode(`${stem} = `));
  runs.forEach((r, i) => { md.append(el('span', { class: r.k, title: r.k === 'c' ? 'consonants' : 'vowels', text: `${r.k.toUpperCase()}(${r.s})` })); if (i < runs.length - 1) md.append(' '); });
  md.append(document.createTextNode(`  →  m = ${m}`));
  window.__lab.last = { word: w, stem, lemma: L.lemma, m, trace };
}

function batch() {
  const words = ($('#btxt').value.match(/[A-Za-z]+(?:'[a-z]+)?/g) || []).map((w) => w.toLowerCase().replace(/'s$/, ''));
  const types = [...new Set(words)];
  const rows = types.map((w) => ({ w, s: porter(w).stem, l: lemmatize(w, LEX).lemma }));
  const sTypes = new Set(rows.map((r) => r.s)).size, lTypes = new Set(rows.map((r) => r.l)).size;
  const pct = (n) => types.length ? Math.round((1 - n / types.length) * 100) + '%' : '—';
  $('#bstats').replaceChildren(...[[types.length, 'word types'], [sTypes, `stems (−${pct(sTypes)})`], [lTypes, `lemmas (−${pct(lTypes)})`]].map(([n, l]) => el('div', {}, [el('b', { text: String(n) }), el('span', { text: l })])));
  const tb = $('#btable tbody'); tb.innerHTML = '';
  rows.sort((a, b) => a.s.localeCompare(b.s) || a.w.localeCompare(b.w));
  for (const r of rows) tb.append(el('tr', {}, [el('td', { text: r.w }), el('td', { text: r.s }), el('td', { text: r.l })]));
  window.__lab.batch = { types: types.length, stems: sTypes, lemmas: lTypes };
}

function failures() {
  const over = $('#over tbody');
  for (const g of OVER) {
    const stems = g.map((w) => porter(w).stem);
    const s = stems[0];
    over.append(el('tr', {}, [el('td', { text: [...new Set(stems)].join(' / ') }), el('td', { text: g.join(', ') }), el('td', { text: g.map((w) => lemmatize(w, LEX, 'noun').lemma).join(', ') })]));
  }
  const under = $('#under tbody');
  for (const g of UNDER) {
    under.append(el('tr', {}, [el('td', { text: g.join(' / ') }), el('td', { text: g.map((w) => porter(w).stem).join(' / ') }), el('td', { text: g.map((w) => lemmatize(w, LEX).lemma).join(' / ') })]));
  }
}

async function validate() {
  const v = await getJSON('/lab/stemming/data/porter-vectors.json');
  const entries = Object.entries(v);
  const bad = entries.filter(([w, s]) => porter(w).stem !== s);
  const p = $('#valid'); p.innerHTML = '';
  p.append(el('span', { class: bad.length ? 'bad' : 'ok', text: `${entries.length - bad.length} / ${entries.length}` }),
    ` words stem identically to NLTK’s PorterStemmer (original-algorithm mode). The test set mixes Porter’s own examples with words sampled across the frequency range of text8.`);
  if (bad.length) p.append(el('span', { text: ' Mismatches: ' + bad.slice(0, 10).map(([w, s]) => `${w} (NLTK ${s}, here ${porter(w).stem})`).join(', ') }));
  window.__lab.validation = { total: entries.length, mismatches: bad.length };
}

async function init() {
  window.__lab = Object.assign(window.__lab || {}, { porter, lemmatize: (w, pos) => lemmatize(w, LEX, pos) });
  try {
    const v = await getJSON('/word2vec/model/vocab.json');
    LEX = new Map(v.words.map((w, i) => [w, v.counts[i]]));
  } catch (e) { fail(e.message); return; }
  $('#wchips').append(...EXAMPLES.map((w) => el('button', { class: 'chip', type: 'button', text: w, onclick: () => { $('#word').value = w; analyze(); } })));
  $('#wform').addEventListener('submit', (e) => { e.preventDefault(); analyze(); });
  $('#pos').addEventListener('change', analyze);
  $('#btxt').value = BATCH;
  let t; $('#btxt').addEventListener('input', () => { clearTimeout(t); t = setTimeout(batch, 150); });
  analyze(); batch(); failures();
  await validate().catch((e) => { $('#valid').textContent = 'Validation data failed to load: ' + e.message; });
  ready();
}
init();
