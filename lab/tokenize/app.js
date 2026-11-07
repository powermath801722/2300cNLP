import { el, $, ready, fail, plotly, plotLayout, PLOT_CONFIG, COLORS, getJSON } from '/lab/shared/lab.js';

// ---------------------------------------------------------------- samples (written for this page / public domain)
const SAMPLES = [
  ['Prose with abbreviations', `Dr. Ada Okafor arrived in St. Louis at 7:45 a.m. on Jan. 3. She had spent 2.5 years mapping river sediment for the U.S. Geological Survey, e.g. measuring how much silt a single flood moves. "It isn't glamorous," she said. "But the river doesn't lie." Her team's report runs to 312 pages. Mr. Hale, the county engineer, called it the best $40,000 the county ever spent. Who could argue? Nobody did.`],
  ['Social media', `omg just tried the #BraidwaterLab demo 😍🔥 tokenizers r hard lol... can't believe it's free?? check it → https://example.com/lab?ref=tw&x=1 @nlp_fan you'd love this!!! 10/10 would tokenize again 👩‍💻 questions? email ada.o@example.org`],
  ['Code & numbers', `In Python, sum(i**2 for i in range(10)) returns 285. Version 3.12.1 shipped on 2023-10-02, and the API answers {"ok": true, "ms": 12.5}. If a >= b && b != 0 then ratio = a/b. Costs rose 4.7% to $1,299.99 (about €1.2k) -> still cheaper than v2.0.`],
  ['Jane Austen, Emma (1815)', `Emma Woodhouse, handsome, clever, and rich, with a comfortable home and happy disposition, seemed to unite some of the best blessings of existence; and had lived nearly twenty-one years in the world with very little to distress or vex her. She was the youngest of the two daughters of a most affectionate, indulgent father; and had, in consequence of her sister's marriage, been mistress of his house from a very early period. Her mother had died too long ago for her to have more than an indistinct remembrance of her caresses; and her place had been supplied by an excellent woman as governess, who had fallen little short of a mother in affection.`],
];

// ---------------------------------------------------------------- tokenizer
const RULES = [
  ['url', String.raw`(?:https?:\/\/|www\.)[^\s<>"]*[^\s<>".,;:!?)\]'"]`],
  ['email', String.raw`[\p{L}\p{N}._%+-]+@[\p{L}\p{N}.-]+\.\p{L}{2,}`],
  ['mention', String.raw`@[\p{L}\p{N}_]+`],
  ['hashtag', String.raw`#[\p{L}\p{N}_]+`],
  ['emoji', String.raw`\p{Regional_Indicator}{2}|\p{Extended_Pictographic}(?:️|\p{Emoji_Modifier}|‍\p{Extended_Pictographic}️?)*`],
  ['word', String.raw`(?:\p{L}\.){2,}`],                                  // U.S.  e.g.  a.m.
  ['num', String.raw`[$€£¥]?\d+(?:[.,:\/-]\d+)*(?:%|st|nd|rd|th)?`],      // 3.14  1,299.99  7:45  2023-10-02  10/10  4.7%  3rd
  ['word', String.raw`\p{L}[\p{L}\p{M}\p{N}_]*(?:[-'’]\p{L}[\p{L}\p{M}]*)*`],
  ['sym', String.raw`==|!=|>=|<=|=>|->|\*\*|&&|\|\||::|\+\+`],
  ['punct', String.raw`\.\.\.|…|[\p{P}]`],
  ['sym', String.raw`[\p{S}]`],
];
const RX = new RegExp(RULES.map(([, r]) => `(${r})`).join('|'), 'gu');
const CLITIC = /^(.+?)(n['’]t|['’](?:s|re|ve|ll|d|m))$/iu;

/** Rule-based word tokenizer. Returns [{text, type, start, end}] (Penn-style clitic splitting). */
export function wordTokenize(text) {
  const out = [];
  for (const m of text.matchAll(RX)) {
    const gi = m.slice(1).findIndex((g) => g !== undefined);
    const type = RULES[gi][0], s = m.index, t = m[0];
    if (type === 'word') {
      const c = t.match(CLITIC);
      if (c && !/\./.test(t)) {
        out.push({ text: c[1], type: 'word', start: s, end: s + c[1].length });
        out.push({ text: c[2], type: 'clitic', start: s + c[1].length, end: s + t.length });
        continue;
      }
    }
    out.push({ text: t, type, start: s, end: s + t.length });
  }
  return out;
}

export const whitespaceTokenize = (text) => text.split(/\s+/).filter(Boolean);

const ABBR = new Set(('mr mrs ms dr prof sr jr st mt ft vs etc al inc ltd co corp dept est approx fig figs vol no nos ' +
  'jan feb mar apr jun jul aug sep sept oct nov dec gen gov sen rep capt lt col sgt rev hon pres ave blvd rd').split(' '));
const ENDERS = new Set(['.', '!', '?', '...', '…']);

/** Abbreviation-aware sentence segmentation over word tokens. Returns sentence strings. */
export function sentences(text, toks = wordTokenize(text)) {
  const bounds = [];
  for (let i = 0; i < toks.length; i++) {
    const t = toks[i];
    if (!ENDERS.has(t.text)) continue;
    // absorb closing quotes / brackets and repeated enders: ?? !!! ."
    let j = i;
    while (j + 1 < toks.length && /^(?:[.!?]|\.\.\.|…|["'”’)\]])$/u.test(toks[j + 1].text) && toks[j + 1].start === toks[j].end) j++;
    const prev = toks[i - 1], next = toks[j + 1];
    if (t.text === '.' && prev && prev.end === t.start) {
      const w = prev.text.toLowerCase();
      if (ABBR.has(w)) { i = j; continue; }                                 // Dr.  Jan.  etc.
      if (prev.type === 'word' && /^\p{Lu}$/u.test(prev.text)) { i = j; continue; } // initials: J. R.
    }
    if (next && !/^[\p{Lu}\p{N}"“'‘(\[]/u.test(next.text) && next.type !== 'emoji') { i = j; continue; } // next word lowercase → probably not a boundary
    bounds.push(toks[j].end);
    i = j;
  }
  // abbreviations written with periods (U.S., a.m.) are boundaries only if the next token is capitalized and not a name-like follower
  const parts = []; let s = 0;
  for (const b of bounds) { const p = text.slice(s, b).trim(); if (p) parts.push(p); s = b; }
  const rest = text.slice(s).trim(); if (rest) parts.push(rest);
  // blank lines always separate
  return parts.flatMap((p) => p.split(/\n\s*\n/).map((x) => x.trim()).filter(Boolean));
}

export const naiveSentences = (text) => text.split(/(?<=[.!?])\s+/).map((s) => s.trim()).filter(Boolean);

// ---------------------------------------------------------------- stats
function counts(tokens) { const m = new Map(); for (const t of tokens) m.set(t, (m.get(t) || 0) + 1); return m; }
function linfit(xs, ys) {
  const n = xs.length; let sx = 0, sy = 0, sxx = 0, sxy = 0;
  for (let i = 0; i < n; i++) { sx += xs[i]; sy += ys[i]; sxx += xs[i] * xs[i]; sxy += xs[i] * ys[i]; }
  const b = (n * sxy - sx * sy) / (n * sxx - sx * sx); return { slope: b, intercept: (sy - b * sx) / n };
}
const fmt = (n) => n.toLocaleString('en-US');

// ---------------------------------------------------------------- UI
const LEGEND = [['word', 'word'], ['clitic', 'clitic'], ['num', 'number'], ['punct', 'punctuation'], ['sym', 'symbol'], ['url', 'URL'], ['email', 'email'], ['mention', '@mention'], ['hashtag', '#hashtag'], ['emoji', 'emoji']];
const MAX_SHOW = 400;
let corpus, STOP;

function tokList(host, items) {
  host.innerHTML = '';
  const frag = document.createDocumentFragment();
  items.slice(0, MAX_SHOW).forEach((t) => frag.append(el('span', { class: 'tok' + (t.type ? ' t-' + t.type : ''), title: t.type || 'token', text: t.text })));
  if (items.length > MAX_SHOW) frag.append(el('span', { class: 'note', text: ` … +${fmt(items.length - MAX_SHOW)} more` }));
  if (!items.length) frag.append(el('span', { class: 'note', text: 'No tokens.' }));
  host.append(frag);
}

function update() {
  const text = $('#txt').value;
  const ws = whitespaceTokenize(text);
  const rx = wordTokenize(text);
  const mode = document.querySelector('input[name=sseg]:checked').value;
  const sents = mode === 'naive' ? naiveSentences(text) : sentences(text, rx);
  const words = rx.map((t) => t.text);
  const c = counts(words);
  const hapax = [...c.values()].filter((v) => v === 1).length;

  tokList($('#ws'), ws.map((t) => ({ text: t })));
  tokList($('#rx'), rx);
  const ol = $('#sents'); ol.innerHTML = '';
  sents.slice(0, 200).forEach((s) => ol.append(el('li', { text: s })));

  const st = [[fmt(rx.length), 'tokens'], [fmt(c.size), 'types'], [rx.length ? (c.size / rx.length).toFixed(3) : '—', 'type/token ratio'],
    [fmt(hapax), 'hapax (seen once)'], [fmt(sents.length), 'sentences'], [fmt(ws.length), 'whitespace tokens']];
  $('#stats').replaceChildren(...st.map(([n, l]) => el('div', {}, [el('b', { text: n }), el('span', { text: l })])));

  renderFreq(rx);
  renderPlots(rx);
  window.__lab.last = { tokens: words, types: c.size, sentences: sents, whitespace: ws.length };
}

function filtered(rx) {
  const lower = $('#lower').checked, stop = $('#stop').checked, wordsOnly = $('#wordsonly').checked;
  return rx.filter((t) => !wordsOnly || t.type === 'word' || t.type === 'clitic')
    .map((t) => (lower ? t.text.toLowerCase() : t.text))
    .filter((w) => !stop || !STOP.has(w.toLowerCase().replace(/’/g, "'")));
}

function renderFreq(rx) {
  const toks = filtered(rx), c = [...counts(toks)].sort((a, b) => b[1] - a[1] || a[0].localeCompare(b[0])).slice(0, 15);
  const max = c.length ? c[0][1] : 1;
  const tb = $('#freq tbody'); tb.innerHTML = '';
  c.forEach(([w, n], i) => tb.append(el('tr', {}, [el('td', { text: i + 1 }), el('td', { text: w }), el('td', { text: n }),
    el('td', {}, el('div', { class: 'fbar', style: { width: (n / max * 100).toFixed(1) + '%' } }))])));
  if (!c.length) tb.append(el('tr', {}, el('td', { colspan: 4, class: 'note', text: 'Nothing to count yet.' })));
}

let plotTimer;
function renderPlots(rx) {
  clearTimeout(plotTimer);
  plotTimer = setTimeout(() => drawPlots(rx), 120);
}

let zipfBase, heapsBase;
function prepCorpusPlots() {
  // text8 rank/frequency, thinned to log-spaced ranks
  const N = corpus.tokens, xs = [], ys = [];
  let last = 0;
  for (let r = 1; r <= corpus.counts.length; r++) {
    if (r < 30 || Math.log(r) - Math.log(last) > 0.02) { xs.push(r); ys.push(corpus.counts[r - 1] / N); last = r; }
  }
  const fitR = [], fitF = [];
  for (let r = 10; r <= corpus.counts.length; r++) { fitR.push(Math.log10(r)); fitF.push(Math.log10(corpus.counts[r - 1] / N)); }
  zipfBase = { xs, ys, fit: linfit(fitR, fitF), words: xs.map((r) => corpus.words[r - 1]) };
  const hx = corpus.heaps.map((p) => p[0]), hy = corpus.heaps.map((p) => p[1]);
  const big = corpus.heaps.filter((p) => p[0] >= 1000);
  heapsBase = { hx, hy, fit: linfit(big.map((p) => Math.log10(p[0])), big.map((p) => Math.log10(p[1]))) };
}

async function drawPlots(rx) {
  let P;
  try { P = await plotly(); } catch (e) { $('#zipf').textContent = e.message; return; }
  const words = rx.filter((t) => t.type === 'word' || t.type === 'clitic').map((t) => t.text.toLowerCase());
  const uc = [...counts(words).values()].sort((a, b) => b - a);
  const ux = uc.map((_, i) => i + 1), uy = uc.map((v) => v / words.length);
  const zf = zipfBase.fit;
  const fitx = [1, corpus.counts.length];
  const traces = [
    { x: zipfBase.xs, y: zipfBase.ys, text: zipfBase.words, mode: 'markers', type: 'scatter', name: 'text8 (17M tokens)', marker: { size: 4, color: COLORS[0] }, hovertemplate: 'rank %{x}: <b>%{text}</b><br>freq %{y:.2e}<extra></extra>' },
    { x: fitx, y: fitx.map((r) => 10 ** (zf.intercept + zf.slope * Math.log10(r))), mode: 'lines', type: 'scatter', name: `fit: slope ${zf.slope.toFixed(2)}`, line: { color: COLORS[0], dash: 'dot', width: 1.5 }, hoverinfo: 'skip' },
  ];
  let note = `text8 fit (ranks 10–5,000): f ∝ r^${zf.slope.toFixed(2)}.`;
  if (uc.length) {
    traces.push({ x: ux, y: uy, mode: 'markers+lines', type: 'scatter', name: 'your text', marker: { size: 5, color: COLORS[1] }, line: { color: COLORS[1], width: 1 }, hovertemplate: 'rank %{x}<br>freq %{y:.3f}<extra>your text</extra>' });
    if (uc.length >= 30) {
      const f = linfit(ux.map(Math.log10), uy.map(Math.log10));
      note += ` Your text: slope ${f.slope.toFixed(2)} over ${uc.length} types (short texts are noisy and flatter, because so many words appear once).`;
    } else note += ' Add more text (30+ distinct words) to fit a slope to yours.';
  }
  $('#zipfnote').textContent = note;
  const axis = (t) => ({ type: 'log', title: { text: t, font: { size: 12, color: '#9aa3ad' } }, gridcolor: 'rgba(228,231,235,.08)', color: '#9aa3ad', exponentformat: 'power' });
  const narrow = $('#zipf').clientWidth < 500;
  const legend = { orientation: 'h', x: 0, y: 1.02, yanchor: 'bottom', font: { size: 12 } };
  P.react('#zipf', traces, plotLayout({ xaxis: axis('rank'), yaxis: axis('relative frequency'), legend, margin: { l: narrow ? 50 : 60, r: 10, t: narrow ? 60 : 36, b: 44 } }), PLOT_CONFIG);

  // Heaps
  const hf = heapsBase.fit, uh = [], seen = new Set();
  words.forEach((w, i) => { seen.add(w); if (i < 50 || (i + 1) % Math.max(1, Math.floor(words.length / 80)) === 0 || i === words.length - 1) uh.push([i + 1, seen.size]); });
  const hx = [10, corpus.tokens];
  const ht = [
    { x: heapsBase.hx, y: heapsBase.hy, mode: 'markers', type: 'scatter', name: 'text8', marker: { size: 5, color: COLORS[0] }, hovertemplate: 'after %{x:,} tokens<br>%{y:,} types<extra></extra>' },
    { x: hx, y: hx.map((n) => 10 ** (hf.intercept + hf.slope * Math.log10(n))), mode: 'lines', type: 'scatter', name: `V = ${(10 ** hf.intercept).toFixed(1)}·N^${hf.slope.toFixed(2)}`, line: { color: COLORS[0], dash: 'dot', width: 1.5 }, hoverinfo: 'skip' },
    { x: hx, y: hx, mode: 'lines', type: 'scatter', name: 'V = N (every token new)', line: { color: 'rgba(228,231,235,.3)', width: 1 }, hoverinfo: 'skip' },
  ];
  if (uh.length) ht.push({ x: uh.map((p) => p[0]), y: uh.map((p) => p[1]), mode: 'lines', type: 'scatter', name: 'your text', line: { color: COLORS[1], width: 2 }, hovertemplate: 'after %{x} tokens<br>%{y} types<extra>your text</extra>' });
  $('#heapsnote').textContent = `Fit on text8 (N ≥ 1,000): β = ${hf.slope.toFixed(2)}. After all ${fmt(corpus.tokens)} tokens there are still ${fmt(corpus.types)} distinct types, and new ones keep arriving.`;
  P.react('#heaps', ht, plotLayout({ xaxis: axis('tokens read (N)'), yaxis: axis('distinct types (V)'), legend, margin: { l: narrow ? 50 : 60, r: 10, t: narrow ? 70 : 36, b: 44 } }), PLOT_CONFIG);
}

// ---------------------------------------------------------------- init
async function init() {
  window.__lab = Object.assign(window.__lab || {}, { wordTokenize, sentences, naiveSentences, whitespaceTokenize });
  const chips = $('#samples');
  SAMPLES.forEach(([name, text], i) => chips.append(el('button', { class: 'chip', type: 'button', text: name, 'data-i': i, onclick: () => { $('#txt').value = text; update(); } })));
  $('#legend').append(...LEGEND.map(([k, l]) => el('span', {}, [el('span', { class: 'tok t-' + k, text: ' ' }), l])));
  $('#txt').value = SAMPLES[0][1];
  try {
    corpus = await getJSON('/lab/tokenize/data/corpus.json');
  } catch (e) { fail(e.message); return; }
  STOP = new Set(corpus.stopwords);
  prepCorpusPlots();
  let t;
  $('#txt').addEventListener('input', () => { clearTimeout(t); t = setTimeout(update, 150); });
  for (const id of ['#stop', '#lower', '#wordsonly']) $(id).addEventListener('change', () => renderFreq(wordTokenize($('#txt').value)));
  document.querySelectorAll('input[name=sseg]').forEach((r) => r.addEventListener('change', update));
  update();
  ready();
}
init();
