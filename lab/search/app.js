import { el, $, ready, fail, esc, plotly, plotLayout, PLOT_CONFIG } from '/lab/shared/lab.js';

const STOP = new Set(('a about above after again against all am an and any are as at be because been before being below between both but by can could did do does doing down during each few for from further had has have having he her here hers herself him himself his how i if in into is it its itself just me more most my myself no nor not now of off on once only or other our ours ourselves out over own same she should so some such than that the their theirs them themselves then there these they this those through to too under until up upon very was we were what when where which while who whom why will with would you your yours yourself yourselves shall may us let one also hath unto thee thou thy').split(' '));

const S = { docs: [], N: 0, stop: true, index: null };

const tokenize = (text) => (text.toLowerCase().match(/[a-z]+/g) || []).filter((w) => w.length > 1);
const keep = (w) => !(S.stop && STOP.has(w));

/** Build the inverted index: term -> { df, post: [[docId, tf]] }, plus per-doc tf maps, lengths and TF-IDF norms. */
function buildIndex() {
  const t0 = performance.now();
  const terms = new Map(), tfs = [], len = new Float64Array(S.N);
  S.docs.forEach((d, id) => {
    const tf = new Map();
    for (const w of tokenize(d.t)) if (keep(w)) tf.set(w, (tf.get(w) || 0) + 1);
    tfs.push(tf);
    let L = 0;
    for (const [w, c] of tf) {
      L += c;
      let e = terms.get(w); if (!e) terms.set(w, (e = { df: 0, post: [] }));
      e.df++; e.post.push([id, c]);
    }
    len[id] = L;
  });
  const avgdl = len.reduce((a, b) => a + b, 0) / S.N;
  const norm = new Float64Array(S.N);
  tfs.forEach((tf, id) => { let s = 0; for (const [w, c] of tf) { const x = tfw(c) * idfTf(terms.get(w).df); s += x * x; } norm[id] = Math.sqrt(s); });
  S.index = { terms, tfs, len, avgdl, norm, ms: performance.now() - t0 };
}

const tfw = (c) => (c > 0 ? 1 + Math.log10(c) : 0);
const idfTf = (df) => Math.log10(S.N / df);
const idfBm = (df) => Math.log(1 + (S.N - df + 0.5) / (df + 0.5));
const bmComp = (tf, dl, k1, b) => (tf * (k1 + 1)) / (tf + k1 * (1 - b + (b * dl) / S.index.avgdl));

function queryTerms(q) {
  const qtf = new Map();
  for (const w of tokenize(q)) if (keep(w)) qtf.set(w, (qtf.get(w) || 0) + 1);
  return qtf;
}

/** Rank with both functions; returns { tf: [{id, score}], bm: [...], qtf, known, unknown } */
export function search(q, k1 = +$('#k1').value, b = +$('#b').value) {
  const { terms, len, norm } = S.index, qtf = queryTerms(q);
  const known = [...qtf.keys()].filter((w) => terms.has(w)), unknown = [...qtf.keys()].filter((w) => !terms.has(w));
  const tfS = new Map(), bmS = new Map();
  let qn = 0;
  for (const w of known) {
    const e = terms.get(w), wq = tfw(qtf.get(w)) * idfTf(e.df), ib = idfBm(e.df);
    qn += wq * wq;
    for (const [id, c] of e.post) {
      tfS.set(id, (tfS.get(id) || 0) + wq * tfw(c) * idfTf(e.df));
      bmS.set(id, (bmS.get(id) || 0) + ib * bmComp(c, len[id], k1, b));
    }
  }
  qn = Math.sqrt(qn) || 1;
  const tf = [...tfS].map(([id, s]) => ({ id, score: s / (qn * (norm[id] || 1)) })).sort((x, y) => y.score - x.score || x.id - y.id);
  const bm = [...bmS].map(([id, s]) => ({ id, score: s })).sort((x, y) => y.score - x.score || x.id - y.id);
  return { tf, bm, qtf, known, unknown };
}

// ───────── rendering ─────────
let last = null, selected = null;
const label = (d) => `${d.y} · ${d.p} · ¶${d.i} of ${d.n}`;

function highlight(text, words, maxWords = 0) {
  const toks = text.split(/(\s+)/);
  let start = 0, end = toks.length;
  if (maxWords) {
    const first = toks.findIndex((t) => words.has((t.toLowerCase().match(/[a-z]+/) || [''])[0]));
    const wi = Math.max(0, first);
    start = Math.max(0, wi - maxWords); end = Math.min(toks.length, wi + maxWords * 2);
  }
  const out = toks.slice(start, end).map((t) => {
    if (/^\s+$/.test(t)) return t;
    return esc(t).replace(/[A-Za-z]+/g, (m) => (words.has(m.toLowerCase()) ? `<mark>${m}</mark>` : m));
  }).join('');
  return (start > 0 ? '… ' : '') + out + (end < toks.length ? ' …' : '');
}

function renderList(host, list, other, words) {
  host.replaceChildren();
  const otherRank = new Map(other.slice(0, 10).map((r, i) => [r.id, i]));
  if (!list.length) { host.append(el('li', { class: 'note', text: 'No paragraph contains any query term.' })); return; }
  list.slice(0, 10).forEach((r, i) => {
    const d = S.docs[r.id];
    const o = otherRank.get(r.id);
    const delta = o == null ? el('span', { class: 'delta down', text: 'not in other top 10' }) : o === i ? el('span', { class: 'delta', text: 'same rank' }) : el('span', { class: 'delta ' + (o > i ? 'up' : 'down'), text: `#${o + 1} in other` });
    const btn = el('button', { type: 'button', 'aria-pressed': String(selected === r.id), 'data-id': r.id, onclick: () => explain(r.id) }, [
      el('div', { class: 'hmeta' }, [el('span', {}, [el('b', { text: `#${i + 1}` }), ' ' + label(d)]), el('span', {}, [r.score.toFixed(3) + ' · ', delta])]),
      el('div', { class: 'snip', html: highlight(d.t, words, 18) }),
    ]);
    host.append(el('li', {}, btn));
  });
}

function run() {
  const q = $('#q').value.trim();
  if (!S.index) return;
  last = search(q);
  const words = new Set(last.known);
  renderList($('#hitsTf'), last.tf, last.bm, words);
  renderList($('#hitsBm'), last.bm, last.tf, words);
  $('#ntf').textContent = `${last.tf.length} matching paragraphs`;
  $('#nbm').textContent = `${last.bm.length} matching paragraphs`;
  // query term table
  const { terms } = S.index, t = $('#qterms');
  const rows = [...last.qtf.keys()].map((w) => {
    const e = terms.get(w);
    return el('tr', {}, e
      ? [el('td', { text: w }), el('td', { text: String(last.qtf.get(w)) }), el('td', { text: String(e.df) }), el('td', { text: idfTf(e.df).toFixed(3) }), el('td', { text: (tfw(last.qtf.get(w)) * idfTf(e.df)).toFixed(3) }), el('td', { text: idfBm(e.df).toFixed(3) })]
      : [el('td', { class: 'unk', text: w }), el('td', { text: String(last.qtf.get(w)) }), el('td', { text: '0' }), el('td', { class: 'unk', colspan: 3, text: 'not in any paragraph: ignored' })]);
  });
  t.replaceChildren(el('thead', {}, el('tr', {}, ['Query term', 'qtf', 'df', 'IDF = log₁₀ N/df', 'Query weight', 'BM25 IDF'].map((h) => el('th', { scope: 'col', text: h })))), el('tbody', {}, rows.length ? rows : el('tr', {}, el('td', { colspan: 6, text: S.stop ? 'Every query word is a stopword. Untick “Remove stopwords” to search for them.' : 'Type a query.' }))));
  const A = new Set(last.tf.slice(0, 10).map((r) => r.id)), B = last.bm.slice(0, 10).map((r) => r.id);
  const shared = B.filter((id) => A.has(id)).length;
  $('#overlap').textContent = last.tf.length ? `The two rankers agree on ${shared} of their top 10 · N = ${S.N.toLocaleString('en-US')} paragraphs, avg. length ${S.index.avgdl.toFixed(1)} terms after preprocessing.` : '';
  history.replaceState(null, '', '#q=' + encodeURIComponent(q));
  if (last.bm.length) explain(selected != null && (last.bm.some((r) => r.id === selected) || last.tf.some((r) => r.id === selected)) ? selected : last.bm[0].id);
  else $('#explainPanel').hidden = true;
  timeline();
}

function explain(id) {
  selected = id;
  document.querySelectorAll('ol.hits button').forEach((b) => b.setAttribute('aria-pressed', String(+b.dataset.id === id)));
  const d = S.docs[id], { terms, tfs, len, norm } = S.index, k1 = +$('#k1').value, b = +$('#b').value;
  const tf = tfs[id];
  $('#explainPanel').hidden = false;
  $('#exTitle').textContent = label(d);
  const rows = []; let cos = 0, bm = 0, qn = 0;
  for (const w of last.known) { const x = tfw(last.qtf.get(w)) * idfTf(terms.get(w).df); qn += x * x; }
  qn = Math.sqrt(qn) || 1;
  for (const w of last.known) {
    const df = terms.get(w).df, c = tf.get(w) || 0;
    const wd = tfw(c) * idfTf(df), wq = tfw(last.qtf.get(w)) * idfTf(df), cc = (wq * wd) / (qn * norm[id]);
    const bc = c ? idfBm(df) * bmComp(c, len[id], k1, b) : 0;
    cos += cc; bm += bc;
    rows.push(el('tr', {}, [el('td', { text: w }), el('td', { text: String(c) }), el('td', { text: String(df) }), el('td', { text: wd.toFixed(3) }), el('td', { text: cc.toFixed(3) }), el('td', { text: bc.toFixed(3) })]));
  }
  rows.push(el('tr', {}, [el('th', { scope: 'row', colspan: 4, text: 'Total' }), el('th', { text: cos.toFixed(3) }), el('th', { text: bm.toFixed(3) })]));
  $('#exTable').replaceChildren(el('thead', {}, el('tr', {}, ['Term', 'tf in ¶', 'df', 'w(t,d)', 'Cosine share', 'BM25 component'].map((h) => el('th', { scope: 'col', text: h })))), el('tbody', {}, rows));
  $('#exMeta').textContent = `Paragraph length ${len[id]} terms vs. average ${S.index.avgdl.toFixed(1)} (BM25 length factor ${(1 - b + (b * len[id]) / S.index.avgdl).toFixed(2)}) · TF-IDF vector norm ${norm[id].toFixed(2)}`;
  const kws = [...tf].filter(([w]) => !STOP.has(w)).map(([w, c]) => [w, tfw(c) * idfTf(terms.get(w).df)]).sort((p, q) => q[1] - p[1]).slice(0, 10);
  $('#exKw').replaceChildren(...kws.map(([w, s]) => el('span', {}, [w, el('small', { text: s.toFixed(2) })])));
  $('#exText').innerHTML = highlight(d.t, new Set(last.known));
}

let tlTimer;
function timeline() {
  clearTimeout(tlTimer);
  tlTimer = setTimeout(async () => {
    let P; try { P = await plotly(); } catch { return; }
    const addrs = [], idx = new Map();
    S.docs.forEach((d) => { const k = d.y + ' ' + d.p; if (!idx.has(k)) { idx.set(k, addrs.length); addrs.push({ k, y: d.y, p: d.p, n: 0, tot: d.n }); } });
    const seen = new Set(last.bm.map((r) => r.id));
    S.docs.forEach((d, id) => { if (seen.has(id)) addrs[idx.get(d.y + ' ' + d.p)].n++; });
    const narrow = $('#timeline').clientWidth < 560;
    P.react('timeline', [{ type: 'bar', x: addrs.map((a) => a.k), y: addrs.map((a) => a.n), marker: { color: '#7ea3c8' },
      customdata: addrs.map((a) => a.tot), hovertemplate: '%{x}<br>%{y} of %{customdata} paragraphs match<extra></extra>' }],
    plotLayout({ margin: { l: 40, r: 8, t: 8, b: narrow ? 40 : 70 }, bargap: 0.15,
      xaxis: { tickangle: -60, tickfont: { size: 9 }, color: '#9aa3ad', showgrid: false, showticklabels: !narrow, title: narrow ? { text: '1789 → 2025', font: { size: 12 } } : undefined },
      yaxis: { title: { text: 'matching ¶', font: { size: 12 } }, gridcolor: 'rgba(228,231,235,.08)', color: '#9aa3ad', rangemode: 'tozero' } }), PLOT_CONFIG);
  }, 60);
}

// ───────── wiring ─────────
$('#qform').addEventListener('submit', (e) => { e.preventDefault(); run(); });
for (const id of ['k1', 'b']) $('#' + id).addEventListener('input', () => { $('#' + id + 'o').textContent = $('#' + id).value; if (last) run(); });
$('#stop').addEventListener('change', () => { S.stop = $('#stop').checked; buildIndex(); run(); });
['liberty and union', 'slavery', 'foreign entanglements', 'depression unemployment', 'nuclear weapons', 'climate', 'space exploration', 'the people'].forEach((q) =>
  $('#chips').append(el('button', { class: 'chip', type: 'button', text: q, onclick: () => { if (q === 'the people' && $('#stop').checked) { $('#stop').checked = false; S.stop = false; buildIndex(); } $('#q').value = q; run(); } })));

(async () => {
  try {
    const r = await fetch('./data/docs.json');
    if (!r.ok) throw new Error(`collection failed to load (${r.status})`);
    const data = await r.json();
    S.docs = data.docs; S.N = S.docs.length;
    buildIndex();
    const h = new URLSearchParams(location.hash.slice(1)).get('q');
    if (h) $('#q').value = h;
    $('#loadbar').style.width = '100%';
    $('#loadtext').textContent = `Ready · ${S.N.toLocaleString('en-US')} paragraphs from ${new Set(S.docs.map((d) => d.y + d.p)).size} addresses · ${S.index.terms.size.toLocaleString('en-US')} terms indexed in ${S.index.ms.toFixed(0)} ms`;
    $('#bSearch').disabled = false;
    run();
    ready({ search: { search, N: S.N } });
  } catch (e) { fail('Couldn’t load the search collection: ' + e.message); }
})();
