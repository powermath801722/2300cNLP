// Semantic search: BM25 vs dense retrieval (all-MiniLM-L6-v2) vs reciprocal rank fusion, embedding map, recall@k.
import { el, $, ready, fail, plotly, plotLayout, PLOT_CONFIG, getJSON, esc } from '/lab/shared/lab.js';
import { pipe, progressUI } from '/lab/shared/hf.js';

window.__lab = window.__lab || {};
const K = 5;
const S = {};

// ---------- BM25 ----------
const STOP = new Set('a an the and or of to in on at for from by with is are was were be been it its this that these those as into than then so not no do does did can could my i me you your we our they their he she his her them how what when why which who whom will would should about there here up down out over under also only very just'.split(' '));
const tokens = (s) => (String(s).toLowerCase().match(/[a-z0-9]+(?:[-'][a-z0-9]+)*/g) || []).map((t) => t.replace(/'s$/, '')).filter((t) => !STOP.has(t));
function buildBM25(docs) {
  const tf = docs.map((d) => { const m = new Map(); for (const t of tokens(d.text)) m.set(t, (m.get(t) || 0) + 1); return m; });
  const len = tf.map((m) => [...m.values()].reduce((a, b) => a + b, 0));
  const avg = len.reduce((a, b) => a + b, 0) / len.length;
  const df = new Map(); for (const m of tf) for (const t of m.keys()) df.set(t, (df.get(t) || 0) + 1);
  const N = docs.length, k1 = 1.5, b = 0.75;
  const idf = (t) => Math.log(1 + (N - (df.get(t) || 0) + 0.5) / ((df.get(t) || 0) + 0.5));
  return (q) => {
    const qt = [...new Set(tokens(q))];
    return tf.map((m, i) => {
      let s = 0;
      for (const t of qt) { const f = m.get(t); if (f) s += idf(t) * (f * (k1 + 1)) / (f + k1 * (1 - b + b * len[i] / avg)); }
      return s;
    });
  };
}
const rankOf = (scores) => scores.map((s, i) => [i, s]).sort((a, b) => b[1] - a[1] || a[0] - b[0]);

// ---------- dense ----------
async function embedQuery(q) {
  const ui = S.ui;
  const fx = await pipe('feature-extraction', S.corpus.model, { ui, label: 'MiniLM (23 MB)', readyText: 'Ready · all-MiniLM-L6-v2 (q8, 23 MB, cached)' });
  const t = await fx(q, { pooling: 'mean', normalize: true });
  return Float32Array.from(t.data);
}
const denseScores = (qv) => S.docs.map((_, i) => { let s = 0; const o = i * S.D; for (let d = 0; d < S.D; d++) s += S.E[o + d] * qv[d]; return s; });
function rrf(...rankings) {
  const sc = new Map();
  for (const r of rankings) r.forEach(([i], k) => sc.set(i, (sc.get(i) || 0) + 1 / (60 + k + 1)));
  return [...sc].sort((a, b) => b[1] - a[1] || a[0] - b[0]);
}

// ---------- PCA map ----------
function pca2(rows, D) {
  const n = rows.length, mean = new Float64Array(D);
  for (const r of rows) for (let d = 0; d < D; d++) mean[d] += r[d] / n;
  const C = new Float64Array(D * D);
  for (const r of rows) for (let i = 0; i < D; i++) { const a = r[i] - mean[i]; for (let j = 0; j < D; j++) C[i * D + j] += a * (r[j] - mean[j]); }
  let tr = 0; for (let i = 0; i < D; i++) tr += C[i * D + i];
  const comps = [], ratio = [];
  for (let c = 0; c < 2; c++) {
    let v = new Float64Array(D).map((_, i) => Math.cos(i * 1.37 + c)), lam = 0;
    for (let it = 0; it < 200; it++) { const w = new Float64Array(D); for (let i = 0; i < D; i++) { let s = 0; for (let j = 0; j < D; j++) s += C[i * D + j] * v[j]; w[i] = s; } lam = Math.hypot(...w) || 1; v = w.map((x) => x / lam); }
    comps.push(v); ratio.push(lam / tr);
    for (let i = 0; i < D; i++) for (let j = 0; j < D; j++) C[i * D + j] -= lam * v[i] * v[j];
  }
  const proj = (r) => comps.map((v) => { let s = 0; for (let d = 0; d < D; d++) s += (r[d] - mean[d]) * v[d]; return s; });
  return { proj, ratio };
}
const topicColor = (k, n) => `hsl(${Math.round((k * 360) / n + 205) % 360} 45% 65%)`;
async function drawMap(qv, top) {
  const P = await plotly();
  const topics = [...new Set(S.docs.map((d) => d.topic))];
  const traces = topics.map((t, k) => {
    const idx = S.docs.map((d, i) => (d.topic === t ? i : -1)).filter((i) => i >= 0);
    return {
      type: 'scatter', mode: 'markers', name: t, x: idx.map((i) => S.xy[i][0]), y: idx.map((i) => S.xy[i][1]),
      text: idx.map((i) => S.docs[i].text), customdata: idx.map(() => t),
      marker: { size: 8, color: topicColor(k, topics.length), opacity: 0.85, line: { width: 0 } },
      hovertemplate: '<b>%{customdata}</b><br>%{text}<extra></extra>',
    };
  });
  if (qv) {
    const qp = S.pca.proj(qv);
    const xs = [], ys = [];
    for (const [i] of top) xs.push(qp[0], S.xy[i][0], null), ys.push(qp[1], S.xy[i][1], null);
    traces.push({ type: 'scatter', mode: 'lines', x: xs, y: ys, line: { color: 'rgba(228,231,235,.45)', width: 1 }, hoverinfo: 'skip', showlegend: false });
    traces.push({ type: 'scatter', mode: 'markers+text', name: 'your query', x: [qp[0]], y: [qp[1]], text: ['query'], textposition: 'top center', textfont: { color: '#ffffff' },
      marker: { size: 16, symbol: 'diamond', color: '#ffffff', line: { color: '#121820', width: 2 } }, hovertemplate: '<b>query</b><extra></extra>' });
  }
  const narrow = $('#map').clientWidth < 600;
  P.react($('#map'), traces, plotLayout({
    showlegend: !narrow, legend: { font: { size: 11 }, bgcolor: 'rgba(0,0,0,0)' },
    margin: { l: 44, r: 10, t: 16, b: 40 },
    xaxis: { title: { text: `PC1 (${(S.pca.ratio[0] * 100).toFixed(1)}%)`, font: { size: 12, color: '#9aa3ad' } }, gridcolor: 'rgba(228,231,235,.08)', zerolinecolor: 'rgba(228,231,235,.22)', color: '#9aa3ad' },
    yaxis: { title: { text: `PC2 (${(S.pca.ratio[1] * 100).toFixed(1)}%)`, font: { size: 12, color: '#9aa3ad' } }, gridcolor: 'rgba(228,231,235,.08)', zerolinecolor: 'rgba(228,231,235,.22)', color: '#9aa3ad' },
  }), PLOT_CONFIG);
}

// ---------- search UI ----------
function highlight(text, q) {
  const qt = new Set(tokens(q));
  return esc(text).replace(/[A-Za-z0-9]+(?:[-'][A-Za-z0-9]+)*/g, (w) => (qt.has(w.toLowerCase().replace(/'s$/, '')) ? `<mark>${w}</mark>` : w));
}
function column(title, sub, ranking, q, rel, fmt) {
  return el('div', { class: 'panel col' }, [
    el('h3', { text: title }), el('span', { class: 'note', text: sub }),
    el('ol', {}, ranking.slice(0, K).map(([i, s], r) => {
      const d = S.docs[i], isRel = rel && rel.has(d.id);
      return el('li', { class: isRel ? 'rel' : '' }, [
        el('div', { class: 'meta' }, [el('b', { text: `#${r + 1}` }), el('span', { class: 'badge', text: d.topic }), el('span', { text: fmt(s) }), isRel ? el('span', { class: 'relmark', text: '✓ relevant' }) : null]),
        el('span', { html: highlight(d.text, q) }),
      ]);
    })),
  ]);
}
async function search(q) {
  q = String(q || '').trim(); if (!q) return;
  $('#go').disabled = true;
  try {
    const labelled = S.corpus.queries.find((x) => x.q.toLowerCase() === q.toLowerCase());
    const rel = labelled ? new Set(labelled.rel) : null;
    const bm = rankOf(S.bm25(q));
    const bmNonzero = bm.filter(([, s]) => s > 0);
    $('#results').replaceChildren(column('BM25 · keywords', 'Ranks by shared words, weighted by rarity.', bmNonzero.length ? bm : [], q, rel, (s) => `score ${s.toFixed(2)}`),
      el('div', { class: 'panel col' }, [el('h3', { text: 'Dense · embeddings' }), el('span', { class: 'note', text: 'Embedding your query…' })]));
    const qv = await embedQuery(q);
    const dn = rankOf(denseScores(qv));
    const hy = rrf(bmNonzero, dn);
    $('#results').replaceChildren(
      column('BM25 · keywords', bmNonzero.length ? 'Ranks by shared words, weighted by rarity.' : 'No passage shares a content word with the query.', bmNonzero, q, rel, (s) => `score ${s.toFixed(2)}`),
      column('Dense · embeddings', 'Ranks by cosine similarity of sentence embeddings.', dn, q, rel, (s) => `cos ${s.toFixed(3)}`),
      column('Hybrid · RRF', 'Reciprocal rank fusion of both lists.', hy, q, rel, (s) => `rrf ${s.toFixed(4)}`),
    );
    window.__lab.last = { q, bm25: bmNonzero.slice(0, K).map(([i]) => S.docs[i].id), dense: dn.slice(0, K).map(([i]) => S.docs[i].id), hybrid: hy.slice(0, K).map(([i]) => S.docs[i].id) };
    await drawMap(qv, dn.slice(0, K));
  } catch (e) { $('#results').replaceChildren(el('p', { class: 'msg', text: 'Search failed: ' + e.message })); }
  finally { $('#go').disabled = false; }
}

// ---------- evaluation ----------
async function evaluate() {
  $('#evalgo').disabled = true;
  try {
    const rows = [], sums = { BM25: { r1: 0, r5: 0, mrr: 0 }, Dense: { r1: 0, r5: 0, mrr: 0 }, Hybrid: { r1: 0, r5: 0, mrr: 0 } };
    for (const [n, x] of S.corpus.queries.entries()) {
      $('#evalstatus').textContent = `Query ${n + 1} of ${S.corpus.queries.length}…`;
      const bm = rankOf(S.bm25(x.q)).filter(([, s]) => s > 0), dn = rankOf(denseScores(await embedQuery(x.q))), hy = rrf(bm, dn);
      const row = { q: x.q, why: x.why };
      for (const [name, r] of [['BM25', bm], ['Dense', dn], ['Hybrid', hy]]) {
        const ids = r.map(([i]) => S.docs[i].id), rel = new Set(x.rel);
        const rec = (k) => ids.slice(0, k).filter((id) => rel.has(id)).length / rel.size;
        const first = ids.findIndex((id) => rel.has(id));
        const m = { r1: rec(1), r5: rec(5), rr: first >= 0 ? 1 / (first + 1) : 0, rank: first >= 0 ? first + 1 : null };
        sums[name].r1 += m.r1; sums[name].r5 += m.r5; sums[name].mrr += m.rr; row[name] = m;
      }
      rows.push(row);
    }
    const n = rows.length;
    $('#evalsum').innerHTML = '<thead><tr><th>System</th><th>Recall@1</th><th>Recall@5</th><th>MRR</th></tr></thead><tbody>' +
      Object.entries(sums).map(([k, v]) => `<tr><td>${k}</td><td>${(v.r1 / n).toFixed(2)}</td><td>${(v.r5 / n).toFixed(2)}</td><td>${(v.mrr / n).toFixed(2)}</td></tr>`).join('') + '</tbody>';
    const cell = (m) => `<td class="${m.rank && m.rank <= 5 ? 'hit' : 'miss'}">${m.rank ? 'rank ' + m.rank : 'not found'}</td>`;
    $('#evaltab').innerHTML = '<thead><tr><th>Query</th><th>Type</th><th>BM25</th><th>Dense</th><th>Hybrid</th></tr></thead><tbody>' +
      rows.map((r) => `<tr><td>${esc(r.q)}</td><td>${esc(r.why)}</td>${cell(r.BM25)}${cell(r.Dense)}${cell(r.Hybrid)}</tr>`).join('') + '</tbody>';
    $('#evalstatus').textContent = 'Done. Rank of the first relevant passage per system (green = in the top 5).';
    window.__lab.eval = Object.fromEntries(Object.entries(sums).map(([k, v]) => [k, { r1: v.r1 / n, r5: v.r5 / n, mrr: v.mrr / n }]));
  } catch (e) { $('#evalstatus').textContent = 'Evaluation failed: ' + e.message; }
  finally { $('#evalgo').disabled = false; }
}

// ---------- init ----------
try {
  const [corpus, eb, sb] = await Promise.all([
    getJSON('./data/corpus.json'),
    fetch('./data/emb.bin').then((r) => { if (!r.ok) throw new Error('emb.bin ' + r.status); return r.arrayBuffer(); }),
    fetch('./data/emb-scales.bin').then((r) => { if (!r.ok) throw new Error('emb-scales.bin ' + r.status); return r.arrayBuffer(); }),
  ]);
  const q8 = new Int8Array(eb), sc = new Float32Array(sb), N = corpus.docs.length, D = q8.length / N;
  const E = new Float32Array(N * D);
  for (let i = 0; i < N; i++) { let n = 0; for (let d = 0; d < D; d++) { const v = q8[i * D + d] * sc[i]; E[i * D + d] = v; n += v * v; } n = Math.sqrt(n); for (let d = 0; d < D; d++) E[i * D + d] /= n; }
  Object.assign(S, { corpus, docs: corpus.docs, E, D, bm25: buildBM25(corpus.docs), ui: progressUI($('#load'), 'MiniLM') });
  S.ui.set('Passages and their embeddings are loaded. The query model (23 MB) downloads on your first search.', 0);
  const rows = Array.from({ length: N }, (_, i) => E.subarray(i * D, (i + 1) * D));
  S.pca = pca2(rows, D); S.xy = rows.map((r) => S.pca.proj(r));
  await drawMap(null, []);
  $('#chips').append(...corpus.queries.map((x) => el('button', { class: 'chip', type: 'button', text: x.q, title: x.why, onclick: () => { $('#q').value = x.q; search(x.q); } })));
  $('#qform').addEventListener('submit', (e) => { e.preventDefault(); search($('#q').value); });
  $('#evalgo').addEventListener('click', evaluate);
  $('#go').disabled = false; $('#evalgo').disabled = false;
  ready({ module: 'semantic-search', docs: N, dim: D });
} catch (e) { fail('Couldn’t load the search corpus: ' + e.message); }
