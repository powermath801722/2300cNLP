import { el, $, ready, fail, plotly, plotLayout, PLOT_CONFIG } from '/lab/shared/lab.js';

const STOP = new Set(('a about above after again against all also am an and any are as at be because been before being below between both but by can could did do does doing down during each either every few for from further had has have having he her here hers herself him himself his how however i if in into is it its itself just let may me might more most must my myself neither no nor not now of off on once one only or other our ours ourselves out over own same shall she should so some such than that the their theirs them themselves then there these they this those through thus to too under until up upon us very was we were what when where whether which while who whom whose why will with within without would yet you your yours yourself yourselves hath thee thou thy unto many much'.split(' ')));

const ABBR = /\b(Mr|Mrs|Ms|Dr|St|Jr|Sr|No|vs|etc|e\.g|i\.e|U\.S|Gen|Gov|Sen|Rep|Prof)\.$/;

/** Split text into paragraphs of {text, heading} sentences. */
export function splitSentences(text) {
  const paras = text.replace(/\r/g, '').split(/\n\s*\n/).map((p) => p.replace(/\s*\n\s*/g, ' ').trim()).filter(Boolean);
  const out = [];
  paras.forEach((p, pi) => {
    if (!/[.!?]["”’')\]]*$/.test(p) && p.split(/\s+/).length < 12) { out.push({ text: p, heading: true, para: pi }); return; }
    const parts = p.split(/(?<=[.!?]["”’')\]]*)\s+(?=["“‘'(\[]?[A-Z0-9])/);
    let buf = '';
    for (const s of parts) {
      buf = buf ? buf + ' ' + s : s;
      if (ABBR.test(buf) || /\b[A-Z]\.$/.test(buf)) continue; // abbreviation or initial: keep going
      out.push({ text: buf, heading: false, para: pi }); buf = '';
    }
    if (buf) out.push({ text: buf, heading: false, para: pi });
  });
  return out;
}

const norm = (w) => (w.length > 4 && w.endsWith('ies') ? w.slice(0, -3) + 'y' : w.length > 3 && w.endsWith('s') && !w.endsWith('ss') ? w.slice(0, -1) : w);
const words = (s) => s.toLowerCase().replace(/[’']s\b/g, '').match(/[a-z]+/g) || [];
const content = (s) => words(s).filter((w) => w.length > 2 && !STOP.has(w)).map(norm);

/** Weighted PageRank on a symmetric weight matrix W (array of Float64Array). Returns { scores, trace }. */
export function pagerank(W, d = 0.85, tol = 1e-4, maxIt = 200) {
  const n = W.length, out = W.map((r) => r.reduce((a, b) => a + b, 0));
  let ws = new Float64Array(n).fill(1); const trace = [];
  for (let it = 0; it < maxIt; it++) {
    const nx = new Float64Array(n).fill(1 - d);
    for (let j = 0; j < n; j++) { if (!out[j]) continue; const f = (d * ws[j]) / out[j]; const r = W[j]; for (let i = 0; i < n; i++) if (r[i]) nx[i] += r[i] * f; }
    let delta = 0; for (let i = 0; i < n; i++) delta = Math.max(delta, Math.abs(nx[i] - ws[i]));
    ws = nx; trace.push(delta);
    if (delta < tol) break;
  }
  return { scores: ws, trace };
}

function simMatrix(sents, mode) {
  const n = sents.length, W = Array.from({ length: n }, () => new Float64Array(n));
  const toks = sents.map((s) => content(s.text));
  if (mode === 'overlap') {
    const sets = toks.map((t) => new Set(t)), lens = sents.map((s) => words(s.text).length);
    for (let i = 0; i < n; i++) for (let j = i + 1; j < n; j++) {
      let c = 0; for (const w of sets[i]) if (sets[j].has(w)) c++;
      const den = Math.log(lens[i]) + Math.log(lens[j]);
      if (c && den > 0) W[i][j] = W[j][i] = c / den;
    }
  } else {
    const df = new Map(); toks.forEach((t) => new Set(t).forEach((w) => df.set(w, (df.get(w) || 0) + 1)));
    const vecs = toks.map((t) => { const tf = new Map(); t.forEach((w) => tf.set(w, (tf.get(w) || 0) + 1)); const v = new Map(); let s = 0; for (const [w, c] of tf) { const x = c * Math.log(n / df.get(w) + 1e-9); if (x > 0) { v.set(w, x); s += x * x; } } return { v, norm: Math.sqrt(s) }; });
    for (let i = 0; i < n; i++) for (let j = i + 1; j < n; j++) {
      const a = vecs[i], b = vecs[j]; if (!a.norm || !b.norm) continue;
      let dot = 0; for (const [w, x] of a.v) { const y = b.v.get(w); if (y) dot += x * y; }
      if (dot) W[i][j] = W[j][i] = dot / (a.norm * b.norm);
    }
  }
  return W;
}

/** TextRank keyword extraction over a co-occurrence graph (window 2). */
export function keywords(text, d = 0.85) {
  const seq = (text.toLowerCase().replace(/[’']s\b/g, '').match(/[a-z]+|[^a-z\s]+/g) || []).map((t) => (/^[a-z]+$/.test(t) && t.length > 2 && !STOP.has(t) ? t : null));
  const vocab = new Map(); seq.forEach((t) => { if (t && !vocab.has(t)) vocab.set(t, vocab.size); });
  const n = vocab.size; if (!n) return [];
  const W = Array.from({ length: n }, () => new Float64Array(n));
  for (let i = 0; i + 1 < seq.length; i++) { const a = seq[i], b = seq[i + 1]; if (a && b && a !== b) { const x = vocab.get(a), y = vocab.get(b); W[x][y] = W[y][x] = 1; } }
  const { scores } = pagerank(W, d);
  const words = [...vocab.keys()];
  const T = Math.max(5, Math.min(15, Math.round(n / 3)));
  const top = new Set(words.map((w, i) => [w, scores[i]]).sort((p, q) => q[1] - p[1]).slice(0, T).map(([w]) => w));
  const sc = (w) => scores[vocab.get(w)];
  const phrases = new Map(), used = new Set();
  for (let i = 0; i < seq.length; i++) {
    if (!seq[i] || !top.has(seq[i])) continue;
    let j = i; while (j + 1 < seq.length && seq[j + 1] && top.has(seq[j + 1])) j++;
    if (j > i) { const ph = seq.slice(i, j + 1); const key = ph.join(' '); phrases.set(key, ph.reduce((a, w) => a + sc(w), 0)); ph.forEach((w) => used.add(w)); i = j; }
  }
  const singles = [...top].filter((w) => !used.has(w)).map((w) => ({ text: w, score: sc(w), phrase: false }));
  return [...[...phrases].map(([text, score]) => ({ text, score, phrase: true })), ...singles].sort((p, q) => q.score - p.score).slice(0, 15);
}

/** Full pipeline. */
export function summarize(text, { mode = 'overlap', d = 0.85, k = 3 } = {}) {
  const all = splitSentences(text).slice(0, 400);
  const cand = all.filter((s) => !s.heading);
  if (!cand.length) return { all, cand, scores: [], trace: [], picked: [], lead: [], W: [] };
  const W = simMatrix(cand, mode);
  const { scores, trace } = pagerank(W, d);
  const order = [...scores.keys()].sort((a, b) => scores[b] - scores[a]);
  const picked = order.slice(0, Math.min(k, cand.length)).sort((a, b) => a - b);
  const lead = [...cand.keys()].slice(0, Math.min(k, cand.length));
  return { all, cand, scores, trace, picked, lead, W, order };
}

// ───────── UI ─────────
let SAMPLES = [], R = null, focus = -1;

function opts() { return { mode: $('#sim').value, d: +$('#damp').value, k: +$('#k').value }; }

function run() {
  const t0 = performance.now();
  R = summarize($('#text').value, opts());
  const ms = performance.now() - t0;
  const { cand, scores, trace, picked, lead } = R;
  const n = cand.length;
  const edges = R.W.reduce((a, r) => a + r.reduce((x, v) => x + (v > 0 ? 1 : 0), 0), 0) / 2;
  $('#stat').textContent = n ? `${n} sentences · ${edges} edges · PageRank converged in ${trace.length} iterations · ${ms.toFixed(0)} ms` : 'Enter some text with at least one sentence.';
  const mx = Math.max(...scores, 1e-9);
  const rankOf = new Map((R.order || []).map((i, r) => [i, r]));
  const li = (i) => el('li', {}, [cand[i].text, el('small', { text: `S${i + 1} · score ${scores[i].toFixed(3)} · rank #${rankOf.get(i) + 1}` })]);
  $('#sumTr').replaceChildren(...picked.map(li));
  $('#sumLead').replaceChildren(...lead.map(li));
  $('#kLead').textContent = String(lead.length);
  const shared = picked.filter((i) => lead.includes(i)).length;
  $('#agree').textContent = n ? `${shared} of ${picked.length} sentences are in both summaries.` : '';
  // document view
  const doc = $('#doc'); doc.replaceChildren();
  let ci = 0, lastPara = -1, p = null;
  for (const s of R.all) {
    if (s.para !== lastPara) { p = el('p', { style: { margin: '0 0 10px' } }); doc.append(p); lastPara = s.para; }
    if (s.heading) { p.append(el('strong', { style: { color: 'var(--text)' }, text: s.text })); continue; }
    const i = ci++;
    const b = el('button', { type: 'button', class: 'sent' + (picked.includes(i) ? ' sel' : '') + (i === focus ? ' hl' : ''), style: { '--w': ((scores[i] / mx) * 100).toFixed(1) + '%' },
      'aria-label': `Sentence ${i + 1}, score ${scores[i].toFixed(3)}, rank ${rankOf.get(i) + 1}${picked.includes(i) ? ', in summary' : ''}`, onclick: () => { focus = focus === i ? -1 : i; run(); } },
    [el('span', { class: 'num', text: 'S' + (i + 1) }), s.text]);
    p.append(b, ' ');
  }
  if (focus >= 0 && focus < n) {
    const nb = [...R.W[focus]].map((w, j) => [j, w]).filter(([, w]) => w > 0).sort((a, b) => b[1] - a[1]);
    $('#stat').textContent = `S${focus + 1}: score ${scores[focus].toFixed(3)}, rank #${rankOf.get(focus) + 1}, linked to ${nb.length} sentences` + (nb.length ? ` (strongest: ${nb.slice(0, 3).map(([j, w]) => `S${j + 1} ${w.toFixed(2)}`).join(', ')})` : '') + '. Click it again to clear.';
  }
  // keywords
  const kws = keywords($('#text').value, +$('#damp').value);
  $('#kw').replaceChildren(...kws.map((k) => el('span', { class: k.phrase ? 'phrase' : '' }, [k.text, el('small', { text: k.score.toFixed(2) })])));
  draw();
}

let drawTimer;
function draw() {
  clearTimeout(drawTimer);
  drawTimer = setTimeout(async () => {
    let P; try { P = await plotly(); } catch { return; }
    const { cand, scores, picked, W, trace } = R, n = cand.length;
    if (!n) { P.purge('graph'); P.purge('conv'); return; }
    const mx = Math.max(...scores), mn = Math.min(...scores);
    const xy = cand.map((_, i) => { const a = Math.PI / 2 - (2 * Math.PI * i) / n; return [Math.cos(a), Math.sin(a)]; });
    // keep each node's 2 strongest edges plus the globally strongest ones
    const keep = new Set(), all = [];
    for (let i = 0; i < n; i++) { const r = [...W[i]].map((w, j) => [j, w]).filter(([, w]) => w > 0).sort((a, b) => b[1] - a[1]); r.slice(0, 2).forEach(([j]) => keep.add(i < j ? `${i},${j}` : `${j},${i}`)); for (const [j, w] of r) if (i < j) all.push([i, j, w]); }
    all.sort((a, b) => b[2] - a[2]).slice(0, Math.round(n * 1.5)).forEach(([i, j]) => keep.add(`${i},${j}`));
    const wmax = Math.max(...all.map((e) => e[2]), 1e-9);
    const buckets = [[], [], []];
    for (const key of keep) { const [i, j] = key.split(',').map(Number); const b = Math.min(2, Math.floor((W[i][j] / wmax) * 3)); buckets[b].push(xy[i][0], xy[j][0], null); buckets[b].push(); buckets[b].ys = (buckets[b].ys || []).concat([xy[i][1], xy[j][1], null]); }
    const lineTraces = buckets.map((b, k) => ({ type: 'scatter', mode: 'lines', x: b, y: b.ys || [], line: { color: '#7ea3c8', width: 1 + k }, opacity: 0.15 + 0.2 * k, hoverinfo: 'skip', showlegend: false }));
    const sel = new Set(picked), narrow = $('#graph').clientWidth < 520;
    const nodes = { type: 'scatter', mode: n <= 45 ? 'markers+text' : 'markers', x: xy.map((p) => p[0]), y: xy.map((p) => p[1]),
      text: cand.map((_, i) => String(i + 1)), textposition: 'middle center', textfont: { size: narrow ? 8 : 10, color: '#121820', family: 'Barlow, sans-serif' },
      customdata: cand.map((s, i) => [scores[i].toFixed(3), s.text.length > 90 ? s.text.slice(0, 88) + '…' : s.text]),
      hovertemplate: 'S%{text} · score %{customdata[0]}<br>%{customdata[1]}<extra></extra>',
      marker: { size: scores.map((s) => (narrow ? 8 : 12) + ((s - mn) / (mx - mn || 1)) * (narrow ? 18 : 30)), color: cand.map((_, i) => (sel.has(i) ? '#d8b66a' : '#7ea3c8')), line: { color: cand.map((_, i) => (i === focus ? '#ffffff' : '#121820')), width: cand.map((_, i) => (i === focus ? 3 : 1)) } }, showlegend: false };
    const ax = { visible: false, range: [-1.25, 1.25], fixedrange: true };
    P.react('graph', [...lineTraces, nodes], plotLayout({ margin: { l: 4, r: 4, t: 4, b: 4 }, xaxis: ax, yaxis: { ...ax, scaleanchor: 'x' }, dragmode: false }), { ...PLOT_CONFIG, displayModeBar: false });
    P.react('conv', [{ type: 'scatter', mode: 'lines+markers', x: trace.map((_, i) => i + 1), y: trace, line: { color: '#d8b66a', width: 2 }, marker: { size: 5 }, hovertemplate: 'iteration %{x}<br>max change %{y:.2e}<extra></extra>' }],
      plotLayout({ margin: { l: 56, r: 10, t: 8, b: 40 }, xaxis: { title: { text: 'iteration', font: { size: 12 } }, gridcolor: 'rgba(228,231,235,.08)', color: '#9aa3ad' }, yaxis: { type: 'log', title: { text: 'max |Δ score|', font: { size: 12 } }, gridcolor: 'rgba(228,231,235,.08)', color: '#9aa3ad', exponentformat: 'power' } }), PLOT_CONFIG);
  }, 40);
}

function loadSample(id) { const s = SAMPLES.find((x) => x.id === id); if (s) { $('#text').value = s.text; focus = -1; run(); } }

$('#bRun').addEventListener('click', () => { focus = -1; run(); });
$('#sample').addEventListener('change', () => loadSample($('#sample').value));
for (const id of ['damp', 'k']) $('#' + id).addEventListener('input', () => { $('#' + id + 'o').textContent = $('#' + id).value; if (R) run(); });
$('#sim').addEventListener('change', () => R && run());
let typing; $('#text').addEventListener('input', () => { clearTimeout(typing); typing = setTimeout(() => { focus = -1; run(); }, 400); });

(async () => {
  try {
    const r = await fetch('./data/samples.json');
    if (!r.ok) throw new Error(`samples failed to load (${r.status})`);
    SAMPLES = (await r.json()).samples;
    $('#sample').replaceChildren(...SAMPLES.map((s) => el('option', { value: s.id, text: s.title })));
    $('#loadbar').style.width = '100%';
    $('#loadtext').textContent = `Ready · ${SAMPLES.length} sample texts (two public-domain inaugural addresses and one article written for this lab)`;
    $('#bRun').disabled = false;
    loadSample(SAMPLES[0].id);
    ready({ textrank: { summarize, keywords, splitSentences, pagerank } });
  } catch (e) { fail('Couldn’t load the sample texts: ' + e.message); }
})();
