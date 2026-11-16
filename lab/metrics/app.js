import { el, $, ready, plotly, plotLayout, PLOT_CONFIG, COLORS } from '/lab/shared/lab.js';
import { pipe, progressUI } from '/lab/shared/hf.js';

const ENC = 'Xenova/all-MiniLM-L6-v2';
// Constructed cases. Notes are the author's reading of meaning, used to make one failure visible per case.
const CASES = [
  { name: 'Paraphrase vs. one wrong word', note: 'A faithful paraphrase shares few words with the reference; small edits that change the meaning keep almost all of them. Every metric here, including the embedding-based one, ranks the wrong date and the negation above the paraphrase.',
    refs: ['The committee postponed the vote until next week.'],
    cands: [['The panel delayed the ballot by seven days.', 'faithful paraphrase'], ['The committee postponed the vote until next year.', 'wrong date'], ['The committee did not postpone the vote until next week.', 'opposite meaning'], ['The committee postponed the vote until next week.', 'exact copy']] },
  { name: 'Swapped roles', note: 'Swapping who did what keeps every word: ROUGE-1 is perfect, and BERTScore rates the role swap above a faithful reordering of the same clauses. A paraphrase with different words scores lowest on every metric.',
    refs: ['the cat sat on the mat while the dog slept by the door'],
    cands: [['the dog sat on the mat while the cat slept by the door', 'roles swapped: different meaning'], ['while the dog slept by the door , the cat sat on the mat', 'reordered: same meaning'], ['a cat was sitting on the rug and a dog was asleep near the door', 'paraphrase: same meaning']] },
  { name: 'One reference is not enough', note: 'Both candidates are correct. Each scores well only against the reference that happens to share its wording. Add the second reference (it is already in the box) or delete it to compare.',
    refs: ['He was born in a small village near the coast.', 'His birthplace was a little seaside town.'],
    cands: [['He was born in a little seaside town.', 'correct'], ['His birthplace was a small village near the coast.', 'correct']] },
  { name: 'Short but precise', note: 'A short candidate can have high precision. BLEU’s brevity penalty and ROUGE’s recall push it down, which may or may not be what you want for summaries.',
    refs: ['After a long debate the city council voted to build a new bridge across the river next spring.'],
    cands: [['The council voted to build a bridge.', 'short, accurate'], ['After a long debate the city council voted to build a new bridge across the river next spring and also discussed parking fees and the weather.', 'complete plus padding'], ['After a long debate the city council voted against building a new bridge across the river next spring.', 'long, wrong outcome']] },
  { name: 'Morphology', note: 'Different word forms share character n-grams but few word n-grams, so chrF gives the inflected paraphrase far more credit than BLEU or ROUGE do. Every metric still ranks the opposite-meaning candidate first, because it copies the most surface form.',
    refs: ['The researchers postponed publishing their findings.'],
    cands: [['The researcher postpones publication of the finding.', 'same meaning, different forms'], ['The engineers delayed releasing their results.', 'same meaning, different words'], ['The researchers published their findings.', 'opposite meaning']] },
];

// ---------- tokenization ----------
// Close to sacreBLEU's "13a": split punctuation from words (keep 3.5, 1,000), collapse spaces.
function tok13a(s, lower) {
  let t = ' ' + s + ' ';
  t = t.replace(/([{-~[-` -&(-+:-@/])/g, ' $1 ');
  t = t.replace(/([^0-9])([.,])/g, '$1 $2 ').replace(/([.,])([^0-9])/g, ' $1 $2').replace(/([0-9])(-)/g, '$1 $2 ');
  t = t.replace(/\s+/g, ' ').trim();
  return (lower ? t.toLowerCase() : t).split(' ').filter(Boolean);
}
const rougeTok = (s) => s.toLowerCase().replace(/[^a-z0-9]+/g, ' ').trim().split(' ').filter(Boolean);
function ngrams(toks, n) { const m = new Map(); for (let i = 0; i + n <= toks.length; i++) { const g = toks.slice(i, i + n).join('\u0001'); m.set(g, (m.get(g) || 0) + 1); } return m; }

// ---------- BLEU (sentence level, multi-reference) ----------
function bleu(cand, refs, { smooth = 'exp', lower = false } = {}) {
  const c = tok13a(cand, lower), R = refs.map((r) => tok13a(r, lower));
  const rows = [];
  let logsum = 0, expK = 1, zero = false; // eslint-disable-line prefer-const
  for (let n = 1; n <= 4; n++) {
    const cg = ngrams(c, n), maxRef = new Map();
    for (const r of R) for (const [g, k] of ngrams(r, n)) maxRef.set(g, Math.max(maxRef.get(g) || 0, k));
    let match = 0, total = 0;
    for (const [g, k] of cg) { total += k; match += Math.min(k, maxRef.get(g) || 0); }
    let p;
    if (total === 0) p = 0;
    else if (match > 0) p = match / total;
    else if (smooth === 'exp') { expK *= 2; p = 1 / (expK * total); }
    else if (smooth === 'add1' && n > 1) p = 1 / (total + 1);
    else p = 0;
    if (smooth === 'add1' && n > 1 && match > 0) p = (match + 1) / (total + 1);
    rows.push({ n, match, total, p });
  }
  // Effective order (as sacreBLEU's sentence BLEU): orders the candidate is too short to contain are skipped.
  const eff = rows.filter((x) => x.total > 0);
  for (const x of eff) { if (x.p === 0) zero = true; else logsum += Math.log(x.p) / eff.length; }
  const cl = c.length;
  let rl = R[0].length; for (const r of R) if (Math.abs(r.length - cl) < Math.abs(rl - cl) || (Math.abs(r.length - cl) === Math.abs(rl - cl) && r.length < rl)) rl = r.length;
  const bp = cl === 0 ? 0 : cl > rl ? 1 : Math.exp(1 - rl / cl);
  return { score: zero || cl === 0 ? 0 : bp * Math.exp(logsum), rows, bp, cl, rl };
}

// ---------- ROUGE ----------
const f1 = (p, r) => (p + r ? (2 * p * r) / (p + r) : 0);
function rougeN(c, r, n) {
  const cg = ngrams(c, n), rg = ngrams(r, n); let m = 0, ct = 0, rt = 0;
  for (const [, k] of cg) ct += k; for (const [g, k] of rg) { rt += k; m += Math.min(k, cg.get(g) || 0); }
  const p = ct ? m / ct : 0, rc = rt ? m / rt : 0; return { p, r: rc, f: f1(p, rc) };
}
function lcs(a, b) { const d = Array.from({ length: a.length + 1 }, () => new Int32Array(b.length + 1)); for (let i = 1; i <= a.length; i++) for (let j = 1; j <= b.length; j++) d[i][j] = a[i - 1] === b[j - 1] ? d[i - 1][j - 1] + 1 : Math.max(d[i - 1][j], d[i][j - 1]); return d[a.length][b.length]; }
function rouge(cand, refs) {
  const c = rougeTok(cand); const best = { r1: null, r2: null, rl: null };
  for (const ref of refs) {
    const r = rougeTok(ref), l = lcs(c, r);
    const rl = { p: c.length ? l / c.length : 0, r: r.length ? l / r.length : 0 }; rl.f = f1(rl.p, rl.r);
    const cur = { r1: rougeN(c, r, 1), r2: rougeN(c, r, 2), rl };
    for (const k of ['r1', 'r2', 'rl']) if (!best[k] || cur[k].f > best[k].f) best[k] = cur[k];
  }
  return best;
}

// ---------- chrF ----------
function chrfOne(cand, ref, lower) {
  const prep = (s) => [...(lower ? s.toLowerCase() : s).replace(/\s+/g, '')];
  const c = prep(cand), r = prep(ref); const per = []; let P = 0, Rr = 0, k = 0;
  for (let n = 1; n <= 6; n++) {
    const cg = ngrams(c, n), rg = ngrams(r, n); let m = 0, ct = 0, rt = 0;
    for (const [, x] of cg) ct += x; for (const [g, x] of rg) { rt += x; m += Math.min(x, cg.get(g) || 0); }
    if (!ct || !rt) { per.push({ n, p: null, r: null }); continue; }
    const p = m / ct, rc = m / rt; per.push({ n, p, r: rc }); P += p; Rr += rc; k++;
  }
  P = k ? P / k : 0; Rr = k ? Rr / k : 0; const b2 = 4;
  return { score: P + Rr ? ((1 + b2) * P * Rr) / (b2 * P + Rr) : 0, P, R: Rr, per };
}
function chrf(cand, refs, lower) { let best = null; for (const r of refs) { const s = chrfOne(cand, r, lower); if (!best || s.score > best.score) best = s; } return best; }

// ---------- BERTScore with MiniLM ----------
let fx = null; const embCache = new Map();
const ui = () => (ui.v ||= progressUI($('#load'), 'MiniLM'));
async function embed(s) {
  if (embCache.has(s)) return embCache.get(s);
  const t = await fx(s, { pooling: 'none' });
  const T = t.dims[1], D = t.dims[2];
  const ids = Array.from(fx.tokenizer(s).input_ids.data, Number);
  const toks = [], vecs = [];
  for (let i = 1; i < T - 1; i++) { // drop [CLS] and [SEP], as BERTScore does
    const v = t.data.slice(i * D, (i + 1) * D); let n = 0; for (const x of v) n += x * x; n = Math.sqrt(n) || 1;
    vecs.push(v.map((x) => x / n)); toks.push(fx.tokenizer.decode([ids[i]]).replace(/^##/, '##'));
  }
  const r = { toks, vecs }; embCache.set(s, r); return r;
}
const dot = (a, b) => { let s = 0; for (let i = 0; i < a.length; i++) s += a[i] * b[i]; return s; };
async function bertscore(cand, refs) {
  const c = await embed(cand); let best = null;
  for (const ref of refs) {
    const r = await embed(ref);
    const S = c.vecs.map((cv) => r.vecs.map((rv) => dot(cv, rv)));
    const P = S.reduce((t, row) => t + Math.max(...row), 0) / S.length;
    const R = r.vecs.reduce((t, _, j) => t + Math.max(...S.map((row) => row[j])), 0) / r.vecs.length;
    const F = f1(P, R);
    if (!best || F > best.F) best = { P, R, F, S, ctoks: c.toks, rtoks: r.toks, ref };
  }
  return best;
}

// ---------- scoring the current case ----------
let results = [], sel = 0, caseIdx = 0;
const lines = (id) => $(id).value.split('\n').map((s) => s.trim()).filter(Boolean);
const opts = () => ({ smooth: $('#smooth').value, lower: $('#lower').checked });
function noteFor(cand) { const c = CASES[caseIdx]; const hit = c && c.cands.find(([t]) => t === cand); return hit ? hit[1] : ''; }

function scoreClassic(refs, cand, o) {
  const b = bleu(cand, refs, o), ro = rouge(cand, refs), ch = chrf(cand, refs, o.lower);
  return { cand, bleu: b, rouge: ro, chrf: ch, bert: null };
}
async function scoreAll(withBert) {
  const refs = lines('#refs'), cands = lines('#cands');
  if (!refs.length || !cands.length) { $('#tbl tbody').replaceChildren(el('tr', {}, el('td', { colspan: 7, text: 'Add at least one reference and one candidate.' }))); results = []; return; }
  const o = opts();
  results = cands.map((c) => scoreClassic(refs, c, o));
  if (withBert && fx) for (const r of results) r.bert = await bertscore(r.cand, refs);
  sel = Math.min(sel, results.length - 1);
  renderTable(); await renderBars(); await renderDetail();
  window.__lab.results = results.map((r) => ({ cand: r.cand, bleu: r.bleu.score, r1: r.rouge.r1.f, r2: r.rouge.r2.f, rl: r.rouge.rl.f, chrf: r.chrf.score, bert: r.bert ? r.bert.F : null }));
}
const COLS = [['BLEU', (r) => r.bleu.score], ['ROUGE-1', (r) => r.rouge.r1.f], ['ROUGE-2', (r) => r.rouge.r2.f], ['ROUGE-L', (r) => r.rouge.rl.f], ['chrF', (r) => r.chrf.score], ['BERTScore F', (r) => (r.bert ? r.bert.F : null)]];
const pc = (v) => (v == null ? '—' : (v * 100).toFixed(1));
function renderTable() {
  const best = COLS.map(([, f]) => Math.max(...results.map((r) => f(r) ?? -1)));
  $('#tbl tbody').replaceChildren(...results.map((r, i) => {
    const tr = el('tr', { 'aria-selected': String(i === sel), tabindex: '0', onclick: () => { sel = i; renderTable(); renderDetail(); }, onkeydown: (e) => { if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); sel = i; renderTable(); renderDetail(); } } }, [
      el('td', { class: 'cand' }, [r.cand, noteFor(r.cand) ? el('small', { text: noteFor(r.cand) }) : null]),
      ...COLS.map(([, f], k) => { const v = f(r); return el('td', { class: 'num' + (v != null && v === best[k] && results.length > 1 ? ' best' : ''), text: pc(v) }); }),
    ]);
    return tr;
  }));
}
async function renderBars() {
  const P = await plotly();
  const cols = COLS.filter(([, f]) => results.some((r) => f(r) != null));
  const traces = results.map((r, i) => ({ type: 'bar', name: `C${i + 1}${noteFor(r.cand) ? ': ' + noteFor(r.cand) : ''}`, x: cols.map(([n]) => n), y: cols.map(([, f]) => (f(r) ?? 0) * 100), marker: { color: COLORS[i % COLORS.length] }, hovertemplate: '%{x}: %{y:.1f}<extra>' + `C${i + 1}` + '</extra>' }));
  P.react($('#bars'), traces, plotLayout({ barmode: 'group', margin: { l: 44, r: 10, t: 10, b: 40 }, legend: { orientation: 'h', y: -0.2, font: { size: 11 } }, yaxis: { range: [0, 100], title: { text: 'score × 100' }, gridcolor: 'rgba(228,231,235,.08)', color: '#9aa3ad' } }), PLOT_CONFIG);
}
async function renderDetail() {
  const r = results[sel]; if (!r) return;
  $('#selcand').textContent = `C${sel + 1}: ${r.cand}`;
  const b = r.bleu;
  $('#bleudet').replaceChildren(el('thead', {}, el('tr', {}, ['n', 'matched / total', 'pₙ'].map((h) => el('th', { text: h })))),
    el('tbody', {}, b.rows.map((x) => el('tr', {}, [el('td', { text: x.n }), el('td', { text: `${x.match} / ${x.total}` }), el('td', { text: x.p ? x.p.toFixed(3) : '0' })]))));
  $('#bleunote').textContent = `Candidate length c = ${b.cl}, closest reference length r = ${b.rl}, brevity penalty ${b.bp.toFixed(3)}. BLEU = BP × exp(¼ Σ log pₙ) = ${(b.score * 100).toFixed(1)}.${b.rows.some((x) => x.match === 0) && $('#smooth').value === 'none' ? ' An n-gram order with no matches makes unsmoothed BLEU 0.' : ''}`;
  $('#rougedet').replaceChildren(el('thead', {}, el('tr', {}, ['', 'P', 'R', 'F1'].map((h) => el('th', { text: h })))),
    el('tbody', {}, [['ROUGE-1', r.rouge.r1], ['ROUGE-2', r.rouge.r2], ['ROUGE-L', r.rouge.rl]].map(([n, x]) => el('tr', {}, [el('td', { text: n }), el('td', { text: pc(x.p) }), el('td', { text: pc(x.r) }), el('td', { text: pc(x.f) })]))));
  $('#chrfdet').replaceChildren(el('thead', {}, el('tr', {}, ['n', 'P', 'R'].map((h) => el('th', { text: h })))),
    el('tbody', {}, [...r.chrf.per.map((x) => el('tr', {}, [el('td', { text: x.n }), el('td', { text: pc(x.p) }), el('td', { text: pc(x.r) })])), el('tr', {}, [el('td', { text: 'avg' }), el('td', { text: pc(r.chrf.P) }), el('td', { text: pc(r.chrf.R) })])]));
  if (r.bert) {
    const P = await plotly(), B = r.bert;
    const mark = B.S.map((row) => { const m = Math.max(...row); return row.map((v) => (v === m ? '○' : '')); });
    P.react($('#heat'), [{ type: 'heatmap', z: B.S, x: B.rtoks.map((t, j) => `${j}:${t}`), y: B.ctoks.map((t, i) => `${i}:${t}`), text: mark, texttemplate: '%{text}', colorscale: [[0, '#121820'], [0.5, '#416180'], [1, '#d6ebff']], hovertemplate: 'cand %{y}<br>ref %{x}<br>cos %{z:.3f}<extra></extra>', colorbar: { tickfont: { color: '#9aa3ad' }, outlinewidth: 0 } }],
      plotLayout({ margin: { l: 100, r: 10, t: 10, b: 90 }, yaxis: { autorange: 'reversed', color: '#9aa3ad', tickfont: { size: 10 } }, xaxis: { color: '#9aa3ad', tickfont: { size: 10 } } }), PLOT_CONFIG);
    $('#bertnote').textContent = `Best reference: “${B.ref}”. Precision ${pc(B.P)} (candidate tokens → reference), recall ${pc(B.R)} (reference tokens → candidate), F ${pc(B.F)}.`;
  }
}

// ---------- metric agreement across all cases ----------
function kendall(a, b) {
  let c = 0, d = 0;
  for (let i = 0; i < a.length; i++) for (let j = i + 1; j < a.length; j++) { const s = Math.sign(a[i] - a[j]) * Math.sign(b[i] - b[j]); if (s > 0) c++; else if (s < 0) d++; }
  return c + d ? (c - d) / (c + d) : 0;
}
async function agreement() {
  const btn = $('#agreebtn'); btn.disabled = true; btn.textContent = 'Computing…';
  const o = opts(), rows = [];
  for (const cs of CASES) for (const [c] of cs.cands) { const r = scoreClassic(cs.refs, c, o); if (fx) r.bert = await bertscore(c, cs.refs); rows.push(r); }
  const cols = COLS.filter(([, f]) => rows.every((r) => f(r) != null));
  const vals = cols.map(([, f]) => rows.map(f));
  const M = cols.map((_, i) => cols.map((__, j) => kendall(vals[i], vals[j])));
  $('#agree').replaceChildren(el('thead', {}, el('tr', {}, [el('th', { text: `τ (${rows.length} candidates)` }), ...cols.map(([n]) => el('th', { text: n }))])),
    el('tbody', {}, cols.map(([n], i) => el('tr', {}, [el('th', { text: n }), ...M[i].map((v, j) => el('td', { text: i === j ? '—' : v.toFixed(2) }))]))));
  window.__lab.agreement = { cols: cols.map(([n]) => n), M, n: rows.length };
  btn.disabled = false; btn.textContent = fx ? 'Recompute agreement' : 'Compute agreement (load BERTScore to include it)';
}

function loadCase(i) {
  caseIdx = i; const c = CASES[i];
  $('#refs').value = c.refs.join('\n'); $('#cands').value = c.cands.map(([t]) => t).join('\n');
  $('#casenote').textContent = c.note; sel = 0;
  [...$('#cases').children].forEach((b, k) => b.setAttribute('aria-pressed', String(k === i)));
  scoreAll(true);
}

async function loadBert() {
  const btn = $('#bert'); btn.disabled = true;
  try {
    fx = await pipe('feature-extraction', ENC, { ui: ui(), label: 'MiniLM (23 MB)', readyText: 'Ready · all-MiniLM-L6-v2 (q8, 23 MB, cached in your browser)' });
    btn.textContent = 'BERTScore loaded';
    await scoreAll(true);
  } catch (e) { btn.disabled = false; }
}

function init() {
  $('#cases').append(...CASES.map((c, i) => el('button', { class: 'chip', type: 'button', 'aria-pressed': 'false', text: c.name, onclick: () => loadCase(i) })));
  $('#score').onclick = () => { caseIdx = -1; scoreAll(true); };
  $('#bert').onclick = loadBert;
  $('#agreebtn').onclick = agreement;
  for (const id of ['#smooth', '#lower']) $(id).addEventListener('change', () => scoreAll(true));
  window.__lab = Object.assign(window.__lab || {}, { bleu, rouge, chrf, loadBert, kendall });
  loadCase(0);
  ready();
}
init();
