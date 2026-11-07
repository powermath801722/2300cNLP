import { el, $, ready, fail, REDUCED } from '/lab/shared/lab.js';

// ───────────────────────── Edit distance ─────────────────────────
function costs() {
  return { ins: +$('#cIns').value, del: +$('#cDel').value, sub: +$('#cSub').value, trans: $('#cTrans').checked ? 1 : null };
}

/** Full DP table (array of Float64Array rows), Damerau OSA when c.trans != null. */
export function editTable(a, b, c) {
  const m = a.length, n = b.length, D = [];
  for (let i = 0; i <= m; i++) { D.push(new Float64Array(n + 1)); D[i][0] = i * c.del; }
  for (let j = 0; j <= n; j++) D[0][j] = j * c.ins;
  for (let i = 1; i <= m; i++) for (let j = 1; j <= n; j++) {
    let v = Math.min(D[i - 1][j] + c.del, D[i][j - 1] + c.ins, D[i - 1][j - 1] + (a[i - 1] === b[j - 1] ? 0 : c.sub));
    if (c.trans != null && i > 1 && j > 1 && a[i - 1] === b[j - 2] && a[i - 2] === b[j - 1]) v = Math.min(v, D[i - 2][j - 2] + c.trans);
    D[i][j] = v;
  }
  return D;
}

/** Backtrace → forward list of ops {op:'M'|'S'|'D'|'I'|'T', a, b} and the set of path cells "i,j". */
export function backtrace(a, b, D, c) {
  let i = a.length, j = b.length; const ops = [], cells = new Set([`${i},${j}`]);
  while (i > 0 || j > 0) {
    if (i > 0 && j > 0 && D[i][j] === D[i - 1][j - 1] + (a[i - 1] === b[j - 1] ? 0 : c.sub)) {
      ops.push({ op: a[i - 1] === b[j - 1] ? 'M' : 'S', a: a[i - 1], b: b[j - 1] }); i--; j--;
    } else if (c.trans != null && i > 1 && j > 1 && a[i - 1] === b[j - 2] && a[i - 2] === b[j - 1] && D[i][j] === D[i - 2][j - 2] + c.trans) {
      ops.push({ op: 'T', a: a[i - 2] + a[i - 1], b: b[j - 2] + b[j - 1] }); i -= 2; j -= 2;
    } else if (i > 0 && D[i][j] === D[i - 1][j] + c.del) {
      ops.push({ op: 'D', a: a[i - 1], b: '' }); i--;
    } else {
      ops.push({ op: 'I', a: '', b: b[j - 1] }); j--;
    }
    cells.add(`${i},${j}`);
  }
  return { ops: ops.reverse(), cells };
}

const ed = { a: '', b: '', c: null, D: null, k: 0, total: 0, timer: null, bt: null };

function cellAt(k) { const w = ed.b.length + 1; return [Math.floor(k / w), k % w]; }

function buildTable() {
  const { a, b } = ed;
  const t = el('table', { class: 'dp' });
  const head = el('tr', {}, [el('th', { scope: 'col', 'aria-label': 'corner' }), el('th', { scope: 'col', text: '#' }), ...[...b].map((ch) => el('th', { scope: 'col', text: ch }))]);
  t.append(el('thead', {}, head));
  const tb = el('tbody');
  for (let i = 0; i <= a.length; i++) {
    const tr = el('tr', {}, el('th', { scope: 'row', text: i === 0 ? '#' : a[i - 1] }));
    for (let j = 0; j <= b.length; j++) tr.append(el('td', { 'data-c': `${i},${j}`, class: 'empty' }));
    tb.append(tr);
  }
  t.append(tb);
  $('#dpwrap').replaceChildren(t);
}

function paint() {
  const { a, b, D, c, k, total } = ed;
  const cells = $('#dpwrap').querySelectorAll('td');
  const done = k >= total;
  const [ci, cj] = k > 0 ? cellAt(k - 1) : [-1, -1];
  const preds = new Set();
  if (!done && k > 0) {
    if (ci > 0) preds.add(`${ci - 1},${cj}`);
    if (cj > 0) preds.add(`${ci},${cj - 1}`);
    if (ci > 0 && cj > 0) preds.add(`${ci - 1},${cj - 1}`);
    if (c.trans != null && ci > 1 && cj > 1 && a[ci - 1] === b[cj - 2] && a[ci - 2] === b[cj - 1]) preds.add(`${ci - 2},${cj - 2}`);
  }
  cells.forEach((td, idx) => {
    const filled = idx < k;
    td.textContent = filled ? fmt(D[Math.floor(idx / (b.length + 1))][idx % (b.length + 1)]) : '';
    td.className = filled ? '' : 'empty';
    const key = td.dataset.c;
    if (done && ed.bt.cells.has(key)) td.classList.add('path');
    else if (!done && idx === k - 1) td.classList.add('cur');
    else if (preds.has(key)) td.classList.add('pred');
  });
  // explanation
  const ex = $('#explain');
  if (done) {
    ex.textContent = `Done: D[${a.length}, ${b.length}] = ${fmt(D[a.length][b.length])}. Highlighted cells are one minimum-cost path, traced back from the bottom-right corner.`;
  } else if (k === 0) {
    ex.textContent = 'Press Step or Play to fill the table one cell at a time.';
  } else if (ci === 0 || cj === 0) {
    ex.textContent = ci === 0 ? `D[0, ${cj}] = ${cj} × insertion cost = ${fmt(D[0][cj])} (build the first ${cj} target letters from nothing)` : `D[${ci}, 0] = ${ci} × deletion cost = ${fmt(D[ci][0])} (delete the first ${ci} source letters)`;
  } else {
    const same = a[ci - 1] === b[cj - 1];
    const parts = [`delete ${a[ci - 1]}: ${fmt(D[ci - 1][cj])}+${c.del}=${fmt(D[ci - 1][cj] + c.del)}`,
      `insert ${b[cj - 1]}: ${fmt(D[ci][cj - 1])}+${c.ins}=${fmt(D[ci][cj - 1] + c.ins)}`,
      same ? `match ${a[ci - 1]}: ${fmt(D[ci - 1][cj - 1])}+0=${fmt(D[ci - 1][cj - 1])}` : `substitute ${a[ci - 1]}→${b[cj - 1]}: ${fmt(D[ci - 1][cj - 1])}+${c.sub}=${fmt(D[ci - 1][cj - 1] + c.sub)}`];
    if (c.trans != null && ci > 1 && cj > 1 && a[ci - 1] === b[cj - 2] && a[ci - 2] === b[cj - 1]) parts.push(`transpose ${a[ci - 2]}${a[ci - 1]}: ${fmt(D[ci - 2][cj - 2])}+${c.trans}=${fmt(D[ci - 2][cj - 2] + c.trans)}`);
    ex.textContent = `D[${ci}, ${cj}] = min( ${parts.join(' ; ')} ) = ${fmt(D[ci][cj])}`;
  }
  $('#bStep').disabled = done; $('#bFill').disabled = done;
}

const fmt = (v) => (Number.isInteger(v) ? String(v) : v.toFixed(1));

function renderResult() {
  const { a, b, D, bt } = ed;
  $('#dist').textContent = fmt(D[a.length][b.length]);
  const n = { S: 0, D: 0, I: 0, T: 0, M: 0 };
  bt.ops.forEach((o) => n[o.op]++);
  $('#opsum').textContent = `${n.M} match${n.M === 1 ? '' : 'es'}, ${n.S} substitution${n.S === 1 ? '' : 's'}, ${n.D} deletion${n.D === 1 ? '' : 's'}, ${n.I} insertion${n.I === 1 ? '' : 's'}` + (ed.c.trans != null ? `, ${n.T} transposition${n.T === 1 ? '' : 's'}` : '');
  const al = $('#align'); al.replaceChildren();
  al.style.gridTemplateRows = 'auto auto auto';
  for (const o of bt.ops) {
    const col = [el('span', { class: o.op, text: o.a || '*' }), el('span', { class: 'op ' + o.op, text: o.op === 'M' ? '·' : o.op }), el('span', { class: o.op, text: o.b || '*' })];
    col.forEach((s) => al.append(s));
  }
  al.setAttribute('aria-label', 'Alignment: ' + bt.ops.map((o) => ({ M: `keep ${o.a}`, S: `${o.a} to ${o.b}`, D: `delete ${o.a}`, I: `insert ${o.b}`, T: `swap ${o.a}` }[o.op])).join(', '));
}

function compute(fillAll = true) {
  stop();
  const a = $('#src').value.trim().toLowerCase().slice(0, 24), b = $('#tgt').value.trim().toLowerCase().slice(0, 24);
  ed.a = a; ed.b = b; ed.c = costs();
  ed.D = editTable(a, b, ed.c); ed.bt = backtrace(a, b, ed.D, ed.c);
  ed.total = (a.length + 1) * (b.length + 1); ed.k = fillAll ? ed.total : 0;
  buildTable(); paint(); renderResult();
}

function step() { if (ed.k < ed.total) { ed.k++; paint(); } if (ed.k >= ed.total) stop(); }
function stop() { clearInterval(ed.timer); ed.timer = null; $('#bPlay').textContent = 'Play'; $('#bPlay').setAttribute('aria-pressed', 'false'); }
function play() {
  if (ed.timer) return stop();
  if (ed.k >= ed.total) { ed.k = 0; paint(); }
  $('#bPlay').textContent = 'Pause'; $('#bPlay').setAttribute('aria-pressed', 'true');
  ed.timer = setInterval(step, +$('#speed').value);
}

$('#edform').addEventListener('submit', (e) => { e.preventDefault(); compute(true); });
['#cIns', '#cDel', '#cSub', '#cTrans'].forEach((s) => $(s).addEventListener('change', () => compute(true)));
$('#bReset').addEventListener('click', () => { stop(); ed.k = 0; paint(); });
$('#bStep').addEventListener('click', () => { stop(); step(); });
$('#bPlay').addEventListener('click', play);
$('#bFill').addEventListener('click', () => { stop(); ed.k = ed.total; paint(); });
$('#speed').addEventListener('input', () => { if (ed.timer) { stop(); play(); } });
[['intention', 'execution'], ['kitten', 'sitting'], ['teh', 'the'], ['acress', 'actress'], ['receive', 'recieve'], ['algorithm', 'altruistic']].forEach(([x, y]) =>
  $('#edchips').append(el('button', { class: 'chip', type: 'button', text: `${x} → ${y}`, onclick: () => {
    $('#src').value = x; $('#tgt').value = y;
    if (x === 'teh' || x === 'receive') $('#cTrans').checked = true;
    compute(true);
  } })));
compute(true);

// ───────────────────────── Noisy-channel corrector ─────────────────────────
const WORDS = new Map(); let TOTAL = 0;
const LETTERS = 'abcdefghijklmnopqrstuvwxyz';

// QWERTY geometry for the keyboard-aware channel model
const KEYPOS = {};
['qwertyuiop', 'asdfghjkl', 'zxcvbnm'].forEach((row, r) => [...row].forEach((ch, c) => { KEYPOS[ch] = [c + [0, 0.25, 0.75][r], r]; }));
function adjacent(x, y) { const p = KEYPOS[x], q = KEYPOS[y]; if (!p || !q) return false; return Math.hypot(p[0] - q[0], p[1] - q[1]) < 1.3; }

// Per-edit probabilities of the channel model (illustrative, not learned from data)
const P_EDIT = { subNear: 0.01, subFar: 0.002, del: 0.004, delDouble: 0.01, ins: 0.004, insDouble: 0.01, trans: 0.008 };
const P_CORRECT = 0.95; // probability a word is typed with no error

function edits1(w) {
  const out = new Set();
  for (let i = 0; i <= w.length; i++) {
    const L = w.slice(0, i), R = w.slice(i);
    if (R) out.add(L + R.slice(1));
    if (R.length > 1) out.add(L + R[1] + R[0] + R.slice(2));
    for (const ch of LETTERS) { if (R) out.add(L + ch + R.slice(1)); out.add(L + ch + R); }
  }
  out.delete(w);
  return out;
}

/** Map candidate → edit distance (0, 1 or 2), dictionary words only. */
function candidateSet(x) {
  const res = new Map();
  if (WORDS.has(x)) res.set(x, 0);
  const e1 = edits1(x);
  for (const w of e1) if (WORDS.has(w) && !res.has(w)) res.set(w, 1);
  if (x.length <= 14) for (const w1 of e1) for (const w2 of edits1(w1)) if (WORDS.has(w2) && !res.has(w2)) res.set(w2, 2);
  return res;
}

const DAMERAU = { ins: 1, del: 1, sub: 1, trans: 1 };
/** P(x | w): align intended word w to typed string x and multiply per-edit probabilities. */
function channel(x, w) {
  if (x === w) return { p: P_CORRECT, ops: [] };
  const D = editTable(w, x, DAMERAU), { ops } = backtrace(w, x, D, DAMERAU);
  let p = 1; const desc = [];
  ops.forEach((o, k) => {
    if (o.op === 'M') return;
    if (o.op === 'S') { const near = adjacent(o.a, o.b); p *= near ? P_EDIT.subNear : P_EDIT.subFar; desc.push(`${o.a}→${o.b}${near ? ' (near key)' : ''}`); }
    else if (o.op === 'T') { p *= P_EDIT.trans; desc.push(`swap ${o.a}`); }
    else if (o.op === 'D') { const prev = ops[k - 1], dbl = (prev && prev.a === o.a) || (ops[k + 1] && ops[k + 1].a === o.a); p *= dbl ? P_EDIT.delDouble : P_EDIT.del; desc.push(`drop ${o.a}${dbl ? ' (double letter)' : ''}`); }
    else { const prev = ops[k - 1], dbl = (prev && prev.b === o.b) || (ops[k + 1] && ops[k + 1].b === o.b); p *= dbl ? P_EDIT.insDouble : P_EDIT.ins; desc.push(`add ${o.b}${dbl ? ' (double letter)' : ''}`); }
  });
  return { p: p * P_CORRECT, ops: desc };
}

/** Ranked candidates for typed word x. */
export function rank(x, model = $('#model').value) {
  x = x.toLowerCase();
  const cs = candidateSet(x), rows = [];
  for (const [w, d] of cs) {
    const prior = WORDS.get(w) / TOTAL, ch = channel(x, w);
    rows.push({ w, d, count: WORDS.get(w), prior, chan: ch.p, ops: ch.ops, score: model === 'norvig' ? -d * 100 + Math.log10(prior) : Math.log10(prior) + Math.log10(ch.p) });
  }
  if (model === 'norvig') {
    // Norvig: known word if present, else most frequent among the fewest-edit tier.
    const best = Math.min(...rows.map((r) => r.d));
    rows.forEach((r) => { r.inTier = r.d === best; });
  }
  rows.sort((p, q) => q.score - p.score);
  // posterior over listed candidates (noisy channel)
  const top = rows.slice(0, 60), mx = top.length ? top[0].score : 0;
  let z = 0; top.forEach((r) => { r.post = Math.pow(10, r.score - mx); z += r.post; });
  top.forEach((r) => { r.post /= z; });
  return top;
}

function showCandidates(x) {
  const model = $('#model').value, t0 = performance.now();
  const rows = rank(x, model), ms = performance.now() - t0;
  const known = WORDS.has(x.toLowerCase());
  $('#candinfo').textContent = rows.length
    ? `${rows.length === 60 ? '60+' : rows.length} dictionary word${rows.length === 1 ? '' : 's'} within two edits of “${x}”${known ? ' (it is itself a known word, so it is kept)' : ''} · ${ms.toFixed(0)} ms`
    : `No dictionary word within two edits of “${x}”.`;
  const head = model === 'norvig'
    ? ['#', 'Candidate', 'Edits', 'Count', 'P(w)']
    : ['#', 'Candidate', 'Edits', 'log₁₀ P(w)', 'log₁₀ P(x|w)', 'Score', 'Posterior'];
  const t = $('#cands'); t.replaceChildren(el('thead', {}, el('tr', {}, head.map((h) => el('th', { scope: 'col', text: h })))));
  const tb = el('tbody');
  rows.slice(0, 10).forEach((r, i) => {
    const name = el('td', {}, [el('b', { class: i === 0 ? 'cand-ok' : '', text: r.w }), r.ops.length ? el('div', { class: 'note', style: { fontSize: '12px' }, text: r.ops.join(', ') }) : null]);
    const cells = model === 'norvig'
      ? [String(i + 1), name, String(r.d), r.count.toLocaleString('en-US'), r.prior.toExponential(2)]
      : [String(i + 1), name, String(r.d), Math.log10(r.prior).toFixed(2), Math.log10(r.chan).toFixed(2), r.score.toFixed(2),
        el('td', {}, [el('div', { text: (r.post * 100).toFixed(1) + '%' }), el('div', { class: 'wbar' }, el('i', { style: { width: (r.post * 100).toFixed(1) + '%' } }))])];
    tb.append(el('tr', {}, cells.map((c) => (c instanceof Node && c.tagName === 'TD' ? c : el('td', {}, c)))));
  });
  t.append(tb);
}

/** Correct a whole sentence; returns [{orig, fixed|null, nos}] tokens. */
export function correctSentence(text, model = $('#model').value) {
  const out = [];
  for (const tok of text.match(/[A-Za-z]+|[^A-Za-z]+/g) || []) {
    if (!/^[A-Za-z]+$/.test(tok)) { out.push({ orig: tok }); continue; }
    const lw = tok.toLowerCase();
    if (WORDS.has(lw) || lw.length < 2) { out.push({ orig: tok }); continue; }
    const best = rank(lw, model)[0];
    if (!best) { out.push({ orig: tok, nos: true }); continue; }
    let fixed = best.w;
    if (tok[0] === tok[0].toUpperCase()) fixed = fixed[0].toUpperCase() + fixed.slice(1);
    out.push({ orig: tok, fixed });
  }
  return out;
}

function runSentence() {
  const res = correctSentence($('#sent').value);
  const host = $('#corrected'); host.replaceChildren();
  let n = 0;
  for (const r of res) {
    if (r.fixed) {
      n++;
      host.append(el('button', { class: 'fix', type: 'button', title: `Show candidates for “${r.orig}”`, 'aria-label': `${r.orig} corrected to ${r.fixed}; show candidates`,
        onclick: () => { $('#word').value = r.orig; showCandidates(r.orig); } }, [el('small', { text: r.orig }), r.fixed]));
    } else if (r.nos) host.append(el('span', { class: 'nos', title: 'No candidate within two edits', text: r.orig }));
    else host.append(document.createTextNode(r.orig));
  }
  host.dataset.changes = String(n);
}

$('#bCorrect').addEventListener('click', runSentence);
$('#wform').addEventListener('submit', (e) => { e.preventDefault(); const w = $('#word').value.trim(); if (w) showCandidates(w.replace(/[^A-Za-z]/g, '')); });
$('#model').addEventListener('change', () => { runSentence(); const w = $('#word').value.trim(); if (w) showCandidates(w); });
[
  'the qiuck brown fox jumpd ovr teh lazzy dog beside the rivr',
  'i beleive this is a goverment of the peple by the peple',
  'speling corection is a clasic aplication of the noisy chanel',
  'i need a peace of paper to rite on',
].forEach((s) => $('#spchips').append(el('button', { class: 'chip', type: 'button', text: s.length > 34 ? s.slice(0, 32) + '…' : s, title: s, onclick: () => { $('#sent').value = s; runSentence(); } })));

(async () => {
  try {
    const r = await fetch('./data/freq.txt');
    if (!r.ok) throw new Error(`word list failed to load (${r.status})`);
    const txt = await r.text();
    for (const line of txt.split('\n')) { const sp = line.indexOf(' '); if (sp > 0) { const c = +line.slice(sp + 1); WORDS.set(line.slice(0, sp), c); TOTAL += c; } }
    $('#loadbar').style.width = '100%';
    $('#loadtext').textContent = `Ready · ${WORDS.size.toLocaleString('en-US')} words with counts from text8 (Wikipedia)`;
    $('#bCorrect').disabled = false; $('#bWord').disabled = false;
    runSentence(); showCandidates($('#word').value.trim());
    ready({ spell: { rank, correctSentence, editTable, size: WORDS.size } });
  } catch (e) { fail('Couldn’t load the spelling dictionary: ' + e.message); }
})();

if (REDUCED) $('#speed').value = 600;
