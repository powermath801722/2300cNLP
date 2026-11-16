import { el, $, ready, fail, getJSON } from '/lab/shared/lab.js';

// ---------------------------------------------------------------------------
// Dependency parser: mirrors training/parsing_train.py exactly (features, hash, transitions).
// ---------------------------------------------------------------------------
let M; // model.json
const enc = new TextEncoder();
const memo = new Map();
function fh(s) {
  let h = memo.get(s);
  if (h === undefined) {
    let x = 0x811c9dc5;
    for (const b of enc.encode(s)) { x ^= b; x = Math.imul(x, 0x01000193) >>> 0; }
    h = x % M.nb; memo.set(s, h);
  }
  return h;
}
const isDigitStr = (s) => s.length > 0 && /^\p{Nd}+$/u.test(s);
function norm(w) {
  if (/\p{Nd}/u.test(w)) return isDigitStr(w.replace(/,/g, '').replace(/\./g, '')) ? '!DIGIT' : '!HASDIGIT';
  return w.toLowerCase();
}
function shape(w) {
  const out = [];
  for (const ch of [...w].slice(0, 6)) {
    const c = /\p{Lu}/u.test(ch) ? 'X' : /\p{Ll}/u.test(ch) ? 'x' : /\p{Nd}/u.test(ch) ? 'd' : ch;
    if (!out.length || out[out.length - 1] !== c) out.push(c);
  }
  return out.join('');
}
const py = (v) => (v === true ? 'True' : v === false ? 'False' : String(v));

function scorer(part, ncls) {
  const rows = M[part].rows, scale = M[part].scale;
  return (ids) => {
    const s = new Float64Array(ncls);
    for (const id of ids) { const r = rows[id]; if (r) for (let k = 0; k < r.length; k += 2) s[r[k]] += r[k + 1] * scale; }
    return s;
  };
}
let tagScore, parseScore;
const argmax = (a) => { let b = 0; for (let i = 1; i < a.length; i++) if (a[i] > a[b]) b = i; return b; };

function tagFeats(words, i, p1, p2) {
  const w = norm(words[i]);
  const ctx = ['<s2>', '<s1>', ...words.map(norm), '</s1>', '</s2>'], j = i + 2;
  const f = ['b', 'w=' + w, 's3=' + w.slice(-3), 's2=' + w.slice(-2), 'p1=' + w.slice(0, 1), 'sh=' + shape(words[i]),
    't1=' + p1, 't2=' + p2, 't12=' + p1 + '|' + p2, 't1w=' + p1 + '|' + w,
    'w-1=' + ctx[j - 1], 's3-1=' + ctx[j - 1].slice(-3), 'w-2=' + ctx[j - 2],
    'w+1=' + ctx[j + 1], 's3+1=' + ctx[j + 1].slice(-3), 'w+2=' + ctx[j + 2]];
  return f.map(fh);
}
function tagSentence(words) {
  const out = []; let p1 = '<s1>', p2 = '<s2>';
  for (let i = 0; i < words.length; i++) { const t = M.tags[argmax(tagScore(tagFeats(words, i, p1, p2)))]; out.push(t); p2 = p1; p1 = t; }
  return out;
}

function parseFeats(words, tags, stack, b, heads, rels, lefts, rights) {
  const n = words.length;
  const W = (i) => (i === 0 ? '<root>' : i > 0 && i <= n ? norm(words[i - 1]) : '<none>');
  const T = (i) => (i === 0 ? '<root>' : i > 0 && i <= n ? tags[i - 1] : '<none>');
  const L = (i) => (i != null && i > 0 && rels[i] ? rels[i] : '<none>');
  const at = (k) => (stack.length >= k ? stack[stack.length - k] : null);
  const s0 = at(1), s1 = at(2), s2 = at(3);
  const b0 = b <= n ? b : null, b1 = b + 1 <= n ? b + 1 : null, b2 = b + 2 <= n ? b + 2 : null;
  const ww = (i) => (i != null ? W(i) : '<none>'), tt = (i) => (i != null ? T(i) : '<none>');
  const lc = (i) => (i != null && lefts[i].length ? lefts[i][0] : null);
  const rc = (i) => (i != null && rights[i].length ? rights[i][rights[i].length - 1] : null);
  const s0w = ww(s0), s0t = tt(s0), s1w = ww(s1), s1t = tt(s1), s2t = tt(s2);
  const b0w = ww(b0), b0t = tt(b0), b1w = ww(b1), b1t = tt(b1), b2t = tt(b2);
  const lc0 = lc(s0), rc0 = rc(s0), lc1 = lc(s1), rc1 = rc(s1);
  const dist = s0 != null && s1 != null ? Math.min(s0 - s1, 5) : -1;
  const nl0 = s0 != null ? lefts[s0].length : -1, nr0 = s0 != null ? rights[s0].length : -1;
  const nl1 = s1 != null ? lefts[s1].length : -1, nr1 = s1 != null ? rights[s1].length : -1;
  const f = ['b',
    's0w=' + s0w, 's0t=' + s0t, 's0wt=' + s0w + '|' + s0t,
    's1w=' + s1w, 's1t=' + s1t, 's1wt=' + s1w + '|' + s1t, 's2t=' + s2t,
    'b0w=' + b0w, 'b0t=' + b0t, 'b0wt=' + b0w + '|' + b0t, 'b1w=' + b1w, 'b1t=' + b1t, 'b2t=' + b2t,
    's0t s1t=' + s0t + '|' + s1t, 's0w s1w=' + s0w + '|' + s1w, 's0w s1t=' + s0w + '|' + s1t, 's0t s1w=' + s0t + '|' + s1w,
    's0t b0t=' + s0t + '|' + b0t, 's1t s0t b0t=' + s1t + '|' + s0t + '|' + b0t, 's2t s1t s0t=' + s2t + '|' + s1t + '|' + s0t,
    's0t b0t b1t=' + s0t + '|' + b0t + '|' + b1t,
    'lc0t=' + tt(lc0), 'rc0t=' + tt(rc0), 'lc1t=' + tt(lc1), 'rc1t=' + tt(rc1),
    'lc0l=' + L(lc0), 'rc0l=' + L(rc0), 'lc1l=' + L(lc1), 'rc1l=' + L(rc1),
    's0t lc0l=' + s0t + '|' + L(lc0), 's1t rc1l=' + s1t + '|' + L(rc1), 's1t lc1l=' + s1t + '|' + L(lc1),
    'd s0t s1t=' + dist + '|' + s0t + '|' + s1t, 'd s0w=' + dist + '|' + s0w, 'd s1w=' + dist + '|' + s1w,
    'v s0=' + nl0 + '|' + nr0 + '|' + s0t, 'v s1=' + nl1 + '|' + nr1 + '|' + s1t,
    'bempty=' + py(b > n) + '|' + stack.length];
  return f.map(fh);
}

function parse(words, tags) {
  const n = words.length, R = M.rels.length;
  const stack = [0]; let b = 1;
  const heads = new Array(n + 1).fill(null), rels = new Array(n + 1).fill(null);
  const lefts = Array.from({ length: n + 1 }, () => []), rights = Array.from({ length: n + 1 }, () => []);
  const steps = [];
  while (!(b > n && stack.length === 1)) {
    const sc = parseScore(parseFeats(words, tags, stack, b, heads, rels, lefts, rights));
    const ok = new Array(M.actions.length).fill(false);
    if (b <= n) ok[0] = true;
    if (stack.length >= 2) {
      const s1 = stack[stack.length - 2];
      if (s1 !== 0) for (let k = 1; k <= R; k++) ok[k] = true;
      if (s1 !== 0 || b > n) for (let k = R + 1; k <= 2 * R; k++) ok[k] = true;
    }
    let best = -1; for (let k = 0; k < sc.length; k++) if (ok[k] && (best < 0 || sc[k] > sc[best])) best = k;
    steps.push({ stack: [...stack], b, action: M.actions[best], arcs: heads.map((h, i) => (h != null ? [h, i, rels[i]] : null)).filter(Boolean) });
    if (best === 0) { stack.push(b); b++; continue; }
    const rel = M.actions[best].slice(2), s0 = stack[stack.length - 1], s1 = stack[stack.length - 2];
    if (M.actions[best][0] === 'L') { heads[s1] = s0; rels[s1] = rel; lefts[s0].push(s1); lefts[s0].sort((a, c) => a - c); stack.splice(stack.length - 2, 1); }
    else { heads[s0] = s1; rels[s0] = rel; rights[s1].push(s0); rights[s1].sort((a, c) => a - c); stack.pop(); }
  }
  steps.push({ stack: [...stack], b, action: null, arcs: heads.map((h, i) => (h != null ? [h, i, rels[i]] : null)).filter(Boolean) });
  return { heads, rels, steps };
}

// EWT-style tokenization: clitics split off (did n't, it 's, ca n't, wo n't), punctuation separate.
function tokenize(s) {
  const out = [];
  for (const m of s.match(/[A-Za-z]+n't|'(?:s|re|ve|ll|d|m)\b|[0-9]+(?:[.,:/-][0-9]+)*|[\p{L}\p{M}]+(?:-[\p{L}\p{M}]+)*|[^\s\p{L}\p{N}]|\S+/giu) || []) {
    const neg = /^([A-Za-z]+)(n't)$/i.exec(m);
    if (neg) out.push(neg[1], neg[2]); // "can't" -> "ca" + "n't", as in EWT
    else out.push(m);
  }
  return out;
}

const EXAMPLES = [
  'The students who read the paper asked a question about its results.',
  'I saw the man with the telescope.',
  "She didn't think the plan would work, but it did.",
  'Give the dog a bone and send me the bill.',
  'The old man the boat.',
  'Researchers in Boston have developed a cheaper test for the virus.',
];

let cur = null, stepIdx = 0;
const EXPLAIN = { SHIFT: 'move the next buffer word onto the stack', L: 's0 becomes the head of s1; pop s1', R: 's1 becomes the head of s0; pop s0' };

function svgEl(tag, attrs, text) {
  const n = document.createElementNS('http://www.w3.org/2000/svg', tag);
  for (const [k, v] of Object.entries(attrs)) n.setAttribute(k, v);
  if (text != null) n.textContent = text;
  return n;
}

function drawArcs(host, words, tags, arcs, newest) {
  const ctx = drawArcs.ctx || (drawArcs.ctx = document.createElement('canvas').getContext('2d'));
  ctx.font = '500 15px Barlow, system-ui, sans-serif';
  const gap = 22, xs = [], ws = [];
  let x = 16;
  const toks = ['ROOT', ...words];
  for (let i = 0; i < toks.length; i++) { const w = Math.max(ctx.measureText(toks[i]).width, 34); ws.push(w); xs.push(x + w / 2); x += w + gap; }
  const width = Math.max(x, 300);
  let maxSpan = 1; for (const [h, d] of arcs) maxSpan = Math.max(maxSpan, Math.abs(h - d));
  const unit = 26, top = 18 + Math.min(maxSpan, 12) * unit, baseY = top + 22, height = baseY + 30;
  const svg = svgEl('svg', { width, height, viewBox: `0 0 ${width} ${height}` });
  const defs = svgEl('defs', {});
  for (const id of ['ah', 'ahn']) {
    const mk = svgEl('marker', { id, viewBox: '0 0 10 10', refX: 9, refY: 5, markerWidth: 7, markerHeight: 7, orient: 'auto-start-reverse' });
    mk.append(svgEl('path', { d: 'M0 0 L10 5 L0 10 z', fill: id === 'ahn' ? '#d8b66a' : 'var(--accent)' }));
    defs.append(mk);
  }
  svg.append(defs);
  toks.forEach((t, i) => {
    svg.append(svgEl('text', { x: xs[i], y: baseY, 'text-anchor': 'middle', class: 'word' }, t));
    svg.append(svgEl('text', { x: xs[i], y: baseY + 18, 'text-anchor': 'middle', class: 'tag' }, i === 0 ? '' : tags[i - 1]));
  });
  const sorted = [...arcs].sort((a, c) => Math.abs(a[0] - a[1]) - Math.abs(c[0] - c[1]));
  for (const [h, d, r] of sorted) {
    const span = Math.min(Math.abs(h - d), 12), y = baseY - 18 - span * unit;
    const x1 = xs[h] + (d > h ? 6 : -6), x2 = xs[d];
    const isNew = newest && newest[0] === h && newest[1] === d;
    const p = svgEl('path', { d: `M${x1} ${baseY - 16} C ${x1} ${y}, ${x2} ${y}, ${x2} ${baseY - 18}`, class: 'arc' + (isNew ? ' new' : '') + (h === 0 ? ' root' : ''), 'marker-end': `url(#${isNew ? 'ahn' : 'ah'})` });
    svg.append(p);
    // a cubic with both control points at y peaks 3/4 of the way from the endpoints to y
    const lx = (x1 + x2) / 2, apex = (baseY - 17) * 0.25 + y * 0.75;
    const t = svgEl('text', { x: lx, y: Math.max(apex - 4, 12), 'text-anchor': 'middle', class: 'lab' + (isNew ? ' new' : '') }, r);
    svg.append(t);
  }
  host.replaceChildren(svg);
}

function renderStep() {
  if (!cur) return;
  const s = cur.steps[stepIdx], n = cur.words.length;
  $('#step').value = stepIdx;
  $('#stepno').textContent = `Step ${stepIdx} of ${cur.steps.length - 1}`;
  const name = (i) => (i === 0 ? 'ROOT' : cur.words[i - 1]);
  $('#stack').replaceChildren(...s.stack.map((i, k) => el('span', { class: 'tok' + (k === s.stack.length - 1 ? ' top' : ''), text: name(i) })));
  const buf = []; for (let i = s.b; i <= n; i++) buf.push(el('span', { class: 'tok' + (i === s.b ? ' b0' : ''), text: name(i) }));
  $('#buffer').replaceChildren(...(buf.length ? buf : [el('span', { class: 'note', text: '(empty)' })]));
  $('#action').textContent = s.action || 'done';
  $('#actexp').textContent = s.action ? '— ' + EXPLAIN[s.action === 'SHIFT' ? 'SHIFT' : s.action[0]] : '— every word has a head';
  const prev = stepIdx > 0 ? cur.steps[stepIdx - 1] : null;
  const newest = prev ? s.arcs.find((a) => !prev.arcs.some((p) => p[1] === a[1])) : null;
  drawArcs($('#arcs'), cur.words, cur.tags, s.arcs, newest);
}

function runParse() {
  const text = $('#sent').value.trim();
  if (!text) return;
  const words = tokenize(text).slice(0, 60);
  const tags = tagSentence(words);
  const { heads, rels, steps } = parse(words, tags);
  cur = { words, tags, heads, rels, steps };
  stepIdx = steps.length - 1;
  $('#step').max = steps.length - 1;
  renderStep();
  const tb = $('#deptable tbody');
  tb.replaceChildren(...words.map((w, i) => el('tr', {}, [
    el('td', { text: String(i + 1) }), el('td', { text: w }), el('td', { text: tags[i] }),
    el('td', { text: heads[i + 1] === 0 ? 'ROOT' : `${words[heads[i + 1] - 1]} (${heads[i + 1]})` }), el('td', { text: rels[i + 1] }),
  ])));
}

function setStep(i) { if (!cur) return; stepIdx = Math.max(0, Math.min(cur.steps.length - 1, i)); renderStep(); }

function renderEval() {
  const e = M.eval, pct = (v) => (v * 100).toFixed(1) + '%';
  const items = [[pct(e.test.uas), 'UAS, test (predicted tags)'], [pct(e.test.las), 'LAS, test (predicted tags)'], [pct(e.test_gold_tags.las), 'LAS, test with gold tags'], [pct(e.test.tag_acc), 'UPOS tagging accuracy, test'], [pct(e.dev.las), 'LAS, dev (predicted tags)']];
  $('#evalgrid').replaceChildren(...items.map(([b, s]) => el('div', {}, [el('b', { text: b }), el('span', { text: s })])));
  $('#evalnote').textContent = `Scored over all ${e.test.tokens.toLocaleString('en-US')} test tokens of ${e.test.sentences.toLocaleString('en-US')} sentences, punctuation included (CoNLL 2018 convention). Without punctuation: UAS ${pct(e.test.uas_nopunct)}, LAS ${pct(e.test.las_nopunct)}. Trained on ${M.train_sentences.toLocaleString('en-US')} sentences of ${M.source}; weights pruned and quantized to 8 bits, and these scores are for the quantized weights.`;
}

// ---------------------------------------------------------------------------
// PCFG + CKY
// ---------------------------------------------------------------------------
const GRAMMAR = `S -> NP VP 1.0
NP -> Det N 0.5
NP -> NP PP 0.2
NP -> 'I' 0.15
NP -> 'she' 0.15
VP -> V NP 0.6
VP -> VP PP 0.4
PP -> P NP 1.0
Det -> 'the' 0.6
Det -> 'a' 0.4
N -> 'man' 0.3
N -> 'telescope' 0.3
N -> 'dog' 0.2
N -> 'park' 0.2
V -> 'saw' 0.6
V -> 'walked' 0.4
P -> 'with' 0.5
P -> 'in' 0.5`;
const CEX = ['I saw the man with the telescope', 'she walked the dog in the park', 'I saw the man in the park with a telescope', 'she saw a dog with a man in the park with the telescope'];

function parseGrammar(text) {
  const bin = [], lex = new Map(), totals = new Map(), errs = [];
  text.split('\n').forEach((line, li) => {
    const s = line.trim(); if (!s || s.startsWith('#')) return;
    let m = /^(\S+)\s*->\s*(\S+)\s+(\S+)\s+([0-9]*\.?[0-9]+(?:e-?\d+)?)$/.exec(s);
    if (m && !m[2].startsWith("'")) { const p = +m[4]; bin.push({ A: m[1], B: m[2], C: m[3], p }); totals.set(m[1], (totals.get(m[1]) || 0) + p); return; }
    m = /^(\S+)\s*->\s*'([^']+)'\s+([0-9]*\.?[0-9]+(?:e-?\d+)?)$/.exec(s);
    if (m) { const p = +m[3], w = m[2].toLowerCase(); if (!lex.has(w)) lex.set(w, []); lex.get(w).push({ A: m[1], p }); totals.set(m[1], (totals.get(m[1]) || 0) + p); return; }
    errs.push(`Line ${li + 1}: “${s}” isn’t a binary rule (A -> B C p) or a lexical rule (A -> 'word' p).`);
  });
  for (const [A, t] of totals) if (Math.abs(t - 1) > 1e-6) errs.push(`Rules for ${A} sum to ${+t.toFixed(4)}, not 1 (allowed, but the model is no longer a proper PCFG).`);
  return { bin, lex, errs };
}

function cky(words, G) {
  const n = words.length;
  const chart = Array.from({ length: n + 1 }, () => Array.from({ length: n + 1 }, () => new Map()));
  for (let i = 0; i < n; i++) {
    for (const { A, p } of G.lex.get(words[i]) || []) {
      const c = chart[i][i + 1].get(A) || { best: 0, inside: 0, bps: [] };
      c.best = Math.max(c.best, p); c.inside += p; c.bps.push({ p, word: words[i] }); chart[i][i + 1].set(A, c);
    }
  }
  for (let len = 2; len <= n; len++) for (let i = 0; i + len <= n; i++) {
    const j = i + len;
    for (let k = i + 1; k < j; k++) for (const r of G.bin) {
      const L = chart[i][k].get(r.B), R = chart[k][j].get(r.C);
      if (!L || !R) continue;
      const c = chart[i][j].get(r.A) || { best: 0, inside: 0, bps: [] };
      c.best = Math.max(c.best, r.p * L.best * R.best); c.inside += r.p * L.inside * R.inside;
      c.bps.push({ p: r.p, k, B: r.B, C: r.C }); chart[i][j].set(r.A, c);
    }
  }
  return chart;
}

function enumerate(chart, A, i, j, cap) {
  const c = chart[i][j].get(A); if (!c) return [];
  const out = [];
  for (const bp of c.bps) {
    if (bp.word) { out.push({ p: bp.p, t: [A, bp.word] }); continue; }
    for (const l of enumerate(chart, bp.B, i, bp.k, cap)) for (const r of enumerate(chart, bp.C, bp.k, j, cap)) {
      out.push({ p: bp.p * l.p * r.p, t: [A, l.t, r.t] }); if (out.length >= cap) return out;
    }
  }
  return out;
}
const bracket = (t) => (typeof t[1] === 'string' ? `(${t[0]} ${t[1]})` : `(${t[0]} ${t.slice(1).map(bracket).join(' ')})`);

function drawTree(host, tree) {
  const ctx = drawArcs.ctx || (drawArcs.ctx = document.createElement('canvas').getContext('2d'));
  ctx.font = '500 14px Barlow, system-ui, sans-serif';
  const leaves = []; const walk = (t, d) => { if (typeof t[1] === 'string') { leaves.push(t); t.depth = d; return; } t.depth = d; t.slice(1).forEach((c) => walk(c, d + 1)); };
  walk(tree, 0);
  let x = 20, maxD = 0;
  for (const lf of leaves) { const w = Math.max(ctx.measureText(lf[1]).width, 40); lf.x = x + w / 2; x += w + 18; maxD = Math.max(maxD, lf.depth); }
  const pos = (t) => { if (typeof t[1] === 'string') return t.x; const xs = t.slice(1).map(pos); t.x = (xs[0] + xs[xs.length - 1]) / 2; return t.x; };
  pos(tree);
  const lvl = 46, width = Math.max(x, 300), height = (maxD + 2) * lvl + 20;
  const svg = svgEl('svg', { width, height, viewBox: `0 0 ${width} ${height}` });
  const draw = (t) => {
    const y = 24 + t.depth * lvl;
    if (typeof t[1] === 'string') {
      const ly = 24 + (maxD + 1) * lvl;
      svg.append(svgEl('line', { x1: t.x, y1: y + 6, x2: t.x, y2: ly - 14, class: 'edge' }));
      svg.append(svgEl('text', { x: t.x, y, 'text-anchor': 'middle', class: 'nt' }, t[0]));
      svg.append(svgEl('text', { x: t.x, y: ly, 'text-anchor': 'middle', class: 'lf' }, t[1]));
      return;
    }
    for (const c of t.slice(1)) svg.append(svgEl('line', { x1: t.x, y1: y + 6, x2: c.x, y2: 24 + c.depth * lvl - 14, class: 'edge' }));
    svg.append(svgEl('text', { x: t.x, y, 'text-anchor': 'middle', class: 'nt' }, t[0]));
    t.slice(1).forEach(draw);
  };
  draw(tree);
  host.replaceChildren(svg);
}

const fmtP = (p) => (p === 0 ? '0' : p >= 1e-3 ? p.toPrecision(3) : p.toExponential(2));

function runCKY() {
  const G = parseGrammar($('#grammar').value);
  const gerr = $('#gerr');
  gerr.hidden = !G.errs.length; gerr.textContent = G.errs.join(' ');
  const words = $('#csent').value.trim().toLowerCase().split(/\s+/).filter(Boolean);
  const cerr = $('#cerr');
  const unknown = words.filter((w) => !G.lex.has(w));
  if (!words.length || unknown.length) {
    cerr.hidden = false; cerr.textContent = unknown.length ? `Not in the grammar’s lexicon: ${unknown.join(', ')}. Add a rule such as N -> '${unknown[0]}' 0.1 (and rebalance that nonterminal’s probabilities).` : 'Type a sentence.';
    $('#parses').replaceChildren(); $('#tree').replaceChildren(); $('#chart').replaceChildren(); $('#csum').textContent = '';
    window.__lab.cky = { parses: [], words };
    return;
  }
  cerr.hidden = true;
  const chart = cky(words, G), n = words.length;
  const top = chart[0][n].get('S');
  const parses = top ? enumerate(chart, 'S', 0, n, 200).sort((a, b) => b.p - a.p) : [];
  const total = top ? top.inside : 0;
  $('#csum').textContent = top
    ? `${parses.length}${parses.length >= 200 ? '+' : ''} parse${parses.length === 1 ? '' : 's'} · P(sentence) = Σ over trees = ${fmtP(total)} · best tree ${fmtP(parses[0].p)} (${(parses[0].p / total * 100).toFixed(1)}% of the probability mass)`
    : 'No parse: the grammar can’t derive S over this sentence.';
  let sel = 0;
  const btns = parses.slice(0, 14).map((pt, i) => el('button', { type: 'button', 'aria-pressed': String(i === 0), onclick: () => { sel = i; btns.forEach((b, k) => b.setAttribute('aria-pressed', String(k === sel))); drawTree($('#tree'), parses[i].t); } }, [
    el('span', { class: 'br', text: bracket(pt.t) }), el('span', { class: 'pr', text: `${fmtP(pt.p)} · ${(pt.p / total * 100).toFixed(1)}%` }),
  ]));
  $('#parses').replaceChildren(...btns, ...(parses.length > 14 ? [el('p', { class: 'note', text: `…and ${parses.length - 14}${parses.length >= 200 ? '+' : ''} more.` })] : []));
  if (parses.length) drawTree($('#tree'), parses[0].t); else $('#tree').replaceChildren();
  // chart table: rows = start i, cols = end j
  const tbl = el('table', { class: 'chart' });
  tbl.append(el('thead', {}, el('tr', {}, [el('th', { text: 'i \\ j' }), ...words.map((w, j) => el('th', { text: `${j + 1}: ${w}` }))])));
  const tb = el('tbody');
  for (let i = 0; i < n; i++) {
    tb.append(el('tr', {}, [el('th', { text: `${i}` }), ...words.map((_, jj) => {
      const j = jj + 1;
      if (j <= i) return el('td', { class: 'off' });
      const cell = [...chart[i][j].entries()].sort((a, b) => b[1].best - a[1].best);
      const bestA = cell.length ? cell[0][0] : null;
      return el('td', {}, cell.length ? cell.map(([A, c]) => el('div', { class: A === bestA ? 'best' : '', text: `${A} ${fmtP(c.best)}` })) : [el('span', { class: 'note', text: '·' })]);
    })]));
  }
  tbl.append(tb);
  $('#chart').replaceChildren(tbl);
  window.__lab.cky = { words, parses: parses.map((p) => ({ p: p.p, tree: bracket(p.t) })), total };
}

// ---------------------------------------------------------------------------
async function init() {
  $('#grammar').value = GRAMMAR;
  $('#cchips').append(...CEX.map((s) => el('button', { class: 'chip', type: 'button', text: s, onclick: () => { $('#csent').value = s; runCKY(); } })));
  $('#ckyform').addEventListener('submit', (e) => { e.preventDefault(); runCKY(); });
  let gt; $('#grammar').addEventListener('input', () => { clearTimeout(gt); gt = setTimeout(runCKY, 300); });
  window.__lab = Object.assign(window.__lab || {}, {});
  runCKY();

  $('#status').textContent = 'Loading the parser (3.0 MB)…';
  try { M = await getJSON('/lab/parsing/data/model.json'); } catch (e) { fail(e.message); return; }
  tagScore = scorer('tagger', M.tags.length); parseScore = scorer('parser', M.actions.length);
  $('#status').textContent = `Ready · ${M.actions.length} actions over ${M.rels.length} relations · ${M.tags.length} UPOS tags`;
  renderEval();
  $('#chips').append(...EXAMPLES.map((s) => el('button', { class: 'chip', type: 'button', text: s, onclick: () => { $('#sent').value = s; runParse(); } })));
  $('#depform').addEventListener('submit', (e) => { e.preventDefault(); runParse(); });
  $('#parse').disabled = false;
  $('#step').addEventListener('input', (e) => setStep(+e.target.value));
  $('#first').onclick = () => setStep(0); $('#last').onclick = () => setStep(Infinity);
  $('#prev').onclick = () => setStep(stepIdx - 1); $('#next').onclick = () => setStep(stepIdx + 1);
  runParse();
  window.__lab.parse = (s) => { const words = Array.isArray(s) ? s : tokenize(s), tags = tagSentence(words), r = parse(words, tags); return { words, tags, heads: r.heads.slice(1), rels: r.rels.slice(1), steps: r.steps.length - 1 }; };
  window.__lab.tokenize = tokenize;
  ready({ parser: true });
}
init();
