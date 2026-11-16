/**
 * Braidwater Text Studio · spell.js — Levenshtein/Damerau edit distance and noisy-channel spelling correction
 * (prior P(w) from text8 word counts × keyboard-aware channel P(x|w)), ported verbatim from /lab/spell/app.js.
 * Side-effect free; /lab/spell/data/freq.txt (60,000 words) loads on the first loadSpell() call.
 *
 * Exports:
 *   loadSpell(url='/lab/spell/data/freq.txt') → Promise<{ correctWord, rank, correctSentence, checkText, has, count, size }>
 *   correctWord(w, model='noisy'|'norvig') → ranked candidates [{ w, d, count, prior, chan, ops, score, post }] (needs loadSpell first)
 *   rank(x, model) — same as correctWord;  correctSentence(text, model) → [{ orig, fixed?, nos? }]
 *   checkText(text, { max=80 }) → [{ word, start, end, suggestions: [{ w, post }] }]
 *       flags words that are not in the dictionary (capitalized words mid-sentence are treated as names and skipped)
 *   editTable(a, b, costs), backtrace(a, b, D, costs)
 */
const WORDS = new Map(); let TOTAL = 0;

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

const LETTERS = 'abcdefghijklmnopqrstuvwxyz';

// QWERTY geometry for the keyboard-aware channel model
const KEYPOS = {};
['qwertyuiop', 'asdfghjkl', 'zxcvbnm'].forEach((row, r) => [...row].forEach((ch, c) => { KEYPOS[ch] = [c + [0, 0.25, 0.75][r], r]; }));
function adjacent(x, y) { const p = KEYPOS[x], q = KEYPOS[y]; if (!p || !q) return false; return Math.hypot(p[0] - q[0], p[1] - q[1]) < 1.3; }

// Per-edit probabilities of the channel model (illustrative, not learned from data)
const P_EDIT = { subNear: 0.01, subFar: 0.002, del: 0.004, delDouble: 0.01, ins: 0.004, insDouble: 0.01, trans: 0.008 };
const P_CORRECT = 0.95; // probability a word is typed with no error
const RARE = 100;       // dictionary words rarer than this are re-checked by the noisy channel

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
export function rank(x, model = 'noisy') {
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

/** Correct a whole sentence; returns [{orig, fixed|null, nos}] tokens. */
export function correctSentence(text, model = 'noisy') {
  const out = [];
  for (const tok of text.match(/[A-Za-z]+|[^A-Za-z]+/g) || []) {
    if (!/^[A-Za-z]+$/.test(tok)) { out.push({ orig: tok }); continue; }
    const lw = tok.toLowerCase();
    if (lw.length < 2) { out.push({ orig: tok }); continue; }
    // Norvig keeps every dictionary word. The noisy channel also lets rare dictionary words (e.g. "teh", 25
    // occurrences in text8) compete with their neighbors, since P(x|w)·P(w) can favor a much more common word.
    if (WORDS.has(lw) && (model === 'norvig' || WORDS.get(lw) >= RARE)) { out.push({ orig: tok }); continue; }
    const best = rank(lw, model)[0];
    if (best && best.w === lw) { out.push({ orig: tok }); continue; }
    if (!best) { out.push({ orig: tok, nos: true }); continue; }
    let fixed = best.w;
    if (tok[0] === tok[0].toUpperCase()) fixed = fixed[0].toUpperCase() + fixed.slice(1);
    out.push({ orig: tok, fixed });
  }
  return out;
}

export const correctWord = rank;

/** Flag out-of-dictionary words in running text with ranked suggestions. */
export function checkText(text, { max = 80 } = {}) {
  const out = [], seen = new Map();
  const re = /[A-Za-z]+(?:['’][A-Za-z]+)*/g; let m;
  while ((m = re.exec(text)) && out.length < max) {
    const tok = m[0], base = tok.replace(/['’].*$/, ''), lw = base.toLowerCase();
    if (lw.length < 3 || WORDS.has(lw)) continue;
    const before = text.slice(0, m.index).replace(/\s+$/, '');
    const sentStart = !before || /[.!?:"“(]$/.test(before);
    if (/^[A-Z]/.test(base) && !sentStart) continue; // probably a name
    if (/^[A-Z]{2,}$/.test(base)) continue;            // acronym
    if (!seen.has(lw)) {
      const c = rank(lw).slice(0, 4).filter((r) => r.w !== lw);
      seen.set(lw, c.map((r) => ({ w: r.w, post: r.post, d: r.d })));
    }
    const sug = seen.get(lw);
    const cap = (w) => (/^[A-Z]/.test(base) ? w[0].toUpperCase() + w.slice(1) : w);
    out.push({ word: base, start: m.index, end: m.index + base.length, suggestions: sug.map((s) => ({ ...s, w: cap(s.w) })) });
  }
  return out;
}

let _p = null;
export function loadSpell(url = '/lab/spell/data/freq.txt') {
  return (_p ||= fetch(url).then((r) => { if (!r.ok) throw new Error(`Couldn’t load ${url} (${r.status})`); return r.text(); }).then((txt) => {
    if (!WORDS.size) for (const line of txt.split('\n')) { const sp = line.indexOf(' '); if (sp > 0) { const c = +line.slice(sp + 1); WORDS.set(line.slice(0, sp), c); TOTAL += c; } }
    return { correctWord: rank, rank, correctSentence, checkText, has: (w) => WORDS.has(w.toLowerCase()), count: (w) => WORDS.get(w.toLowerCase()) || 0, size: WORDS.size };
  }).catch((e) => { _p = null; throw e; }));
}
