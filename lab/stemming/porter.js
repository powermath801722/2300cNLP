// Porter (1980) stemmer, original algorithm, with a per-step trace.
// Matches NLTK's PorterStemmer(mode=ORIGINAL_ALGORITHM); see data/porter-vectors.json.

const isCons = (w, i) => {
  const c = w[i];
  if ('aeiou'.includes(c)) return false;
  if (c === 'y') return i === 0 ? true : !isCons(w, i - 1);
  return true;
};
/** m in [C](VC)^m[V] */
export function measure(stem) {
  let m = 0, i = 0, n = stem.length;
  while (i < n && isCons(stem, i)) i++;
  while (i < n) {
    while (i < n && !isCons(stem, i)) i++;
    if (i >= n) break;
    while (i < n && isCons(stem, i)) i++;
    m++;
  }
  return m;
}
const hasVowel = (s) => [...s].some((_, i) => !isCons(s, i));
const endsDouble = (w) => w.length >= 2 && w[w.length - 1] === w[w.length - 2] && isCons(w, w.length - 1);
const cvc = (w) => { const n = w.length; return n >= 3 && isCons(w, n - 3) && !isCons(w, n - 2) && isCons(w, n - 1) && !'wxy'.includes(w[n - 1]); };

const M0 = { test: (s) => measure(s) > 0, text: 'm > 0' };
const M1 = { test: (s) => measure(s) > 1, text: 'm > 1' };
const V = { test: hasVowel, text: '*v* (stem has a vowel)' };

// Apply the first rule whose suffix matches; if its condition fails, stop (longest-match semantics).
function applyList(word, rules) {
  for (const [suf, rep, cond] of rules) {
    if (word.endsWith(suf)) {
      const stem = word.slice(0, word.length - suf.length);
      const ok = !cond || cond.test(stem);
      return { word: ok ? stem + rep : word, rule: `${suf || '∅'} → ${rep || '∅'}`, cond: cond ? cond.text : '', ok, stem };
    }
  }
  return { word, rule: null };
}

const STEP2 = [['ational', 'ate'], ['tional', 'tion'], ['enci', 'ence'], ['anci', 'ance'], ['izer', 'ize'], ['abli', 'able'], ['alli', 'al'], ['entli', 'ent'], ['eli', 'e'], ['ousli', 'ous'], ['ization', 'ize'], ['ation', 'ate'], ['ator', 'ate'], ['alism', 'al'], ['iveness', 'ive'], ['fulness', 'ful'], ['ousness', 'ous'], ['aliti', 'al'], ['iviti', 'ive'], ['biliti', 'ble']].map(([a, b]) => [a, b, M0]);
const STEP3 = [['icate', 'ic'], ['ative', ''], ['alize', 'al'], ['iciti', 'ic'], ['ical', 'ic'], ['ful', ''], ['ness', '']].map(([a, b]) => [a, b, M0]);
const ION = { test: (s) => measure(s) > 1 && /[st]$/.test(s), text: 'm > 1 and stem ends in s or t' };
const STEP4 = ['al', 'ance', 'ence', 'er', 'ic', 'able', 'ible', 'ant', 'ement', 'ment', 'ent', 'ion', 'ou', 'ism', 'ate', 'iti', 'ous', 'ive', 'ize'].map((s) => [s, '', s === 'ion' ? ION : M1]);

/** Returns { stem, trace: [{step, before, after, rule, cond, ok}] } */
export function porter(input) {
  let w = input.toLowerCase();
  const trace = [];
  const log = (step, before, r) => trace.push({ step, before, after: r.word, rule: r.rule, cond: r.cond || '', ok: r.ok !== false });

  // Step 1a
  let b = w, r = applyList(w, [['sses', 'ss'], ['ies', 'i'], ['ss', 'ss'], ['s', '']]);
  w = r.word; log('1a', b, r);

  // Step 1b
  b = w;
  if (w.endsWith('eed')) {
    const stem = w.slice(0, -3), ok = measure(stem) > 0;
    if (ok) w = stem + 'ee';
    log('1b', b, { word: w, rule: 'eed → ee', cond: 'm > 0', ok });
  } else {
    let inter = null, suf = null;
    for (const s of ['ed', 'ing']) if (w.endsWith(s)) { const st = w.slice(0, -s.length); if (hasVowel(st)) { inter = st; suf = s; } else log('1b', b, { word: w, rule: `${s} → ∅`, cond: V.text, ok: false }); break; }
    if (inter !== null) {
      log('1b', b, { word: inter, rule: `${suf} → ∅`, cond: V.text, ok: true });
      const b2 = inter;
      let r2 = applyList(inter, [['at', 'ate'], ['bl', 'ble'], ['iz', 'ize']]);
      if (!r2.rule) {
        if (endsDouble(inter)) {
          const last = inter[inter.length - 1], ok = !'lsz'.includes(last);
          r2 = { word: ok ? inter.slice(0, -1) : inter, rule: `*d → single ${last}`, cond: 'not *l, *s, *z', ok };
        } else {
          const ok = measure(inter) === 1 && cvc(inter);
          r2 = { word: ok ? inter + 'e' : inter, rule: '∅ → e', cond: 'm = 1 and *o (cvc)', ok };
        }
      }
      w = r2.word;
      log('1b′', b2, r2);
    } else if (!trace.some((t) => t.step === '1b')) log('1b', b, { word: w, rule: null });
  }

  // Step 1c
  b = w; r = applyList(w, [['y', 'i', V]]); w = r.word; log('1c', b, r);
  // Step 2–4
  b = w; r = applyList(w, STEP2); w = r.word; log('2', b, r);
  b = w; r = applyList(w, STEP3); w = r.word; log('3', b, r);
  b = w; r = applyList(w, STEP4); w = r.word; log('4', b, r);
  // Step 5a
  b = w;
  if (w.endsWith('e')) {
    const stem = w.slice(0, -1), m = measure(stem), ok = m > 1 || (m === 1 && !cvc(stem));
    if (ok) w = stem;
    log('5a', b, { word: w, rule: 'e → ∅', cond: 'm > 1, or m = 1 and not *o', ok });
  } else log('5a', b, { word: w, rule: null });
  // Step 5b
  b = w;
  if (w.endsWith('ll')) {
    const ok = measure(w) > 1; if (ok) w = w.slice(0, -1);
    log('5b', b, { word: w, rule: 'll → l', cond: 'm > 1', ok });
  } else log('5b', b, { word: w, rule: null });
  return { stem: w, trace };
}

export const stem = (w) => porter(w).stem;
