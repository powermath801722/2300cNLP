/**
 * Braidwater Text Studio · tokenize.js — rule-based word tokenizer and sentence splitter.
 * Ported verbatim from /lab/tokenize/app.js (same rules, same output). Side-effect free, no data to load.
 *
 * Exports:
 *   wordTokenize(text) → [{ text, type, start, end }]   type: word|clitic|num|url|email|mention|hashtag|emoji|punct|sym
 *   tokenize(text)     → alias of wordTokenize
 *   sentences(text, toks?) → string[]                   abbreviation-aware segmentation
 *   naiveSentences(text) → string[]                      split on . ! ? + whitespace
 *   whitespaceTokenize(text) → string[]
 *   words(text) → string[]                               word tokens only (lowercased), clitics excluded
 */
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

export const tokenize = wordTokenize;

/** Lowercased word tokens (letters/numbers words only, no clitics or punctuation). */
export const words = (text) => wordTokenize(text).filter((t) => t.type === 'word').map((t) => t.text.toLowerCase());
