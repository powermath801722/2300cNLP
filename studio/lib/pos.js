/**
 * Braidwater Text Studio · pos.js — bigram HMM part-of-speech tagger (Penn Treebank tags) decoded with Viterbi,
 * with suffix/capitalization classes for unknown words. Ported verbatim from /lab/pos/app.js
 * (trained on the NLTK Treebank sample by training/pos_train.py; 95.2% held-out accuracy).
 * Side-effect free; /lab/pos/data/hmm.json loads on the first loadPOS() call.
 *
 * Exports:
 *   loadPOS(url='/lab/pos/data/hmm.json') → Promise<{ tag(tokens) → string[], tagSentence(s) → {words, tags}, tokenize(s), tags, metrics }>
 *   PENN — tag → description;  coarse(tag) → 'noun'|'verb'|'adj'|'adv'|'pron'|'det'|'adp'|'conj'|'num'|'punct'|'other'
 */
export const PENN = {
  CC: 'Coordinating conjunction', CD: 'Cardinal number', DT: 'Determiner', EX: 'Existential there', FW: 'Foreign word',
  IN: 'Preposition / subordinating conj.', JJ: 'Adjective', JJR: 'Adjective, comparative', JJS: 'Adjective, superlative',
  LS: 'List item marker', MD: 'Modal', NN: 'Noun, singular or mass', NNS: 'Noun, plural', NNP: 'Proper noun, singular',
  NNPS: 'Proper noun, plural', PDT: 'Predeterminer', POS: 'Possessive ending', PRP: 'Personal pronoun', 'PRP$': 'Possessive pronoun',
  RB: 'Adverb', RBR: 'Adverb, comparative', RBS: 'Adverb, superlative', RP: 'Particle', SYM: 'Symbol', TO: 'to',
  UH: 'Interjection', VB: 'Verb, base form', VBD: 'Verb, past tense', VBG: 'Verb, gerund / present participle',
  VBN: 'Verb, past participle', VBP: 'Verb, non-3rd person singular present', VBZ: 'Verb, 3rd person singular present',
  WDT: 'Wh-determiner', WP: 'Wh-pronoun', 'WP$': 'Possessive wh-pronoun', WRB: 'Wh-adverb',
  '.': 'Sentence-final punctuation', ',': 'Comma', ':': 'Colon, semicolon, dash', '``': 'Opening quote', "''": 'Closing quote',
  '-LRB-': 'Left bracket', '-RRB-': 'Right bracket', '$': 'Dollar sign', '#': 'Pound sign',
};
let H, T, tags;

// Must mirror unk_class() in training/pos_train.py
function unkClass(w, first) {
  if (/^[0-9.,:/-]+$/.test(w) && /[0-9]/.test(w)) return 'UNK-num';
  if (w.includes('-')) return 'UNK-hyph';
  const lw = w.toLowerCase(), cap = /^[A-Z]/.test(w) && !first;
  for (const suf of H.unk_suffixes) if (lw.endsWith(suf) && lw.length > suf.length + 2) return (cap ? 'UNK-cap-' : 'UNK-') + suf;
  return cap ? 'UNK-cap' : 'UNK';
}
function key(w, i) {
  if (H.emit[w]) return { k: w, unk: false };
  if (i === 0 && H.emit[w.toLowerCase()]) return { k: w.toLowerCase(), unk: false };
  let k = unkClass(w, i === 0);
  if (!H.emit[k]) k = k.startsWith('UNK-cap') ? 'UNK-cap' : 'UNK'; // class never seen in training
  return { k, unk: true };
}
function tokenize(s) {
  return (s.match(/[A-Za-z]+(?=n't)|n't|'s|'re|'ve|'ll|'d|'m|\$|[0-9]+(?:[.,:/-][0-9]+)*|[A-Za-z]+(?:[-'][A-Za-z]+)*|[.,;:!?()"`]|\S/g) || [])
    .map((t) => ({ '(': '-LRB-', ')': '-RRB-', ';': ';' }[t] || t));
}

function viterbi(words) {
  const n = words.length, NEG = -Infinity;
  const V = [], B = [], keys = words.map((w, i) => key(w, i));
  for (let i = 0; i < n; i++) {
    const e = H.emit[keys[i].k] || {};
    const row = new Array(T).fill(NEG), back = new Array(T).fill(-1);
    const anyTag = tags.some((tg) => e[tg] != null);
    for (let t = 0; t < T; t++) {
      // If no tag ever emitted this key, fall back to the smoothing floor so decoding never dead-ends.
      const em = anyTag ? e[tags[t]] : H.floor[tags[t]];
      if (em == null) continue;
      if (i === 0) { row[t] = H.trans[T][t] + em; continue; }
      let best = NEG, arg = -1;
      for (let p = 0; p < T; p++) { const v = V[i - 1][p] + H.trans[p][t]; if (v > best) { best = v; arg = p; } }
      row[t] = best + em; back[t] = arg;
    }
    V.push(row); B.push(back);
  }
  let last = -1, bestEnd = NEG;
  for (let t = 0; t < T; t++) { const v = V[n - 1][t] + H.trans[t][T]; if (v > bestEnd) { bestEnd = v; last = t; } }
  const path = [last];
  for (let i = n - 1; i > 0; i--) path.push(B[i][path[path.length - 1]]);
  path.reverse();
  return { V, B, path, keys, logp: bestEnd };
}

// Most-frequent-tag baseline: argmax_t P(w|t)P(t) = argmax_t count(w,t); unknown words → NN.
const total = () => Object.values(H.tag_counts).reduce((a, b) => a + b, 0);
function baseline(words) {
  const N = total();
  return words.map((w, i) => {
    const k = H.emit[w] ? w : (i === 0 && H.emit[w.toLowerCase()]) ? w.toLowerCase() : null;
    if (!k) return 'NN';
    let best = null, bv = -Infinity;
    for (const [t, lp] of Object.entries(H.emit[k])) { const v = lp + Math.log(H.tag_counts[t] / N); if (v > bv) { bv = v; best = t; } }
    return best;
  });
}

export function coarse(t) {
  if (/^NN/.test(t)) return 'noun';
  if (/^VB|^MD$/.test(t)) return 'verb';
  if (/^JJ/.test(t)) return 'adj';
  if (/^RB|^WRB$/.test(t)) return 'adv';
  if (/^PRP|^WP/.test(t)) return 'pron';
  if (/^DT$|^PDT$|^WDT$/.test(t)) return 'det';
  if (/^IN$|^TO$|^RP$/.test(t)) return 'adp';
  if (t === 'CC') return 'conj';
  if (t === 'CD') return 'num';
  if (/^[.,:`'$#]|^-[LR]RB-$|^''$|^``$/.test(t)) return 'punct';
  return 'other';
}

let _p = null;
export function loadPOS(url = '/lab/pos/data/hmm.json') {
  return (_p ||= fetch(url).then((r) => { if (!r.ok) throw new Error(`Couldn’t load ${url} (${r.status})`); return r.json(); }).then((data) => {
    H = data; tags = H.tags; T = tags.length;
    const tag = (words) => (words.length ? viterbi(words).path.map((t) => tags[t]) : []);
    return { tag, tokenize, tagSentence: (s) => { const w = tokenize(s); return { words: w, tags: tag(w) }; }, tags, metrics: H.metrics };
  }).catch((e) => { _p = null; throw e; }));
}
