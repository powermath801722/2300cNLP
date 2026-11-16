/**
 * Braidwater Text Studio · style.js — grammar and style signals from POS tags. Side-effect free, no data
 * (pass in a tagger from pos.js).
 *
 * Exports:
 *   posProfile(taggedSentences) → { counts: { noun, verb, adj, adv, pron, det, adp, conj, num, other }, ratios, total }
 *       punctuation is excluded; ratios sum to 1
 *   passives(taggedSentences) → [{ sentence, phrase, byAgent }]
 *       heuristic: a form of "be"/"get", up to two adverbs or "not", then a past participle (VBN)
 *   sentenceLengths(taggedSentences) → { lengths, long: [indices with > 30 words], mean, sd }
 *   taggedSentences format: [{ index, text, words: string[], tags: string[] }]
 */
import { coarse } from './pos.js';

const BE = new Set(['am', 'is', 'are', 'was', 'were', 'be', 'been', 'being', "'s", "'re", "'m", 'get', 'gets', 'got', 'gotten', 'getting']);
const KEYS = ['noun', 'verb', 'adj', 'adv', 'pron', 'det', 'adp', 'conj', 'num', 'other'];

export function posProfile(sents) {
  const counts = Object.fromEntries(KEYS.map((k) => [k, 0]));
  let total = 0;
  for (const s of sents) for (const t of s.tags) { const c = coarse(t); if (c === 'punct') continue; counts[c] += 1; total++; }
  const ratios = Object.fromEntries(KEYS.map((k) => [k, total ? counts[k] / total : 0]));
  return { counts, ratios, total };
}

export function passives(sents) {
  const out = [];
  for (const s of sents) {
    const w = s.words, t = s.tags;
    for (let i = 0; i < w.length; i++) {
      if (!BE.has(w[i].toLowerCase()) || !/^VB|^MD/.test(t[i]) && !/^'/.test(w[i])) continue;
      let j = i + 1;
      while (j < w.length && j <= i + 3 && (/^RB/.test(t[j]) || /^(not|n't)$/i.test(w[j]))) j++;
      if (j < w.length && t[j] === 'VBN') {
        const byAgent = w.slice(j + 1, j + 6).some((x) => x.toLowerCase() === 'by');
        out.push({ sentence: s.index, phrase: w.slice(i, j + 1).join(' ').replace(/ n't/g, "n't"), byAgent });
        i = j;
      }
    }
  }
  return out;
}

export function sentenceLengths(sents) {
  const lengths = sents.map((s) => s.words.filter((x, k) => !/^[.,:;!?`'"()$#-]/.test(s.tags[k]) && /\w/.test(x)).length);
  const n = lengths.length || 1, mean = lengths.reduce((a, b) => a + b, 0) / n;
  const sd = Math.sqrt(lengths.reduce((a, b) => a + (b - mean) ** 2, 0) / n);
  return { lengths, long: lengths.map((l, i) => [l, i]).filter(([l]) => l > 30).map(([, i]) => i), mean, sd };
}
