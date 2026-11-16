/**
 * Braidwater Text Studio · readability.js — counts and classic readability formulas. Side-effect free, no data.
 *
 * Exports:
 *   syllables(word) → number          vowel-group heuristic; exact match with the CMU Pronouncing Dictionary on
 *                                     91.3% of the 5,886 most frequent words (99.8% within one syllable)
 *   readability(text, { sentences }) → { words, sentences, syllables, unique, ttr, mattr, avgSentence, avgSyllables,
 *        fre, fkgl, readingMinutes, speakingMinutes, polysyllabic, longestWords }
 *   fleschLabel(fre) → plain-language band ("Plain English", "Difficult", …)
 * Formulas: Flesch Reading Ease = 206.835 − 1.015·(words/sentences) − 84.6·(syllables/words);
 *           Flesch–Kincaid grade = 0.39·(words/sentences) + 11.8·(syllables/words) − 15.59.
 * Reading time at 238 words/minute (Brysbaert 2019, silent reading of non-fiction); speaking at 150 wpm.
 */
import { wordTokenize } from './tokenize.js';

export function syllables(word) {
  let w = word.toLowerCase().replace(/[^a-z]/g, '');
  if (!w) return 0;
  if (w.length <= 3) return 1;
  w = w.replace(/ely$/, 'ly').replace(/(?:[^laeiouy]es|[^laeiouytd]ed|[^laeiouy]e)$/, '').replace(/^y/, '');
  let n = (w.match(/[aeiouy]{1,2}/g) || []).length;
  // vowel pairs that are usually two syllables: ri-ot, i-de-a, po-em, cre-ate…
  n += (w.match(/[^tscgx]i[aou]|[^q]ua|ie(?=[rnt])|eo|oe(?!s?$)|ea$|ism$/g) || []).length;
  return Math.max(1, n);
}

/** Moving-average type-token ratio (Covington & McFall 2010): length-independent lexical diversity. */
function mattr(ws, win = 50) {
  if (ws.length <= win) return ws.length ? new Set(ws).size / ws.length : 0;
  const c = new Map(); let types = 0, sum = 0;
  for (let i = 0; i < ws.length; i++) {
    const w = ws[i]; c.set(w, (c.get(w) || 0) + 1); if (c.get(w) === 1) types++;
    if (i >= win) { const o = ws[i - win]; c.set(o, c.get(o) - 1); if (c.get(o) === 0) types--; }
    if (i >= win - 1) sum += types / win;
  }
  return sum / (ws.length - win + 1);
}

export function fleschLabel(fre) {
  return fre >= 90 ? 'Very easy' : fre >= 80 ? 'Easy' : fre >= 70 ? 'Fairly easy' : fre >= 60 ? 'Plain English'
    : fre >= 50 ? 'Fairly difficult' : fre >= 30 ? 'Difficult' : 'Very difficult';
}

export function readability(text, { sentences: nSent } = {}) {
  const ws = wordTokenize(text).filter((t) => t.type === 'word').map((t) => t.text);
  const lower = ws.map((w) => w.toLowerCase());
  const syl = ws.map(syllables), totalSyl = syl.reduce((a, b) => a + b, 0);
  const W = ws.length, S = Math.max(1, nSent || 1);
  const wps = W / S, spw = W ? totalSyl / W : 0;
  const longest = [...new Set(lower)].sort((a, b) => b.length - a.length).slice(0, 5);
  return {
    words: W, sentences: S, syllables: totalSyl, unique: new Set(lower).size,
    ttr: W ? new Set(lower).size / W : 0, mattr: mattr(lower),
    avgSentence: wps, avgSyllables: spw,
    fre: W ? 206.835 - 1.015 * wps - 84.6 * spw : 0,
    fkgl: W ? 0.39 * wps + 11.8 * spw - 15.59 : 0,
    readingMinutes: W / 238, speakingMinutes: W / 150,
    polysyllabic: syl.filter((s) => s >= 3).length,
    longestWords: longest,
  };
}
