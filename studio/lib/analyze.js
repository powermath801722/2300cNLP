/**
 * Braidwater Text Studio · analyze.js — the full classic (no-download) analysis pipeline as one pure function.
 * Side-effect free: data for each model loads lazily on the first call and is cached.
 *
 * Exports:
 *   loadClassic() → Promise<{ sentiment, pos, topics, spell }>   loads all shipped models (≈1.7 MB total)
 *   analyzeText(text, { summaryLength, spelling=true }) → Promise<report>   spelling:false leaves style.spelling null (it is the slow step)
 *   splitText(text) → [{ text, heading, para }]   the sentence list every section indexes into (first 400 kept)
 *   summaryFor(text, k) → { picked: number[] (sentence indices), lead }   recompute the summary only
 *   REPORT_VERSION
 *
 * report = {
 *   version, generatedAt, timings: { totalMs, ... },
 *   headings: [{ para, text }]  (short unterminated lines; not counted as sentences),
 *   sentences: [{ index, text, para, words, sentiment: { vader, nb, lr } }],
 *   overview: readability(...) & { paragraphs, longestSentences: [index] },
 *   sentiment: { overall: { vader, nb, lr, label }, mostPositive: [index], mostNegative: [index], arc: [vader compound per sentence] },
 *   style: { pos: posProfile, passives, lengths, spelling: [{ word, start, end, suggestions }] },
 *   keyIdeas: { summary: [index], summaryLength, keywords: [{ text, score, phrase }] },
 *   topics: { mixture: [{ k, label, p, color }], known, total, coverage } | { mixture: [], known: 0, ... },
 * }
 */
import { summarize, keywords } from './textrank.js';
import { sentences as tokSentences } from './tokenize.js';
import { readability } from './readability.js';
import { loadSentiment } from './sentiment.js';
import { loadPOS } from './pos.js';
import { loadTopics } from './topics.js';
import { loadSpell } from './spell.js';
import { posProfile, passives, sentenceLengths } from './style.js';

export const REPORT_VERSION = 1;

export function loadClassic() {
  return Promise.all([loadSentiment(), loadPOS(), loadTopics(), loadSpell()])
    .then(([sentiment, pos, topics, spell]) => ({ sentiment, pos, topics, spell }));
}

/**
 * Paragraph-aware splitting (headings = short unterminated paragraphs, as in the TextRank module) with the
 * tokenize module's abbreviation-aware sentence segmentation inside each paragraph (handles Nov., Dr., U.S., initials).
 */
export function splitText(text) {
  const paras = text.replace(/\r/g, '').split(/\n\s*\n/).map((p) => p.replace(/\s*\n\s*/g, ' ').trim()).filter(Boolean);
  const out = [];
  paras.forEach((p, para) => {
    if (!/[.!?]["”’')\]]*$/.test(p) && p.split(/\s+/).length < 12) { out.push({ text: p, heading: true, para }); return; }
    for (const s of tokSentences(p)) out.push({ text: s, heading: false, para });
  });
  return out;
}

const defaultK = (n) => Math.max(1, Math.min(5, Math.round(n * 0.2)));

export function summaryFor(text, k) {
  const r = summarize(text, { k, split: splitText });
  return { picked: r.picked, lead: r.lead };
}

const label3 = (x) => (x >= 0.05 ? 'pos' : x <= -0.05 ? 'neg' : 'neu');

export async function analyzeText(text, { summaryLength, spelling = true } = {}) {
  const M = await loadClassic();
  const t = {}, T0 = performance.now();
  let t0 = T0;
  const lap = (k) => { const n = performance.now(); t[k] = +(n - t0).toFixed(1); t0 = n; };

  const all = splitText(text).slice(0, 400);
  const cand = all.filter((s) => !s.heading);
  const paragraphs = new Set(all.map((s) => s.para)).size;
  lap('split');

  const sentences = cand.map((s, index) => {
    const v = M.sentiment.vader(s.text), nb = M.sentiment.nb(s.text), lr = M.sentiment.lr(s.text);
    return { index, text: s.text, para: s.para,
      sentiment: { vader: { label: v.label, compound: +v.score.toFixed(4) }, nb: { label: nb.label, prob: +nb.prob.toFixed(4) }, lr: { label: lr.label, prob: +lr.prob.toFixed(4) } } };
  });
  lap('sentiment');

  const tagged = cand.map((s, index) => {
    const { words, tags } = M.pos.tagSentence(s.text.slice(0, 1500));
    sentences[index].words = words.filter((w) => /\w/.test(w)).length;
    return { index, text: s.text, words, tags };
  });
  lap('pos');

  const overview = readability(text, { sentences: Math.max(1, cand.length) });
  overview.paragraphs = paragraphs;
  overview.longestSentences = sentences.map((s) => [s.words, s.index]).sort((a, b) => b[0] - a[0]).slice(0, 3).map(([, i]) => i);
  lap('readability');

  const arc = sentences.map((s) => s.sentiment.vader.compound);
  const meanV = arc.length ? arc.reduce((a, b) => a + b, 0) / arc.length : 0;
  const nbDoc = M.sentiment.nb(text), lrDoc = M.sentiment.lr(text);
  const votes = [label3(meanV), nbDoc.label, lrDoc.label];
  const pos = votes.filter((x) => x === 'pos').length, neg = votes.filter((x) => x === 'neg').length;
  const order = arc.map((v, i) => [v, i]).sort((a, b) => b[0] - a[0]);
  const sentiment = {
    overall: {
      vader: { label: label3(meanV), compound: +meanV.toFixed(4), note: 'mean compound over sentences' },
      nb: { label: nbDoc.label, prob: +nbDoc.prob.toFixed(4) },
      lr: { label: lrDoc.label, prob: +lrDoc.prob.toFixed(4) },
      label: pos >= 2 ? 'pos' : neg >= 2 ? 'neg' : 'mixed',
      agreement: `${Math.max(pos, neg, votes.filter((x) => x === 'neu').length)} of 3`,
    },
    mostPositive: order.filter(([v]) => v > 0.05).slice(0, 3).map(([, i]) => i),
    mostNegative: order.slice().reverse().filter(([v]) => v < -0.05).slice(0, 3).map(([, i]) => i),
    arc,
  };
  lap('docSentiment');

  const style = { pos: posProfile(tagged), passives: passives(tagged), lengths: sentenceLengths(tagged) };
  lap('style');
  style.spelling = spelling ? M.spell.checkText(text) : null;
  if (spelling) lap('spelling');

  const k = summaryLength || defaultK(cand.length);
  const sum = summarize(text, { k, split: splitText });
  const keyIdeas = { summary: sum.picked, lead: sum.lead, summaryLength: Math.min(k, cand.length), keywords: keywords(text).slice(0, 12) };
  lap('textrank');

  const inf = M.topics.infer(text, 300);
  const topics = inf.theta
    ? { mixture: inf.theta.map((p, k2) => ({ k: k2, label: M.topics.labels[k2], p: +p.toFixed(4), color: M.topics.palette[k2] })).sort((a, b) => b.p - a.p),
      known: inf.known, total: inf.total, coverage: inf.total ? inf.known / inf.total : 0, corpus: M.topics.meta.corpus }
    : { mixture: [], known: 0, total: inf.total, coverage: 0, corpus: M.topics.meta.corpus };
  lap('topics');

  t.totalMs = +(performance.now() - T0).toFixed(1);
  return { version: REPORT_VERSION, generatedAt: new Date().toISOString(), timings: t, chars: text.length,
    headings: all.filter((s) => s.heading).map((s) => ({ para: s.para, text: s.text })),
    sentences, overview, sentiment, style, keyIdeas, topics };
}
