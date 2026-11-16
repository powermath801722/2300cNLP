/**
 * Braidwater Text Studio · ner.js — named entities with BERT-base NER (CoNLL-2003: PER, ORG, LOC, MISC),
 * grouping WordPiece predictions into entities exactly as /lab/ner/app.js does (first-piece tag, BIO spans).
 * Side-effect free; transformers.js and the ≈109 MB model load only when loadNER() is called.
 *
 * Exports:
 *   NER_MODEL = 'Xenova/bert-base-NER', NER_SIZE_MB = 109, TYPES (labels and colors)
 *   loadNER({ ui }) → Promise<{ entities(text, { onProgress }) → Promise<[{ type, text, start, end, score }]> }>
 *       long text is processed in sentence-aligned chunks; offsets are into the full text
 *   group(text, tokenPredictions) → entities for one chunk
 */
export const NER_MODEL = 'Xenova/bert-base-NER';
export const NER_SIZE_MB = 109;
export const TYPES = {
  PER: { name: 'Person', color: '#d8b66a' },
  ORG: { name: 'Organization', color: '#7ea3c8' },
  LOC: { name: 'Location', color: '#8fbf9a' },
  MISC: { name: 'Miscellaneous', color: '#b49bd6' },
};

// Map WordPiece (##) / SentencePiece (▁) tokens back to character offsets in the original text.
// transformers.js tokenizers don't return offset mappings, so we walk the text left to right.
// Returns one entry per token: { start, end, cont } (cont = continues the previous word) or null
// for special/unknown tokens that can't be located.
const SPECIAL = /^\[(CLS|SEP|PAD|MASK)\]$|^<\/?s>$|^<pad>$|^<unk>$/;

function alignTokens(text, tokens, { lower = false } = {}) {
  const hay = lower ? text.toLowerCase() : text;
  let cur = 0;
  const out = [];
  for (const t of tokens) {
    if (SPECIAL.test(t) || t === '[UNK]') { out.push(null); continue; }
    let piece = t, cont = false;
    if (piece.startsWith('##')) { piece = piece.slice(2); cont = true; }
    if (piece.startsWith('▁')) piece = piece.slice(1);
    if (!piece) { out.push(null); continue; }
    const p = lower ? piece.toLowerCase() : piece;
    const i = hay.indexOf(p, cur);
    // A token that only matches far ahead is a misalignment (e.g. after an [UNK]); skip it rather than jump.
    if (i < 0 || i - cur > 40) { out.push(null); continue; }
    out.push({ start: i, end: i + p.length, cont });
    cur = i + p.length;
  }
  return out;
}

/** Group WordPiece predictions into words (first-piece tag) and words into BIO entities. */
export function group(text, toks) {
  const offs = alignTokens(text, toks.map((t) => t.word));
  const words = [];
  toks.forEach((t, i) => {
    const o = offs[i];
    const cont = t.word.startsWith('##');
    if (cont && words.length) {
      const w = words[words.length - 1];
      w.pieces.push(t);
      if (o) w.end = o.end;
    } else {
      words.push({ tag: t.entity, start: o ? o.start : null, end: o ? o.end : null, pieces: [t] });
    }
  });
  const ents = [];
  let cur = null;
  for (const w of words) {
    const m = /^([BI])-(\w+)$/.exec(w.tag);
    if (!m) { cur = null; continue; }
    const [, bi, type] = m;
    if (bi === 'B' || !cur || cur.type !== type) {
      cur = { type, words: [w] };
      ents.push(cur);
    } else cur.words.push(w);
  }
  return ents.map((e) => {
    const pieces = e.words.flatMap((w) => w.pieces);
    const start = e.words[0].start, end = e.words[e.words.length - 1].end;
    return {
      type: e.type, start, end,
      text: start != null && end != null ? text.slice(start, end) : e.words.map((w) => w.pieces.map((p) => p.word.replace(/^##/, '')).join('')).join(' '),
      score: pieces.reduce((a, p) => a + p.score, 0) / pieces.length,
      pieces: pieces.map((p) => p.word),
    };
  });
}

/** Split into chunks of whole sentences of at most ~maxChars characters. */
function chunks(text, maxChars = 1200) {
  const out = []; let start = 0;
  const re = /[.!?]["”’)\]]*\s+/g; let m, last = 0;
  while ((m = re.exec(text))) {
    const end = m.index + m[0].length;
    if (end - start > maxChars && last > start) { out.push([start, last]); start = last; }
    last = end;
  }
  if (text.length - start > maxChars && last > start) { out.push([start, last]); start = last; }
  if (start < text.length) out.push([start, text.length]);
  return out.filter(([a, b]) => text.slice(a, b).trim());
}

let _p = null;
export function loadNER({ ui } = {}) {
  return (_p ||= import('/lab/shared/hf.js').then(({ pipe }) => pipe('token-classification', NER_MODEL, { ui, label: `BERT-base NER (≈${NER_SIZE_MB} MB)`, readyText: 'Entity model ready · cached for next time' }))
    .then((ner) => ({
      async entities(text, { onProgress } = {}) {
        const parts = chunks(text), all = [];
        for (let i = 0; i < parts.length; i++) {
          const [a, b] = parts[i], sub = text.slice(a, b);
          const toks = await ner(sub, { ignore_labels: [] });
          for (const e of group(sub, toks)) all.push({ ...e, start: e.start == null ? null : e.start + a, end: e.end == null ? null : e.end + a });
          onProgress && onProgress((i + 1) / parts.length);
          await new Promise((r) => setTimeout(r, 0));
        }
        return all;
      },
    })).catch((e) => { _p = null; throw e; }));
}
