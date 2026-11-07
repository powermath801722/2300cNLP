// Map WordPiece (##) / SentencePiece (▁) tokens back to character offsets in the original text.
// transformers.js tokenizers don't return offset mappings, so we walk the text left to right.
// Returns one entry per token: { start, end, cont } (cont = continues the previous word) or null
// for special/unknown tokens that can't be located.
const SPECIAL = /^\[(CLS|SEP|PAD|MASK)\]$|^<\/?s>$|^<pad>$|^<unk>$/;

export function alignTokens(text, tokens, { lower = false } = {}) {
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
