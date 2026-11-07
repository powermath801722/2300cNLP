import { el, $, ready, fail, getJSON } from '/lab/shared/lab.js';

const EXAMPLES = [
  'I can can a can .',
  'Time flies like an arrow ; fruit flies like a banana .',
  'The old man the boat .',
  'They refuse to permit us to obtain the refuse permit .',
  'The complex houses married and single soldiers and their families .',
  'Colorless green ideas sleep furiously .',
  'She quickly realized the stocks were rising unexpectedly .',
];

const PENN = {
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
const cls = (t) => (/^NN/.test(t) ? 'c-N' : /^VB|^MD/.test(t) ? 'c-V' : /^JJ/.test(t) ? 'c-J' : /^RB|^WRB/.test(t) ? 'c-R' : /^DT|^PDT|^WDT/.test(t) ? 'c-D' : /^PRP|^WP|^IN|^TO|^CC/.test(t) ? 'c-P' : 'c-O');

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

function render() {
  const words = tokenize($('#sent').value.trim());
  const host = $('#tagged'); host.innerHTML = ''; $('#trellis').innerHTML = '';
  if (!words.length) { $('#tagnote').textContent = 'Type a sentence first.'; return; }
  if (words.length > 40) { $('#tagnote').textContent = 'Please keep it under 40 tokens.'; return; }
  const r = viterbi(words), base = baseline(words);
  const tagOf = r.path.map((t) => tags[t]);
  let diffs = 0;
  words.forEach((w, i) => {
    const t = tagOf[i], b = base[i], d = b !== t; diffs += d;
    host.append(el('div', { class: 'tw' + (r.keys[i].unk ? ' unk' : ''), 'data-tag': t }, [
      el('span', { class: 'w', text: w }),
      el('span', { class: 't ' + cls(t), tabindex: '0', title: `${t}: ${PENN[t] || t}${r.keys[i].unk ? ` · unknown word, treated as ${r.keys[i].k}` : ''}`, text: t }),
      el('span', { class: 'b' + (d ? ' diff' : ''), title: 'Most-frequent-tag baseline', text: b }),
    ]));
  });
  const unk = r.keys.filter((k) => k.unk).length;
  $('#tagnote').textContent = `Sequence log-probability ${r.logp.toFixed(1)} · ${unk} unknown word${unk === 1 ? '' : 's'} (*) · baseline differs on ${diffs} token${diffs === 1 ? '' : 's'} (gold).`;

  // trellis: rows = tags possible for at least one word
  const rows = tags.map((t, j) => j).filter((j) => r.V.some((col) => col[j] > -Infinity));
  const tbl = $('#trellis');
  tbl.append(el('thead', {}, el('tr', {}, [el('th', { text: 'tag \\ word' }), ...words.map((w) => el('th', { scope: 'col', text: w }))])));
  const tb = el('tbody');
  for (const j of rows) {
    tb.append(el('tr', {}, [el('th', { scope: 'row', title: PENN[tags[j]] || '', text: tags[j] }), ...words.map((w, i) => {
      const v = r.V[i][j];
      if (v === -Infinity) return el('td', { class: 'na', text: '—' });
      const best = r.path[i] === j;
      return el('td', { class: best ? 'best' : '', 'aria-label': `${w} as ${tags[j]}: ${v.toFixed(1)}${i ? ' from ' + tags[r.B[i][j]] : ''}${best ? ', on best path' : ''}` },
        [v.toFixed(1), el('small', { text: i ? '← ' + tags[r.B[i][j]] : 'start' })]);
    })]));
  }
  tbl.append(tb);
  window.__lab.last = { words, tags: tagOf, baseline: base };
}

async function init() {
  try { H = await getJSON('./data/hmm.json'); } catch (e) { fail(e.message); return; }
  tags = H.tags; T = tags.length;
  const m = H.metrics;
  $('#stats').append(...[
    [(m.hmm_acc * 100).toFixed(1) + '%', 'HMM accuracy, held-out tokens'],
    [(m.baseline_acc * 100).toFixed(1) + '%', 'most-frequent-tag baseline'],
    [(m.unk_acc * 100).toFixed(1) + '%', `on ${m.unk_tokens} unseen words`],
    [m.test_tokens.toLocaleString(), `test tokens (${m.test_sents} sentences)`],
    [T, 'tags'],
  ].map(([b, s]) => el('div', {}, [el('b', { text: String(b) }), el('span', { text: s })])));
  $('#conf').append(el('thead', {}, el('tr', {}, ['Gold', 'Predicted', 'Count'].map((x) => el('th', { text: x })))),
    el('tbody', {}, m.top_confusions.map(([g, p, c]) => el('tr', {}, [el('td', { title: PENN[g], text: g }), el('td', { title: PENN[p], text: p }), el('td', { text: String(c) })]))));
  $('#legend').append(...tags.map((t) => el('div', {}, [el('code', { class: cls(t), text: t }), el('span', { text: PENN[t] || t })])));
  $('#chips').append(...EXAMPLES.map((s) => el('button', { class: 'chip', type: 'button', text: s.replace(/ ([.,;!?])/g, '$1'), onclick: () => { $('#sent').value = s; render(); } })));
  $('#form').addEventListener('submit', (e) => { e.preventDefault(); render(); });
  $('#go').disabled = false;
  $('#status').textContent = `Ready · ${Object.keys(H.emit).length.toLocaleString()} known words and unknown-word classes · tokens are split Penn-style (don't → do n't).`;
  render();
  ready({ pos: true, tag: (s) => { const w = tokenize(s); return viterbi(w).path.map((t) => tags[t]); } });
}
init();
