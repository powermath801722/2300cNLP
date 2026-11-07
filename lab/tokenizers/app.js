// Subword tokenizers: BPE trained from scratch (Sennrich et al. 2016) + pretrained GPT-2 / BERT / T5 tokenizers.
import { el, $, ready, plotly, plotLayout, PLOT_CONFIG, COLORS } from '/lab/shared/lab.js';
import { tokenizer, progressUI } from '/lab/shared/hf.js';

const EOW = '</w>';
const CORPORA = {
  classic: Array(5).fill('low').concat(Array(2).fill('lower'), Array(6).fill('newest'), Array(3).fill('widest')).join(' '),
  prose: `The river rises in the hills and runs slowly toward the sea. Rivers carry water, sand and stories.
Along the riverside the walkers walked, the talkers talked and the readers read. Slower rivers widen; faster rivers deepen.
The newest bridges are lower than the oldest bridges, and the widest river is not always the deepest river.
Every reader rereads the river differently, and every rereading changes the reader. Unreadable maps were redrawn,
redrawn maps were reread, and the rereadable maps were kept. Walking, talking, reading and rewriting: the river keeps going.`,
};

// ---------- BPE ----------
function words(text) {
  const freq = new Map();
  for (const w of String(text).toLowerCase().match(/[\p{L}\p{N}']+/gu) || []) freq.set(w, (freq.get(w) || 0) + 1);
  return freq;
}
function train(text, maxMerges = 400) {
  const freq = words(text);
  const entries = [...freq].map(([w, f]) => ({ w, f, s: [...w, EOW] }));
  const alphabet = new Set(); for (const e of entries) for (const c of e.s) alphabet.add(c);
  const corpusLen = () => entries.reduce((t, e) => t + e.f * e.s.length, 0);
  const merges = [], history = [{ vocab: alphabet.size, tokens: corpusLen() }];
  for (let m = 0; m < maxMerges; m++) {
    const counts = new Map();
    for (const e of entries) for (let i = 0; i < e.s.length - 1; i++) { const k = e.s[i] + '\u0000' + e.s[i + 1]; counts.set(k, (counts.get(k) || 0) + e.f); }
    let best = null, bc = 0;
    for (const [k, c] of counts) if (c > bc || (c === bc && best !== null && k < best)) { best = k; bc = c; }
    if (!best || bc < 2) break;
    const [a, b] = best.split('\u0000'), ab = a + b;
    for (const e of entries) {
      const out = [];
      for (let i = 0; i < e.s.length; i++) { if (i < e.s.length - 1 && e.s[i] === a && e.s[i + 1] === b) { out.push(ab); i++; } else out.push(e.s[i]); }
      e.s = out;
    }
    merges.push({ a, b, ab, count: bc });
    history.push({ vocab: alphabet.size + merges.length, tokens: corpusLen() });
  }
  return { freq, merges, history, alphabet };
}
function encode(word, merges, k) {
  const rank = new Map(); for (let i = 0; i < k; i++) rank.set(merges[i].a + '\u0000' + merges[i].b, i);
  let s = [...String(word).toLowerCase(), EOW];
  for (;;) {
    let bi = -1, br = Infinity;
    for (let i = 0; i < s.length - 1; i++) { const r = rank.get(s[i] + '\u0000' + s[i + 1]); if (r !== undefined && r < br) { br = r; bi = i; } }
    if (bi < 0) break;
    const { a, b, ab } = merges[br], out = [];
    for (let i = 0; i < s.length; i++) { if (i < s.length - 1 && s[i] === a && s[i + 1] === b) { out.push(ab); i++; } else out.push(s[i]); }
    s = out;
  }
  return s;
}

const B = { model: null, k: 0, playing: null };
const tokChip = (t, i, extra = '') => el('span', { class: `tok${i % 2 ? ' alt' : ''}${extra}`, text: t });

function retrain() {
  B.model = train($('#corpus').value);
  const K = B.model.merges.length;
  const kIn = $('#k'); kIn.max = K; $('#kmax').textContent = K;
  B.k = Math.min(K, B.k || K);
  if (!B.k) B.k = Math.min(K, 10);
  kIn.value = B.k;
  const tb = $('#mergetab tbody');
  tb.replaceChildren(...B.model.merges.map((m, i) => el('tr', { 'data-i': i }, [
    el('td', { text: String(i + 1) }), el('td', { text: `${m.a} + ${m.b}` }), el('td', {}, el('code', { text: m.ab })), el('td', { text: String(m.count) }),
  ])));
  render();
}
function render() {
  const { merges, history, freq, alphabet } = B.model, k = B.k;
  $('#kval').textContent = k; document.querySelectorAll('.kv').forEach((n) => (n.textContent = k)); $('#k').value = k;
  $('#mergetab tbody').querySelectorAll('tr').forEach((tr, i) => { tr.className = i === k - 1 ? 'cur' : i >= k ? 'future' : ''; });
  const cur = $('#mergetab tbody tr.cur'); if (cur) cur.scrollIntoView({ block: 'nearest' });
  const h = history[k];
  $('#bpestats').textContent = `${freq.size} distinct words · ${alphabet.size} base symbols · vocabulary ${h.vocab} · training text is ${h.tokens} tokens (${history[0].tokens} as characters, ${(100 * (1 - h.tokens / history[0].tokens)).toFixed(0)}% shorter).`;
  const top = [...freq].sort((a, b) => b[1] - a[1] || a[0].localeCompare(b[0])).slice(0, 12);
  $('#segs').replaceChildren(...top.map(([w, f]) => el('div', { class: 'w' }, [el('small', { text: '×' + f }), ...encode(w, merges, k).map((t, i) => tokChip(t, i))])));
  segHeld();
  drawGrowth();
  window.__lab.bpe = { k, merges: merges.length, vocab: h.vocab, tokens: h.tokens, held: encode($('#heldout').value || '', merges, k) };
}
function segHeld() {
  const w = ($('#heldout').value || '').trim();
  const s = w ? encode(w, B.model.merges, B.k) : [];
  $('#heldseg').replaceChildren(...s.map((t, i) => tokChip(t, i)), el('span', { class: 'note', style: { marginLeft: '8px' }, text: s.length ? `${s.length} tokens` : '' }));
}
let growthT;
function drawGrowth() {
  clearTimeout(growthT);
  growthT = setTimeout(async () => {
    const P = await plotly(), h = B.model.history, xs = h.map((_, i) => i);
    P.react($('#growth'), [
      { x: xs, y: h.map((r) => r.vocab), name: 'vocabulary size', line: { color: COLORS[0], width: 2 } },
      { x: xs, y: h.map((r) => r.tokens), name: 'training text length (tokens)', yaxis: 'y2', line: { color: COLORS[1], width: 2 } },
    ], plotLayout({
      margin: { l: 56, r: 64, t: 30, b: 44 }, showlegend: true,
      legend: { orientation: 'h', x: 0, y: 1.08, yanchor: 'bottom', bgcolor: 'rgba(0,0,0,0)' },
      xaxis: { title: { text: 'merges', font: { size: 12, color: '#9aa3ad' } }, gridcolor: 'rgba(228,231,235,.08)', color: '#9aa3ad' },
      yaxis: { title: { text: 'vocabulary', font: { size: 12, color: COLORS[0] } }, gridcolor: 'rgba(228,231,235,.08)', color: '#9aa3ad' },
      yaxis2: { title: { text: 'tokens', font: { size: 12, color: COLORS[1] } }, overlaying: 'y', side: 'right', color: '#9aa3ad', showgrid: false },
      shapes: [{ type: 'line', x0: B.k, x1: B.k, yref: 'paper', y0: 0, y1: 1, line: { color: 'rgba(228,231,235,.45)', dash: 'dot', width: 1 } }],
    }), PLOT_CONFIG);
  }, 60);
}

// ---------- pretrained comparison ----------
const TOKENIZERS = [
  { id: 'Xenova/gpt2', name: 'GPT-2', kind: 'Byte-level BPE · 50,257' },
  { id: 'Xenova/bert-base-uncased', name: 'BERT (uncased)', kind: 'WordPiece · 30,522' },
  { id: 'Xenova/t5-small', name: 'T5', kind: 'SentencePiece unigram · 32,100' },
];
const EXAMPLES = [
  ['Words', 'Tokenization is unbelievably important for understanding transformers.'],
  ['Numbers', 'In 2026 the ticket cost $1,234.56, up 12.5% from 987.65.'],
  ['Accents', 'A naïve café owner served crème brûlée in the façade’s shade.'],
  ['Emoji', 'I ❤️ NLP 🤖✨ and 🌊 rivers!'],
  ['Other scripts', '自然语言处理 · обработка естественного языка · معالجة اللغة'],
  ['Code', 'def square(x):\n    return x ** 2  # power'],
  ['Rare word', 'pneumonoultramicroscopicsilicovolcanoconiosis and Braidwater'],
];
async function compare() {
  const text = $('#cmpin').value;
  const ui = progressUI($('#cmpload'), 'tokenizers');
  $('#cmpgo').disabled = true;
  try {
    ui.set('Loading tokenizer files (about 5 MB total, cached after the first time)…', 0.1);
    const toks = await Promise.all(TOKENIZERS.map((t) => tokenizer(t.id)));
    ui.done('Ready · 3 tokenizers');
    const result = {};
    $('#cmpout').replaceChildren(...TOKENIZERS.map((t, k) => {
      const tok = toks[k];
      const pieces = tok.tokenize(text);
      const ids = tok.encode(text, { add_special_tokens: false });
      const unkId = tok.model.tokens_to_ids ? tok.model.tokens_to_ids.get(tok.unk_token) : undefined;
      const unk = ids.filter((id) => id === unkId).length;
      result[t.name] = { n: ids.length, unk, pieces };
      const chips = pieces.map((p, i) => {
        const id = ids[i];
        let dec = ''; try { dec = id != null ? tok.decode([id]) : ''; } catch { /* ignore */ }
        const isUnk = id === unkId;
        const n = el('span', { class: `tok${i % 2 ? ' alt' : ''}${isUnk ? ' unk' : ''}`, tabindex: '0', text: p.replace(/\n/g, '⏎'), title: `id ${id ?? '?'} · decodes to “${dec}”` });
        return n;
      });
      return el('div', { class: 'panel tkcard' }, [
        el('h3', { text: t.name }), el('span', { class: 'badge', style: { alignSelf: 'flex-start' }, text: t.kind }),
        el('div', { class: 'mini' }, [
          el('span', {}, [el('b', { text: String(ids.length) }), ' tokens']),
          el('span', {}, [el('b', { text: ids.length ? (text.length / ids.length).toFixed(2) : '–' }), ' chars/token']),
          el('span', {}, [el('b', { text: String(unk) }), ' unknown']),
        ]),
        el('div', { class: 'tkline' }, chips),
      ]);
    }));
    window.__lab.compare = result;
  } catch (e) { ui.error(e); }
  finally { $('#cmpgo').disabled = false; }
}

// ---------- init ----------
$('#corpus').value = CORPORA.classic;
$('#c-classic').addEventListener('click', () => { $('#corpus').value = CORPORA.classic; $('#heldout').value = 'lowest'; B.k = 0; retrain(); });
$('#c-prose').addEventListener('click', () => { $('#corpus').value = CORPORA.prose; $('#heldout').value = 'rereadable'; B.k = 0; retrain(); });
let rt; $('#corpus').addEventListener('input', () => { clearTimeout(rt); rt = setTimeout(() => { B.k = 0; retrain(); }, 250); });
$('#k').addEventListener('input', (e) => { B.k = +e.target.value; render(); });
$('#step').addEventListener('click', () => { if (B.k < B.model.merges.length) { B.k++; render(); } });
$('#heldout').addEventListener('input', () => { segHeld(); window.__lab.bpe.held = encode($('#heldout').value || '', B.model.merges, B.k); });
$('#play').addEventListener('click', () => {
  if (B.playing) { clearInterval(B.playing); B.playing = null; $('#play').textContent = 'Play'; return; }
  if (B.k >= B.model.merges.length) B.k = 0;
  $('#play').textContent = 'Pause';
  B.playing = setInterval(() => {
    if (B.k >= B.model.merges.length) { clearInterval(B.playing); B.playing = null; $('#play').textContent = 'Play'; return; }
    B.k++; render();
  }, matchMedia('(prefers-reduced-motion: reduce)').matches ? 900 : 350);
});
$('#cmpin').value = EXAMPLES[0][1];
$('#cmpchips').append(...EXAMPLES.map(([label, t]) => el('button', { class: 'chip', type: 'button', text: label, onclick: () => { $('#cmpin').value = t; compare(); } })));
$('#cmpgo').addEventListener('click', compare);
B.k = 0;
retrain();
ready({ module: 'tokenizers' });
