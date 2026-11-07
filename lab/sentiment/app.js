import { el, $, ready, fail, getJSON, esc } from '/lab/shared/lab.js';

const SAMPLES = [
  ['The acting was not good at all.', 'Negation: bag-of-words models see “good”.'],
  ['Oh great, another three-hour sequel. Just what I needed.', 'Sarcasm: positive words, negative meaning.'],
  ['The battery life is terrible but the screen is absolutely gorgeous!', 'Contrast: the clause after “but” dominates.'],
  ['The steering on this car is unpredictable.', 'Domain shift: “unpredictable” is praise in film reviews.'],
  ['I expected to hate it, but honestly it was wonderful.', 'Expectation vs outcome.'],
  ['The film was BRILLIANT!!!', 'Caps and exclamation marks as intensity.'],
  ['Nothing about this movie was boring.', 'Negation of a negative word.'],
  ['I want my two hours back.', 'Implicit sentiment with no opinion words.'],
];

// ---------- VADER-style lexicon + rules (after Hutto & Gilbert 2014) ----------
const B_INCR = 0.293, B_DECR = -0.293, C_INCR = 0.733, N_SCALAR = -0.74;
const BOOST = Object.fromEntries([
  ...'absolutely amazingly awfully completely considerably decidedly deeply enormously entirely especially exceptionally extremely fabulously flipping flippin fricking frickin frigging friggin fully fucking greatly hella highly hugely incredibly intensely majorly more most particularly purely quite really remarkably so substantially thoroughly totally tremendously uber unbelievably unusually utterly very'.split(' ').map((w) => [w, B_INCR]),
  ...'almost barely hardly just enough kind of kinda kindof less little marginally occasionally partly scarcely slightly somewhat sort of sorta sortof'.split(' ').map((w) => [w, B_DECR]),
]);
const NEGATE = new Set("aint arent cannot cant couldnt darent didnt doesnt ain't aren't can't couldn't daren't didn't doesn't dont hadnt hasnt havent isnt mightnt mustnt neither don't hadn't hasn't haven't isn't mightn't mustn't neednt needn't never none nope nor not nothing nowhere oughtnt shant shouldnt uhuh wasnt werent oughtn't shan't shouldn't uh-uh wasn't weren't without wont wouldnt won't wouldn't rarely seldom despite".split(' '));

function vader(text, LEX) {
  const raw = text.split(/\s+/).map((t) => t.replace(/^[^\w:;=<>()'-]+|[^\w:;=<>()'-]+$/g, '')).filter((t) => t.length > 1 || /\w/.test(t));
  const isCap = (w) => /[A-Za-z]/.test(w) && w === w.toUpperCase();
  const mixed = raw.some(isCap) && !raw.every(isCap);
  const lower = raw.map((w) => w.toLowerCase().replace(/^'+|'+$/g, ''));
  const val = raw.map((w, i) => {
    const lw = lower[i];
    let v = LEX[lw] ?? LEX[raw[i]];
    if (v == null || BOOST[lw] != null) return 0;
    if (mixed && isCap(w)) v += v > 0 ? C_INCR : -C_INCR;
    for (let k = 1; k <= 3 && i - k >= 0; k++) {
      const p = lower[i - k], b = BOOST[p];
      if (b != null) { let s = (v > 0 ? b : -b) * [1, 0.95, 0.9][k - 1]; if (mixed && isCap(raw[i - k])) s += v > 0 ? C_INCR : -C_INCR; v += s; }
      if (NEGATE.has(p) || p.endsWith("n't")) v *= N_SCALAR;
    }
    return v;
  });
  const bi = lower.indexOf('but');
  if (bi >= 0) for (let i = 0; i < val.length; i++) { if (i < bi) val[i] *= 0.5; else if (i > bi) val[i] *= 1.5; }
  let s = val.reduce((a, b) => a + b, 0);
  const ex = Math.min((text.match(/!/g) || []).length, 4) * 0.292;
  const q = (text.match(/\?/g) || []).length, qa = q > 1 ? (q <= 3 ? q * 0.18 : 0.96) : 0;
  if (s > 0) s += ex + qa; else if (s < 0) s -= ex + qa;
  const compound = s / Math.sqrt(s * s + 15);
  const label = compound >= 0.05 ? 'pos' : compound <= -0.05 ? 'neg' : 'neu';
  return { label, score: compound, prob: (compound + 1) / 2, words: raw.map((w, i) => [w, val[i]]), detail: `compound ${compound.toFixed(3)}` };
}

// ---------- Naive Bayes + logistic regression (trained in training/sentiment_train.py) ----------
function makeBow(M) {
  const re = new RegExp(M.token_re, 'g');
  const idx = new Map(M.vocab.map((w, i) => [w, i]));
  const tokens = (text) => {
    const out = []; let m; const lt = text.toLowerCase(); re.lastIndex = 0;
    // keep original spans for display
    while ((m = re.exec(lt))) out.push({ w: m[0], i: idx.get(m[0]), s: m.index, e: m.index + m[0].length });
    return out;
  };
  const sig = (z) => 1 / (1 + Math.exp(-z));
  const spans = (text, toks, contrib) => {
    const res = []; let last = 0;
    toks.forEach((t, k) => { if (t.s > last) res.push([text.slice(last, t.s), null]); res.push([text.slice(t.s, t.e), contrib[k]]); last = t.e; });
    if (last < text.length) res.push([text.slice(last), null]);
    return res;
  };
  const nb = (text) => {
    const t = tokens(text);
    let z = M.nb.prior[1] - M.nb.prior[0];
    const c = t.map((x) => x.i == null ? 0 : M.nb.logp_pos[x.i] - M.nb.logp_neg[x.i]);
    for (const v of c) z += v;
    const p = sig(z);
    return { label: p >= 0.5 ? 'pos' : 'neg', prob: p, score: z, spans: spans(text, t, t.map((x, k) => x.i == null ? null : c[k])), detail: `log-odds ${z.toFixed(2)} · P(pos) ${p.toFixed(3)}`, known: t.filter((x) => x.i != null).length };
  };
  const lr = (text) => {
    const t = tokens(text), seen = new Set();
    let z = M.lr.bias;
    const c = t.map((x) => { if (x.i == null || seen.has(x.i)) return x.i == null ? null : 0; seen.add(x.i); return M.lr.w[x.i]; });
    for (const v of c) if (v) z += v;
    const p = sig(z);
    return { label: p >= 0.5 ? 'pos' : 'neg', prob: p, score: z, spans: spans(text, t, c), detail: `z = ${z.toFixed(2)} · P(pos) ${p.toFixed(3)}` };
  };
  return { nb, lr };
}

// ---------- rendering ----------
const NAMES = { vader: 'Lexicon + rules', nb: 'Naive Bayes', lr: 'Logistic regression', tf: 'DistilBERT (SST-2)' };
const LABEL = { pos: 'Positive', neg: 'Negative', neu: 'Neutral' };
function shade(v, scale) {
  if (v == null || v === 0) return '';
  const a = Math.min(Math.abs(v) / scale, 1) * 0.55 + 0.08;
  return v > 0 ? `rgba(143,191,154,${a.toFixed(2)})` : `rgba(217,140,122,${a.toFixed(2)})`;
}
function wordsEl(spans, scale, fmt) {
  const p = el('p', { class: 'words' });
  for (const [txt, v] of spans) {
    if (v == null) { p.append(txt); continue; }
    p.append(el('span', { tabindex: v ? '0' : null, title: `${txt}: ${fmt(v)}`, 'aria-label': v ? `${txt}, ${fmt(v)}` : null, style: { background: shade(v, scale) }, text: txt }));
  }
  return p;
}
function card(key, r, body) {
  return el('div', { class: 'panel card', 'data-method': key }, [
    el('h3', { text: NAMES[key] }),
    el('div', { class: 'verdict' }, [el('span', { class: 'lab ' + r.label, 'data-label': r.label, text: LABEL[r.label] }), el('span', { class: 'note', text: r.detail })]),
    el('div', { class: 'meter', role: 'img', 'aria-label': `Positive probability ${Math.round(r.prob * 100)}%` }, [el('i', { style: { width: (r.prob * 100).toFixed(1) + '%' } }), el('b')]),
    body,
  ]);
}

let LEX, M, B, tfPipe = null;
const state = { tf: null };

async function analyze() {
  const text = $('#txt').value.trim();
  const host = $('#res'); host.innerHTML = '';
  if (!text) { host.append(el('p', { class: 'msg', text: 'Type some text first.' })); return; }
  const v = vader(text, LEX), n = B.nb(text), l = B.lr(text);
  host.append(
    card('vader', v, wordsEl(v.words.flatMap(([w, x], i) => (i ? [[' ', null]] : []).concat([[w, x || 0]])), 3, (x) => `valence ${x.toFixed(2)}`)),
    card('nb', n, [wordsEl(n.spans, 1.5, (x) => `log-ratio ${x >= 0 ? '+' : ''}${x.toFixed(2)}`), n.known === 0 ? el('p', { class: 'why', text: 'No known words: falls back to the class prior.' }) : null]),
    card('lr', l, wordsEl(l.spans, 0.25, (x) => `weight ${x >= 0 ? '+' : ''}${x.toFixed(3)}`)),
  );
  const tfCard = el('div', { class: 'panel card', 'data-method': 'tf' }, [el('h3', { text: NAMES.tf }), el('p', { class: 'note', text: tfPipe ? 'Running…' : 'Not loaded. Use “Load DistilBERT” to compare a fine-tuned transformer (downloads once, then cached).' })]);
  host.append(tfCard);
  if (tfPipe) {
    const [r] = await tfPipe(text);
    const p = r.label === 'POSITIVE' ? r.score : 1 - r.score;
    const res = { label: p >= 0.5 ? 'pos' : 'neg', prob: p, detail: `${r.label.toLowerCase()} · confidence ${r.score.toFixed(3)}` };
    tfCard.replaceWith(card('tf', res, el('p', { class: 'why', text: 'Whole-sentence decision from self-attention; not a sum of word scores.' })));
  }
  await renderAgreement();
}

async function renderAgreement() {
  const t = $('#agree'); t.innerHTML = '';
  const cols = ['vader', 'nb', 'lr'].concat(tfPipe ? ['tf'] : []);
  t.append(el('thead', {}, el('tr', {}, [el('th', { text: 'Example' }), ...cols.map((c) => el('th', { text: NAMES[c].split(' (')[0] }))])));
  const tb = el('tbody'); t.append(tb);
  let disagree = 0;
  for (const [s] of SAMPLES) {
    const labs = [vader(s, LEX).label, B.nb(s).label, B.lr(s).label];
    if (tfPipe) { const [r] = await tfPipe(s); labs.push(r.label === 'POSITIVE' ? 'pos' : 'neg'); }
    if (new Set(labs.map((x) => (x === 'neu' ? 'neu' : x))).size > 1) disagree++;
    tb.append(el('tr', {}, [el('td', { text: s.length > 46 ? s.slice(0, 44) + '…' : s, title: s }), ...labs.map((x) => el('td', { class: x, text: LABEL[x] }))]));
  }
  $('#agree-note').textContent = `The methods disagree on ${disagree} of ${SAMPLES.length} examples.` + (tfPipe ? '' : ' Load DistilBERT to add a fourth column.');
}

async function loadTf() {
  const btn = $('#tfbtn'); btn.disabled = true;
  try {
    const { pipe, progressUI } = await import('/lab/shared/hf.js');
    const ui = progressUI($('#tfload'), 'DistilBERT');
    tfPipe = await pipe('sentiment-analysis', 'Xenova/distilbert-base-uncased-finetuned-sst-2-english', { ui, label: 'DistilBERT', readyText: 'DistilBERT ready · runs locally' });
    window.__lab.tf = true;
    btn.textContent = 'DistilBERT loaded';
    await analyze();
  } catch (e) {
    btn.disabled = false; btn.textContent = 'Retry loading DistilBERT';
  }
}

async function init() {
  try {
    [LEX, M] = await Promise.all([getJSON('./data/vader.json'), getJSON('./data/models.json')]);
  } catch (e) { fail(e.message); return; }
  B = makeBow(M);
  const met = M.metrics;
  $('#acc').append(
    el('thead', {}, el('tr', {}, [el('th', { text: 'Method' }), el('th', { text: 'Accuracy' }), el('th', { text: 'Notes' })])),
    el('tbody', {}, [
      ['Logistic regression', met.lr_acc, `binary features, L2, C = ${M.lr.C}`],
      ['Naive Bayes', met.nb_acc, 'counts, add-one smoothing'],
      ['Lexicon + rules (VADER)', met.vader_acc, 'no training; compound ≥ 0 → positive'],
      ['Chance', 0.5, 'balanced classes'],
    ].map(([a, b, c]) => el('tr', {}, [el('td', { text: a }), el('td', { text: (b * 100).toFixed(1) + '%' }), el('td', { text: c })]))),
  );
  $('#samples').append(...SAMPLES.map(([s, why]) => el('button', { class: 'sample', type: 'button', onclick: () => { $('#txt').value = s; analyze(); } }, [el('q', { text: s }), el('small', { text: why })])));
  $('#go').addEventListener('click', analyze);
  $('#txt').addEventListener('keydown', (e) => { if (e.key === 'Enter' && (e.ctrlKey || e.metaKey)) analyze(); });
  $('#tfbtn').addEventListener('click', loadTf);
  $('#go').disabled = false; $('#tfbtn').disabled = false;
  $('#status').textContent = `Ready · ${met.vocab_size.toLocaleString()}-word vocabulary · ${Object.keys(LEX).length.toLocaleString()} lexicon entries. Ctrl+Enter analyzes.`;
  window.__sent = { vader: (t) => vader(t, LEX), nb: B.nb, lr: B.lr };
  await analyze();
  ready({ sentiment: true });
}
init();
