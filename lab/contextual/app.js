// Contextual embeddings & masked LM: DistilBERT fill-mask; MiniLM token vectors vs a static Word2Vec vector.
import { el, $, ready, plotly, plotLayout, PLOT_CONFIG, getJSON } from '/lab/shared/lab.js';
import { pipe, loadModel, progressUI, softmax } from '/lab/shared/hf.js';

window.__lab = window.__lab || {};
const MLM = 'Xenova/distilbert-base-uncased';
const ENC = 'Xenova/all-MiniLM-L6-v2';

// ---------- masked LM ----------
const MLM_EX = [
  'She walked to the [MASK] to deposit her paycheck.',
  'They had a picnic on the grassy [MASK] of the river.',
  'The capital of France is [MASK].',
  'The nurse said that [MASK] would be late.',
  'The engineer said that [MASK] would be late.',
  'I [MASK] coffee every [MASK].',
];
let mlmUI;
async function predict() {
  const text = $('#mlmin').value.trim(), msg = $('#mlmmsg');
  msg.hidden = true;
  const n = (text.match(/\[MASK\]/g) || []).length;
  if (!n) { msg.hidden = false; msg.textContent = 'Add at least one [MASK] token (in capitals, with square brackets).'; return; }
  if (n > 4) { msg.hidden = false; msg.textContent = 'Use at most four [MASK] tokens.'; return; }
  $('#mlmgo').disabled = true;
  try {
    // transformers.js' fill-mask pipeline only fills the first [MASK], so read the logits directly:
    // softmax over the vocabulary at every mask position (each mask predicted with the others still masked).
    const { tokenizer: tok, model } = await loadModel(MLM, 'masked', { ui: mlmUI, label: 'DistilBERT (68 MB)', readyText: 'Ready · distilbert-base-uncased (q8, 68 MB, cached)' });
    const enc = tok(text);
    const ids = Array.from(enc.input_ids.data, Number), maskId = tok.mask_token_id;
    const { logits } = await model(enc);
    const V = logits.dims[2];
    const out = [];
    ids.forEach((id, pos) => {
      if (id !== maskId) return;
      const pr = softmax(logits.data.subarray(pos * V, (pos + 1) * V));
      const top = [];
      for (let i = 0; i < V; i++) { if (top.length < 10 || pr[i] > top[top.length - 1][1]) { top.push([i, pr[i]]); top.sort((a, b) => b[1] - a[1]); if (top.length > 10) top.pop(); } }
      out.push(top.map(([i, s]) => ({ token_str: tok.decode([i]).trim(), score: s })));
    });
    $('#preds').replaceChildren(...out.map((preds, k) => el('div', { class: 'panel pred' }, [
      el('h3', { text: n > 1 ? `[MASK] #${k + 1}` : '[MASK]' }),
      el('span', { class: 'note', text: n > 1 ? 'Predicted independently: other masks stay masked.' : 'Top 10 tokens by probability.' }),
      el('ol', {}, preds.map((p) => el('li', {}, [
        el('div', { class: 'w' }, [el('span', { text: p.token_str }), el('div', { class: 'track' }, el('i', { style: { width: (p.score / preds[0].score * 100).toFixed(1) + '%' } }))]),
        el('span', { class: 'p', text: (p.score * 100).toFixed(1) + '%' }),
      ]))),
    ])));
    window.__lab.mlm = { text, preds: out.map((ps) => ps.map((p) => [p.token_str, p.score])) };
  } catch (e) { msg.hidden = false; msg.textContent = 'Prediction failed: ' + e.message; }
  finally { $('#mlmgo').disabled = false; }
}

// ---------- contextual vs static ----------
const CTX_EX = {
  bank: ['I sat on the bank of the river and watched the boats.', 'Erosion slowly wore away the muddy bank of the stream.', 'The bank approved her loan on Friday.', 'He withdrew cash from the bank before the trip.', 'You can bank on her to finish the job.'],
  play: ['The children play in the park after school.', 'We watched a play by Shakespeare at the theater.', 'The actors rehearsed the play for weeks.', 'Press play to start the video.', 'Dogs love to play fetch.'],
  bat: ['The bat flew out of the cave at dusk.', 'A small bat hung upside down in the barn.', 'He swung the bat and hit a home run.', 'She bought a new cricket bat.'],
  spring: ['Flowers bloom in the spring.', 'The air is warm in late spring.', 'A coiled spring pushed the lid open.', 'They drank cold water from a mountain spring.'],
};
let encUI;
const W2V = {};
async function loadW2V() {
  if (W2V.ready) return;
  const [vocab, vb, sb] = await Promise.all([
    getJSON('/word2vec/model/vocab.json'),
    fetch('/word2vec/model/vectors.bin').then((r) => r.arrayBuffer()),
    fetch('/word2vec/model/scales.bin').then((r) => r.arrayBuffer()),
  ]);
  Object.assign(W2V, { words: vocab.words, index: new Map(vocab.words.map((w, i) => [w, i])), q: new Int8Array(vb), sc: new Float32Array(sb), D: vocab.dim, ready: true });
}
function w2vNeighbors(word, k = 10) {
  const i = W2V.index.get(word); if (i == null) return null;
  const D = W2V.D, q = W2V.q, sc = W2V.sc, N = W2V.words.length;
  const v = new Float32Array(D); for (let d = 0; d < D; d++) v[d] = q[i * D + d] * sc[i];
  const out = [];
  for (let j = 0; j < N; j++) { if (j === i) continue; let s = 0; for (let d = 0; d < D; d++) s += q[j * D + d] * v[d]; s *= sc[j]; out.push([j, s]); }
  const vn = Math.hypot(...v);
  return out.sort((a, b) => b[1] - a[1]).slice(0, k).map(([j, s]) => [W2V.words[j], s / vn]);
}
function findSpan(ids, sub) {
  for (let i = 0; i + sub.length <= ids.length; i++) { let ok = true; for (let k = 0; k < sub.length; k++) if (ids[i + k] !== sub[k]) { ok = false; break; } if (ok) return i; }
  return -1;
}
async function compareContexts() {
  const target = $('#target').value.trim().toLowerCase(), msg = $('#ctxmsg');
  const sents = $('#sents').value.split('\n').map((s) => s.trim()).filter(Boolean);
  msg.hidden = true;
  if (!target || /\s/.test(target)) { msg.hidden = false; msg.textContent = 'Enter a single target word.'; return; }
  if (sents.length < 2) { msg.hidden = false; msg.textContent = 'Enter at least two sentences, one per line.'; return; }
  $('#ctxgo').disabled = true;
  try {
    const [fx] = await Promise.all([pipe('feature-extraction', ENC, { ui: encUI, label: 'MiniLM (23 MB)', readyText: 'Ready · all-MiniLM-L6-v2 (q8, 23 MB, cached)' }), loadW2V()]);
    const tok = fx.tokenizer;
    const sub = Array.from(tok(target, { add_special_tokens: false }).input_ids.data, Number);
    const vecs = [], used = [], skipped = [];
    for (const s of sents) {
      const ids = Array.from(tok(s).input_ids.data, Number);
      const at = findSpan(ids, sub);
      if (at < 0) { skipped.push(s); continue; }
      const t = await fx(s, { pooling: 'none' });
      const D = t.dims[2], v = new Float32Array(D);
      for (let k = 0; k < sub.length; k++) for (let d = 0; d < D; d++) v[d] += t.data[(at + k) * D + d] / sub.length;
      const n = Math.hypot(...v); vecs.push(v.map((x) => x / n)); used.push(s);
    }
    if (skipped.length) { msg.hidden = false; msg.textContent = `“${target}” not found as a whole word in: ${skipped.map((s) => '“' + s + '”').join(' ')}`; }
    if (used.length < 2) throw new Error(`Need at least two sentences containing “${target}”.`);
    const n = used.length, M = used.map((_, i) => used.map((_, j) => vecs[i].reduce((t, x, d) => t + x * vecs[j][d], 0)));
    const labels = used.map((s, i) => `S${i + 1}`);
    const P = await plotly();
    P.react($('#heat'), [{
      type: 'heatmap', z: M, x: labels, y: labels, zmin: Math.min(...M.flat()), zmax: 1,
      colorscale: [[0, '#121820'], [0.5, '#416180'], [1, '#d6ebff']],
      text: M.map((r) => r.map((v) => v.toFixed(2))), texttemplate: '%{text}', textfont: { size: 12 },
      customdata: used.map((a) => used.map((b) => [a, b])),
      hovertemplate: '%{customdata[0]}<br>vs<br>%{customdata[1]}<br><b>cos %{z:.3f}</b><extra></extra>',
      colorbar: { tickfont: { color: '#9aa3ad' }, outlinewidth: 0 },
    }], plotLayout({ margin: { l: 44, r: 16, t: 16, b: 40 }, yaxis: { autorange: 'reversed', color: '#9aa3ad' }, xaxis: { color: '#9aa3ad', side: 'bottom' } }), PLOT_CONFIG);
    // summary
    let lo = [1, 0, 0], hi = [-1, 0, 0];
    for (let i = 0; i < n; i++) for (let j = i + 1; j < n; j++) { if (M[i][j] < lo[0]) lo = [M[i][j], i, j]; if (M[i][j] > hi[0]) hi = [M[i][j], i, j]; }
    $('#ctxsum').replaceChildren(
      ...used.map((s, i) => el('span', {}, [el('b', { text: `S${i + 1} ` }), s])),
      el('span', {}, ['Most similar pair: ', el('b', { text: `S${hi[1] + 1}–S${hi[2] + 1} (${hi[0].toFixed(2)})` }), '. Least similar: ', el('b', { text: `S${lo[1] + 1}–S${lo[2] + 1} (${lo[0].toFixed(2)})` }), '.']),
    );
    const nn = w2vNeighbors(target);
    $('#static').replaceChildren(nn
      ? el('span', {}, [`Word2Vec has exactly one vector for “${target}”, so every cell of its heatmap would be 1.00. Its nearest neighbors blend all senses:`])
      : el('span', { text: `“${target}” is not in the 30,000-word Word2Vec vocabulary.` }),
      nn ? el('div', { class: 'nn' }, nn.map(([w, s]) => el('span', { class: 'tok', title: `cos ${s.toFixed(3)}`, text: w }))) : null);
    window.__lab.ctx = { target, n, M, w2v: nn ? nn.map(([w]) => w) : null };
  } catch (e) { msg.hidden = false; msg.textContent = e.message; }
  finally { $('#ctxgo').disabled = false; }
}

// ---------- init ----------
mlmUI = progressUI($('#mlmload'), 'DistilBERT');
mlmUI.set('The model (68 MB) downloads on your first prediction and is cached afterwards.', 0);
encUI = progressUI($('#ctxload'), 'MiniLM');
encUI.set('The encoder (23 MB) downloads on first use and is cached afterwards.', 0);
$('#mlmchips').append(...MLM_EX.map((t) => el('button', { class: 'chip', type: 'button', text: t, onclick: () => { $('#mlmin').value = t; predict(); } })));
$('#mlmform').addEventListener('submit', (e) => { e.preventDefault(); predict(); });
$('#sents').value = CTX_EX.bank.join('\n');
$('#ctxchips').append(...Object.entries(CTX_EX).map(([w, ss]) => el('button', { class: 'chip', type: 'button', text: w, onclick: () => { $('#target').value = w; $('#sents').value = ss.join('\n'); compareContexts(); } })));
$('#ctxgo').addEventListener('click', compareContexts);
ready({ module: 'contextual' });
