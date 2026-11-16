// Retrieval-augmented generation: dense / BM25 / hybrid retrieval over a fictional corpus, a small T5 generator
// (or extractive reader), and an NLI faithfulness check of every answer sentence against the retrieved passages.
import { el, $, ready, fail, getJSON } from '/lab/shared/lab.js';
import { pipe, progressUI, transformers as T } from '/lab/shared/hf.js';

window.__lab = window.__lab || {};
const S = { busy: false };
const NLI_MODEL = 'Xenova/nli-deberta-v3-xsmall';
const QA_MODEL = 'Xenova/distilbert-base-cased-distilled-squad';
const COL = { supported: '#8fbf9a', contradicted: '#d98c7a', unsupported: '#d8b66a' };
const EXTRA_Q = [
  { q: 'Who is the mayor of Vellbridge?', note: 'The answer is in a distractor passage; dense retrieval should still find it.' },
  { q: 'When was Vellmouth founded?', note: 'Try with distractors on: Vellbridge was founded in 1290.' },
  { q: 'What colour are the doors of the Tidehouse?', note: 'Not in the corpus: a faithful system should say it doesn’t know.' },
  { q: 'Which festival remembers the 1588 storm surge?', note: 'A paraphrase: BM25 has few words to match.' },
];

// ---------- retrieval ----------
const STOP = new Set('a an the and or of to in on at for from by with is are was were be been it its this that these those as into than then so not no do does did can could my i me you your we our they their he she his her them how what when why which who whom will would should about there here up down out over under also only very just'.split(' '));
const tokens = (s) => (String(s).toLowerCase().match(/[a-z0-9]+(?:[-'][a-z0-9]+)*/g) || []).map((t) => t.replace(/'s$/, '')).filter((t) => !STOP.has(t));
function buildBM25(docs) {
  const tf = docs.map((d) => { const m = new Map(); for (const t of tokens(d.text)) m.set(t, (m.get(t) || 0) + 1); return m; });
  const len = tf.map((m) => [...m.values()].reduce((a, b) => a + b, 0));
  const avg = len.reduce((a, b) => a + b, 0) / len.length;
  const df = new Map(); for (const m of tf) for (const t of m.keys()) df.set(t, (df.get(t) || 0) + 1);
  const N = docs.length, k1 = 1.5, b = 0.75;
  const idf = (t) => Math.log(1 + (N - (df.get(t) || 0) + 0.5) / ((df.get(t) || 0) + 0.5));
  return (q) => {
    const qt = [...new Set(tokens(q))];
    return tf.map((m, i) => { let s = 0; for (const t of qt) { const f = m.get(t); if (f) s += idf(t) * (f * (k1 + 1)) / (f + k1 * (1 - b + b * len[i] / avg)); } return s; });
  };
}
const ranked = (scores) => scores.map((s, i) => [i, s]).sort((a, b) => b[1] - a[1] || a[0] - b[0]);
function rrf(...rankings) {
  const sc = new Map();
  for (const r of rankings) r.forEach(([i], k) => sc.set(i, (sc.get(i) || 0) + 1 / (60 + k + 1)));
  return [...sc].sort((a, b) => b[1] - a[1] || a[0] - b[0]);
}
async function embed(q) {
  const fx = await pipe('feature-extraction', S.corpus.model, { ui: S.uiEmb, label: 'MiniLM embeddings (23 MB)', readyText: 'Retriever ready · all-MiniLM-L6-v2 (8-bit, cached)' });
  const t = await fx(q, { pooling: 'mean', normalize: true });
  return Float32Array.from(t.data);
}
const dense = (qv) => S.docs.map((_, i) => { let s = 0; const o = i * S.D; for (let d = 0; d < S.D; d++) s += S.E[o + d] * qv[d]; return s; });

/** Returns {list:[{i,score}], qv} ranked by the chosen retriever. */
async function retrieve(q, mode) {
  let qv = null, list;
  if (mode !== 'bm25') qv = await embed(q);
  if (mode === 'dense') list = ranked(dense(qv));
  else if (mode === 'bm25') list = ranked(S.bm25(q));
  else list = rrf(ranked(dense(qv)), ranked(S.bm25(q)));
  return { list: list.map(([i, score]) => ({ i, score })), qv };
}

// ---------- generation ----------
const PROMPT_HEAD = 'Answer the question using only the context below. If the context does not contain the answer, say "I don\'t know".';
function buildPrompt(q, passages) {
  if (!passages) return `Question: ${q}\nAnswer:`;
  return `${PROMPT_HEAD}\n\nContext:\n${passages.map((p) => `[${p.id}] ${p.text}`).join('\n')}\n\nQuestion: ${q}\nAnswer:`;
}
async function generate(genId, q, passages) {
  if (genId === 'extractive') {
    if (!passages) return { text: '(An extractive reader needs passages, so it cannot answer closed-book.)', prompt: '—', extractive: true, none: true };
    const qa = await pipe('question-answering', QA_MODEL, { ui: S.uiGen, label: 'DistilBERT-SQuAD (66 MB)', readyText: 'Reader ready · DistilBERT-SQuAD (8-bit, cached)' });
    const ctx = passages.map((p) => p.text).join(' ');
    const r = await qa(q, ctx);
    const ans = (Array.isArray(r) ? r[0] : r) || {};
    return { text: ans.answer || '', score: ans.score, prompt: `Question: ${q}\nContext (concatenated passages):\n${ctx}`, extractive: true };
  }
  const label = genId.split('/')[1];
  const g = await pipe('text2text-generation', genId, { ui: S.uiGen, label: `${label} (≈95 MB)`, readyText: `Generator ready · ${label} (8-bit, cached)` });
  const prompt = buildPrompt(q, passages);
  const r = await g(prompt, { max_new_tokens: 64 });
  return { text: (r[0] && r[0].generated_text || '').trim(), prompt };
}

// ---------- NLI verification ----------
let nliP = null;
function nli() {
  nliP ||= Promise.all([
    T.AutoTokenizer.from_pretrained(NLI_MODEL, { progress_callback: (p) => S.uiNli.onProgress(p) }),
    T.AutoModelForSequenceClassification.from_pretrained(NLI_MODEL, { dtype: 'q8', device: 'wasm', progress_callback: (p) => S.uiNli.onProgress(p) }),
  ]).then(([tok, model]) => {
    S.uiNli.done('Verifier ready · DeBERTa-v3-xsmall NLI (8-bit, cached)');
    const id2 = model.config.id2label, idx = (n) => Number(Object.keys(id2).find((k) => id2[k].toLowerCase() === n));
    return { tok, model, E: idx('entailment'), N: idx('neutral'), C: idx('contradiction') };
  }).catch((e) => { nliP = null; S.uiNli.error(e); throw e; });
  return nliP;
}
const softmax = (xs) => { const m = Math.max(...xs); const e = xs.map((x) => Math.exp(x - m)); const s = e.reduce((a, b) => a + b, 0); return e.map((x) => x / s); };
const IDK = /\b(i don'?t know|not (provided|mentioned|given|stated)|does not (contain|mention|say)|no information|cannot be determined|unknown)\b/i;
function claimsOf(answer, q) {
  const a = answer.trim();
  if (!a) return [];
  if (IDK.test(a)) return [{ text: a, idk: true }];
  const sents = a.split(/(?<=[.!?])\s+(?=[A-Z0-9"“])/).map((s) => s.trim()).filter(Boolean);
  return sents.map((s) => {
    // Very short answers ("1342", "Ilse Marrable") are not sentences; NLI needs a proposition, so prefix the question.
    const words = s.split(/\s+/).length;
    return { text: s, hyp: words < 4 ? `${q.replace(/\?\s*$/, '')}: ${s.replace(/\.$/, '')}.` : s };
  });
}
/** For each claim: best entailing passage, verdict. */
async function verify(claims, passages) {
  const out = [];
  if (!claims.length) return out;
  const { tok, model, E, C } = await nli();
  for (const c of claims) {
    if (c.idk) { out.push({ ...c, verdict: 'abstained' }); continue; }
    const enc = tok(passages.map((p) => p.text), { text_pair: new Array(passages.length).fill(c.hyp), padding: true, truncation: true });
    const o = await model(enc);
    const L = Array.from(o.logits.data), k = o.logits.dims[1];
    const rows = passages.map((p, i) => { const pr = softmax(L.slice(i * k, i * k + k)); return { p, pE: pr[E], pC: pr[C] }; });
    const best = rows.reduce((a, b) => (b.pE > a.pE ? b : a));
    const worst = rows.reduce((a, b) => (b.pC > a.pC ? b : a));
    const verdict = best.pE >= 0.5 ? 'supported' : worst.pC >= 0.5 ? 'contradicted' : 'unsupported';
    out.push({ ...c, verdict, best, worst });
  }
  return out;
}

// ---------- UI ----------
function setStage(s) { for (const n of document.querySelectorAll('#pipeline .st')) n.classList.toggle('on', n.dataset.s === s); }
function showMsg(t) { const m = $('#msg'); m.hidden = !t; m.textContent = t || ''; }
function cfg() { return { k: +$('#k').value, mode: $('#mode').value, gen: $('#gen').value, distract: $('#distract').checked, closed: $('#closed').checked }; }

/** Choose the passages that go into the prompt (top-k, plus forced distractors if asked). */
function contextFor(list, qv, c) {
  const top = list.slice(0, c.k).map(({ i, score }) => ({ ...S.docs[i], score, rank: list.findIndex((x) => x.i === i) + 1 }));
  if (c.distract) {
    const ids = new Set(top.map((p) => p.id));
    const dscore = qv ? dense(qv) : S.bm25($('#q').value);
    const cand = S.docs.map((d, i) => ({ d, i, s: dscore[i] })).filter((x) => x.d.topic.startsWith('Vellbridge') && !ids.has(x.d.id)).sort((a, b) => b.s - a.s).slice(0, 2);
    // Put distractors first: models are often swayed by early context.
    top.unshift(...cand.map((x) => ({ ...x.d, score: x.s, distractor: true })));
  }
  return top;
}

function renderPassages(list, used) {
  const usedIds = new Set(used.map((p) => p.id));
  const show = [...used.filter((p) => p.distractor), ...list.slice(0, Math.max(8, cfg().k)).map(({ i, score }) => ({ ...S.docs[i], score }))];
  const seen = new Set();
  $('#passages').replaceChildren(...show.filter((p) => !seen.has(p.id) && seen.add(p.id)).map((p) => {
    const d = p.distractor || p.topic.startsWith('Vellbridge');
    return el('li', { class: (usedIds.has(p.id) ? 'used ' : '') + (d ? 'distractor' : '') }, [
      el('span', { class: 'pid', text: p.id }), el('span', { text: p.text }), el('span', { class: 'sc', text: p.score.toFixed(3) }),
      el('span', { class: 'meta', text: `${p.topic}${usedIds.has(p.id) ? ' · in prompt' : ''}${p.distractor ? ' · injected distractor' : ''}` }),
    ]);
  }));
  $('#ret-note').textContent = `Ranked by ${{ dense: 'cosine similarity (MiniLM)', bm25: 'BM25 score', hybrid: 'reciprocal-rank-fusion score' }[cfg().mode]}. Outlined passages went into the prompt; red ones are about Vellbridge.`;
}
function renderAnswer(host, vhost, gen, checks) {
  host.replaceChildren();
  if (!gen.text) { host.append(el('span', { class: 'note', text: '(empty answer)' })); vhost.replaceChildren(); return; }
  if (gen.none) { host.append(el('span', { class: 'note', text: gen.text })); vhost.replaceChildren(); return; }
  checks.forEach((c, i) => {
    if (i) host.append(' ');
    if (c.verdict === 'abstained') { host.append(el('span', { text: c.text })); return; }
    host.append(el('span', { class: 'claim', style: { '--c': COL[c.verdict] }, text: c.text }));
    if (c.verdict === 'supported') host.append(el('span', { class: 'cite', text: `[${c.best.p.id}]` }));
  });
  vhost.replaceChildren(...checks.map((c) => {
    if (c.verdict === 'abstained') return el('li', { style: { '--c': '#9aa3ad' } }, [el('b', { text: 'Abstained. ' }), 'The model says the context doesn’t contain the answer. That is the right behaviour when retrieval misses.']);
    const txt = c.verdict === 'supported'
      ? [el('b', { text: 'Supported ' }), `by ${c.best.p.id} (entailment ${c.best.pE.toFixed(2)}): “${c.best.p.text}”`]
      : c.verdict === 'contradicted'
        ? [el('b', { text: 'Contradicted ' }), `by ${c.worst.p.id} (contradiction ${c.worst.pC.toFixed(2)}): “${c.worst.p.text}”`]
        : [el('b', { text: 'Not supported. ' }), `No passage entails it (best: ${c.best.p.id}, entailment ${c.best.pE.toFixed(2)}).`];
    return el('li', { style: { '--c': COL[c.verdict] } }, [el('span', { class: 'note', text: `“${c.text}” ` }), el('br'), ...txt]);
  }));
}

/** One full RAG pass; returns everything for display / evaluation. */
async function ragOnce(q, c) {
  setStage('ret');
  const { list, qv } = await retrieve(q, c.mode);
  const used = contextFor(list, qv, c);
  setStage('gen');
  const gen = await generate(c.gen, q, used);
  const cb = c.closed ? await generate(c.gen, q, null) : null;
  setStage('nli');
  const checks = await verify(claimsOf(gen.text, q), used);
  const cbChecks = cb && !cb.none ? await verify(claimsOf(cb.text, q), used) : [];
  setStage(null);
  return { list, used, gen, checks, cb, cbChecks };
}

async function run() {
  if (S.busy) return;
  const q = $('#q').value.trim();
  if (!q) return void showMsg('Type a question first.');
  showMsg(''); S.busy = true; $('#run').disabled = $('#eval').disabled = true;
  const c = cfg(), t0 = performance.now();
  $('#status').textContent = 'Running…';
  try {
    const r = await ragOnce(q, c);
    renderPassages(r.list, r.used);
    $('#prompt').textContent = r.gen.prompt;
    renderAnswer($('#answer'), $('#verdicts'), r.gen, r.checks);
    $('#cbwrap').hidden = !r.cb;
    if (r.cb) {
      renderAnswer($('#cb-answer'), $('#cb-verdicts'), r.cb, r.cbChecks);
      const sup = (xs) => xs.filter((x) => x.verdict === 'supported').length;
      $('#cb-why').textContent = r.cb.none
        ? 'An extractive reader can only copy a span from passages, so without retrieval it has nothing to say: it cannot hallucinate, but it cannot answer either.'
        : `Closed-book, the model only has its weights, which have never seen Vellmouth, so any specific name, date or number it gives is invented. The closed-book answer is checked against the same retrieved passages: ${sup(r.cbChecks)} of ${r.cbChecks.filter((x) => x.verdict !== 'abstained').length} claims supported, against ${sup(r.checks)} of ${r.checks.filter((x) => x.verdict !== 'abstained').length} with retrieval.`;
    }
    const faith = r.checks.filter((x) => x.verdict !== 'abstained');
    window.__lab.last = { q, answer: r.gen.text, used: r.used.map((p) => p.id), verdicts: r.checks.map((x) => x.verdict), closedBook: r.cb && r.cb.text, closedVerdicts: r.cbChecks.map((x) => x.verdict) };
    $('#status').textContent = `Done in ${((performance.now() - t0) / 1000).toFixed(1)} s · ${r.used.length} passages in the prompt · ${faith.filter((x) => x.verdict === 'supported').length}/${faith.length} answer claims supported.`;
  } catch (e) {
    showMsg('Something went wrong: ' + (e.message || e)); window.__lab.runError = String(e.message || e); setStage(null);
  } finally { S.busy = false; $('#run').disabled = $('#eval').disabled = false; }
}

const norm = (s) => String(s).toLowerCase().replace(/[^a-z0-9: ]+/g, ' ').replace(/\s+/g, ' ').trim();
const correct = (ans, gold) => {
  const a = norm(ans), g = norm(gold).replace(/^the /, '');
  if (a.includes(g)) return true;
  const gt = g.split(' ').filter((w) => w.length > 2); // multi-part answers: most content words present
  return gt.length > 2 && gt.filter((w) => a.includes(w)).length / gt.length >= 0.75;
};
async function evaluate() {
  if (S.busy) return;
  S.busy = true; $('#run').disabled = $('#eval').disabled = true;
  const c = { ...cfg(), closed: true };
  const rows = [], tb = $('#eval-table tbody');
  tb.replaceChildren();
  try {
    for (const [n, item] of S.corpus.queries.entries()) {
      $('#eval-status').textContent = `Question ${n + 1} of ${S.corpus.queries.length}…`;
      const r = await ragOnce(item.q, c);
      const gold = new Set(item.gold), hit = r.used.some((p) => gold.has(p.id));
      const claims = r.checks.filter((x) => x.verdict !== 'abstained');
      const row = { q: item.q, gold: item.answer, hit, ans: r.gen.text, ok: correct(r.gen.text, item.answer), faith: claims.length ? claims.filter((x) => x.verdict === 'supported').length / claims.length : null, cb: r.cb.text, cbOk: !r.cb.none && correct(r.cb.text, item.answer) };
      rows.push(row);
      tb.append(el('tr', {}, [
        el('td', { text: row.q }), el('td', { text: row.gold }), el('td', { class: row.hit ? 'ok' : 'bad', text: row.hit ? 'hit' : 'miss' }),
        el('td', { text: row.ans }), el('td', { class: row.ok ? 'ok' : 'bad', text: row.ok ? '✓' : '✗' }),
        el('td', { text: row.faith == null ? 'abstained' : Math.round(row.faith * 100) + '%' }),
        el('td', { text: row.cb }), el('td', { class: row.cbOk ? 'ok' : 'bad', text: row.cbOk ? '✓' : '✗' }),
      ]));
    }
    const mean = (xs) => xs.reduce((a, b) => a + b, 0) / (xs.length || 1);
    const fs = rows.filter((r) => r.faith != null).map((r) => r.faith);
    const summ = { recall: mean(rows.map((r) => +r.hit)), acc: mean(rows.map((r) => +r.ok)), faith: fs.length ? mean(fs) : 0, cbAcc: mean(rows.map((r) => +r.cbOk)) };
    $('#eval-summary').replaceChildren(...[[`${Math.round(summ.recall * 100)}%`, `recall@${c.k}`], [`${Math.round(summ.acc * 100)}%`, 'RAG answers correct'], [`${Math.round(summ.faith * 100)}%`, 'answer claims supported'], [`${Math.round(summ.cbAcc * 100)}%`, 'closed-book answers correct']].map(([b, s]) => el('div', {}, [el('b', { text: b }), el('span', { text: s })])));
    window.__lab.eval = { ...summ, k: c.k, mode: c.mode, gen: c.gen, distract: c.distract, rows };
    $('#eval-status').textContent = `Done · ${c.mode}, k = ${c.k}, ${c.gen.split('/').pop()}${c.distract ? ', with distractors' : ''}.`;
  } catch (e) {
    $('#eval-status').textContent = 'Evaluation failed: ' + (e.message || e); window.__lab.runError = String(e.message || e);
  } finally { S.busy = false; $('#run').disabled = $('#eval').disabled = false; setStage(null); }
}

async function init() {
  S.corpus = await getJSON('/lab/rag/data/corpus.json');
  S.docs = S.corpus.docs;
  const [eb, sb] = await Promise.all(['emb.bin', 'emb-scales.bin'].map((f) => fetch('/lab/rag/data/' + f).then((r) => { if (!r.ok) throw new Error('Couldn’t load ' + f); return r.arrayBuffer(); })));
  const q8 = new Int8Array(eb), sc = new Float32Array(sb);
  S.D = q8.length / sc.length; S.E = new Float32Array(q8.length);
  for (let i = 0; i < sc.length; i++) for (let d = 0; d < S.D; d++) S.E[i * S.D + d] = q8[i * S.D + d] * sc[i];
  S.bm25 = buildBM25(S.docs);
  S.uiEmb = progressUI($('#load-emb'), 'MiniLM'); S.uiEmb.set('Retriever: all-MiniLM-L6-v2 · ≈23 MB, loads on first question', 0);
  S.uiGen = progressUI($('#load-gen'), 'generator'); S.uiGen.set('Generator: LaMini-Flan-T5-77M · ≈95 MB, loads on first question', 0);
  S.uiNli = progressUI($('#load-nli'), 'NLI verifier'); S.uiNli.set('Verifier: DeBERTa-v3-xsmall NLI · ≈87 MB, loads on first question', 0);

  const chips = [...S.corpus.queries.slice(0, 6).map((x) => ({ q: x.q })), ...EXTRA_Q];
  $('#samples').replaceChildren(...chips.map((x) => el('button', { class: 'chip', type: 'button', text: x.q, title: x.note || '', onclick: () => { $('#q').value = x.q; if (x.q.startsWith('When was Vellmouth')) $('#distract').checked = true; run(); } })));
  $('#k').addEventListener('input', () => { $('#k-val').textContent = $('#k').value; });
  $('#run').addEventListener('click', run);
  $('#q').addEventListener('keydown', (e) => { if (e.key === 'Enter') run(); });
  $('#eval').addEventListener('click', evaluate);
  window.__lab.retrieve = async (q, mode = 'dense', k = 3) => (await retrieve(q, mode)).list.slice(0, k).map(({ i }) => S.docs[i].id);
  window.__lab.run = run; window.__lab.evaluate = evaluate;
  ready({ passages: S.docs.length });
}
init().catch((e) => fail(e));
