import { el, $, ready, fail, getJSON } from '/lab/shared/lab.js';
import { progressUI } from '/lab/shared/hf.js';
import { lexModels, distilbert, sigmoid } from '/lab/attribution/models.js';
import { segment, rebuild, rng } from '/lab/attribution/text.js';

// [sentence, intended label] — chosen so the two instant models classify them correctly before the attack.
const SAMPLES = [
  ['Great performances make this a truly enjoyable film.', 'pos'],
  ['The movie is boring, stupid and a waste of time.', 'neg'],
  ['A smart, funny and wonderfully acted drama.', 'pos'],
  ['The worst film of the year: bad acting and a ridiculous plot.', 'neg'],
];
let expected = null;
const STOP = new Set(`a about above after again against all am an and any are as at be because been before being below between both but by can did do does doing down during each few for from further had has have having he her here hers herself him himself his how i if in into is it its itself just me more most my myself no nor not now of off on once only or other our ours ourselves out over own same she should so some such than that the their theirs them themselves then there these they this those through to too under until up very was we were what when where which while who whom why will with you your yours yourself yourselves s t don didn doesn isn wasn weren won wouldn couldn shouldn`.split(/\s+/));

let lex, bert, W2V, LEX, SST;
const state = window.__adv = {};

// ---------- Word2Vec neighbours from the shipped model ----------
async function loadW2V() {
  const [voc, vb, sb] = await Promise.all([getJSON('/word2vec/model/vocab.json'), fetch('/word2vec/model/vectors.bin').then((r) => r.arrayBuffer()), fetch('/word2vec/model/scales.bin').then((r) => r.arrayBuffer())]);
  const q = new Int8Array(vb), sc = new Float32Array(sb), N = voc.words.length, d = voc.dim;
  const V = new Float32Array(N * d);
  for (let i = 0; i < N; i++) for (let j = 0; j < d; j++) V[i * d + j] = q[i * d + j] * sc[i];
  const idx = new Map(voc.words.map((w, i) => [w, i]));
  return {
    has: (w) => idx.has(w),
    neighbours(w, k, minCos) {
      const i = idx.get(w); if (i == null) return [];
      const top = [];
      for (let r = 0; r < N; r++) {
        if (r === i) continue;
        let s = 0; for (let j = 0; j < d; j++) s += V[i * d + j] * V[r * d + j];
        if (s < minCos) continue;
        if (top.length < k) { top.push([s, r]); top.sort((a, b) => b[0] - a[0]); } else if (s > top[k - 1][0]) { top[k - 1] = [s, r]; top.sort((a, b) => b[0] - a[0]); }
      }
      return top.map(([s, r]) => ({ w: voc.words[r], cos: s }));
    },
  };
}
const relation = (a, b) => (LEX.ant[a] && LEX.ant[a].includes(b) ? 'antonym' : LEX.syn[a] && LEX.syn[a].includes(b) ? 'synonym' : 'neither');
const posOK = (a, b) => { const pa = LEX.pos[a], pb = LEX.pos[b]; if (!pa) return true; if (!pb) return false; return [...pa].some((c) => pb.includes(c)); };
const matchCase = (orig, w) => (orig[0] === orig[0].toUpperCase() && orig[0] !== orig[0].toLowerCase() ? w[0].toUpperCase() + w.slice(1) : w);

async function scorerFor(kind) {
  if (kind === 'bert') { if (!bert) bert = await distilbert(progressUI($('#bertload'), 'DistilBERT')); return bert.score; }
  return kind === 'nb' ? lex.nb : lex.lr;
}

/** Greedy word-substitution attack. Returns details including query count. */
async function attack(text, kind, { k = 15, minCos = 0.5, ratio = 0.3, posFilter = true, signal } = {}) {
  const score = await scorerFor(kind);
  let queries = 0;
  const f = async (texts) => { if (signal && signal.aborted) throw new DOMException('Cancelled', 'AbortError'); queries += texts.length; return score(texts, { signal }); };
  const parts = segment(text).map((p) => ({ ...p }));
  const words = parts.map((p, i) => (p.word ? i : -1)).filter((i) => i >= 0);
  const [z0] = await f([text]);
  const y = z0 >= 0 ? 1 : -1, P = (z) => sigmoid(y * z);
  const p0 = P(z0);
  // importance by deletion
  const del = await f(words.map((i) => rebuild(parts, (j) => j !== i)));
  const order = words.map((i, n) => [i, p0 - P(del[n])]).sort((a, b) => b[1] - a[1]).map(([i]) => i);
  const budget = Math.max(1, Math.ceil(ratio * words.length));
  const subs = []; let cur = p0, curZ = z0, flipped = false;
  for (const i of order) {
    if (subs.length >= budget || flipped) break;
    const orig = parts[i].t, lw = orig.toLowerCase();
    if (STOP.has(lw) || lw.length < 2 || !/^[a-z]+$/.test(lw) || !W2V.has(lw)) continue;
    const cands = W2V.neighbours(lw, k, minCos).filter((c) => /^[a-z]+$/.test(c.w) && !STOP.has(c.w) && (!posFilter || posOK(lw, c.w)));
    if (!cands.length) continue;
    const texts = cands.map((c) => { const save = parts[i].t; parts[i].t = matchCase(orig, c.w); const s = rebuild(parts, () => true); parts[i].t = save; return s; });
    const zs = await f(texts);
    const ps = zs.map(P);
    const flips = cands.map((c, n) => [c, ps[n], zs[n]]).filter(([, p]) => p < 0.5);
    let pick;
    if (flips.length) { pick = flips.sort((a, b) => b[0].cos - a[0].cos)[0]; flipped = true; } // closest successful substitute
    else { const b = ps.indexOf(Math.min(...ps)); if (ps[b] >= cur) continue; pick = [cands[b], ps[b], zs[b]]; }
    parts[i].t = matchCase(orig, pick[0].w);
    subs.push({ i, from: orig, to: parts[i].t, cos: pick[0].cos, rel: relation(lw, pick[0].w), p: pick[1] });
    cur = pick[1]; curZ = pick[2];
  }
  return { text, adv: rebuild(parts, () => true), parts, subs, flipped, queries, words: words.length, p0, p1: cur, label0: y > 0 ? 'pos' : 'neg', z1: curZ };
}

function highlight(res) {
  const subIdx = new Map(res.subs.map((s) => [s.i, s]));
  const p = el('p');
  let first = true;
  res.parts.forEach((pt, i) => {
    if (!first && !/^[.,!?;:)'"]$/.test(pt.t)) p.append(' ');
    first = false;
    const s = subIdx.get(i);
    p.append(s ? el('mark', { class: 'sub ' + (s.rel === 'antonym' ? 'ant' : s.rel === 'synonym' ? 'syn' : ''), title: `${s.from} → ${s.to} (cos ${s.cos.toFixed(2)}, WordNet: ${s.rel})`, text: pt.t }) : pt.t);
  });
  return p;
}
const lab = (l) => el('span', { class: 'lab ' + l, text: l === 'pos' ? 'positive' : 'negative' });

function renderSingle(res) {
  $('#result').hidden = false;
  const after = res.flipped ? (res.label0 === 'pos' ? 'neg' : 'pos') : res.label0;
  $('#pair').replaceChildren(
    el('div', {}, [el('span', { class: 'plabel', text: 'Original' }), el('p', {}, [res.text, ' ', lab(res.label0), ` P = ${res.p0.toFixed(3)}`]),
      ...(res.expected && res.expected !== res.label0 ? [el('p', { class: 'msg', text: 'This model already gets the original sentence wrong, so “flipping” it would make it right. Try another model or example.' })] : [])]),
    el('div', {}, [el('span', { class: 'plabel', text: res.flipped ? 'Adversarial: label flipped' : 'Best attempt: label not flipped' }), highlight(res), el('p', {}, [lab(after), ` P(original label) = ${res.p1.toFixed(3)} · ${res.subs.length}/${res.words} words changed · ${res.queries} model queries`])]));
  $('#subs').replaceChildren(el('thead', {}, el('tr', {}, ['Original', 'Substitute', 'Cosine', 'WordNet relation', 'P(original label) after'].map((h) => el('th', { text: h })))),
    el('tbody', {}, res.subs.map((s) => el('tr', {}, [el('td', { text: s.from }), el('td', { text: s.to }), el('td', { text: s.cos.toFixed(3) }), el('td', { class: s.rel === 'synonym' ? 'good' : s.rel === 'antonym' ? 'bad' : 'mid', text: s.rel }), el('td', { text: s.p.toFixed(3) })]))));
}
const opts = () => ({ k: +$('#k').value, minCos: +$('#cos').value, ratio: +$('#ratio').value, posFilter: $('#posf').checked });

async function single() {
  $('#msg').hidden = true; $('#go').disabled = true;
  try {
    const text = $('#txt').value.trim(); if (!text) throw new Error('Type a sentence first.');
    $('#status').textContent = 'Attacking…';
    const t0 = performance.now();
    const res = await attack(text, $('#model').value, opts());
    res.expected = expected;
    state.last = res; renderSingle(res);
    $('#status').textContent = `Done in ${((performance.now() - t0) / 1000).toFixed(1)} s.`;
    window.__lab.attacks = (window.__lab.attacks || 0) + 1;
  } catch (e) { $('#msg').textContent = e.message || String(e); $('#msg').hidden = false; $('#status').textContent = ''; }
  finally { $('#go').disabled = false; }
}

// ---------- batch ----------
let bctrl = null;
async function batch() {
  if (bctrl) return;
  bctrl = new AbortController(); $('#batch').disabled = true; $('#bcancel').disabled = false;
  const kind = $('#model').value, n = +$('#bn').value, st = $('#bstatus');
  try {
    if (!SST) SST = await getJSON('/lab/calibration/data/sst2.json');
    const score = await scorerFor(kind);
    const r = rng(7), ids = SST.sentence.map((s, i) => i).filter((i) => { const m = SST.sentence[i].split(' ').length; return m >= 6 && m <= 20; });
    for (let i = ids.length - 1; i > 0; i--) { const j = Math.floor(r() * (i + 1)); [ids[i], ids[j]] = [ids[j], ids[i]]; }
    // keep sentences the model classifies correctly
    const pool = ids.slice(0, n * 3), zs = await score(pool.map((i) => SST.sentence[i]), { signal: bctrl.signal });
    const chosen = pool.filter((i, j) => (zs[j] >= 0 ? 1 : 0) === SST.label[i]).slice(0, n);
    const results = [];
    for (const [c, i] of chosen.entries()) {
      st.textContent = `Attacking ${c + 1}/${chosen.length}…`;
      results.push(await attack(SST.sentence[i], kind, { ...opts(), signal: bctrl.signal }));
    }
    const succ = results.filter((x) => x.flipped);
    const allSyn = succ.filter((x) => x.subs.every((s) => s.rel === 'synonym')).length;
    const anyAnt = succ.filter((x) => x.subs.some((s) => s.rel === 'antonym')).length;
    const mean = (a) => (a.length ? a.reduce((x, y) => x + y, 0) / a.length : 0);
    const B = { n: results.length, success: succ.length / Math.max(1, results.length), perturbed: mean(succ.map((x) => x.subs.length / x.words)), queries: mean(results.map((x) => x.queries)), allSyn: succ.length ? allSyn / succ.length : 0, anyAnt: succ.length ? anyAnt / succ.length : 0 };
    state.batch = { kind, ...B, results };
    $('#bbig').hidden = false;
    $('#bbig').replaceChildren(...[
      [(B.success * 100).toFixed(0) + '%', `attack success (${succ.length}/${B.n})`],
      [(B.perturbed * 100).toFixed(0) + '%', 'words changed per success'],
      [B.queries.toFixed(0), 'model queries per sentence'],
      [(B.allSyn * 100).toFixed(0) + '%', 'successes using only WordNet synonyms'],
      [(B.anyAnt * 100).toFixed(0) + '%', 'successes using an antonym'],
    ].map(([b, s]) => el('div', {}, [el('b', { text: b }), el('span', { text: s })])));
    $('#bex').replaceChildren(el('p', { class: 'note', text: 'Successful attacks (click to inspect):' }), ...succ.slice(0, 12).map((x) => el('button', { class: 'ex', type: 'button', onclick: () => { $('#txt').value = x.text; renderSingle(x); $('#result').scrollIntoView({ block: 'nearest' }); } }, [el('span', { class: 'o', text: x.text }), el('br'), highlight(x)])));
    st.textContent = `${NAME[kind]}: ${B.n} sentences attacked.`;
    window.__lab.batch = B;
  } catch (e) { st.textContent = e.name === 'AbortError' ? 'Cancelled.' : 'Error: ' + (e.message || e); }
  finally { bctrl = null; $('#batch').disabled = false; $('#bcancel').disabled = true; }
}
const NAME = { lr: 'Logistic regression', nb: 'Naive Bayes', bert: 'DistilBERT' };

// ---------- CheckList-style suite ----------
const NOUN = ['movie', 'film', 'show', 'plot', 'script', 'cast', 'soundtrack', 'ending', 'story', 'performance'];
const POSA = ['good', 'great', 'excellent', 'wonderful', 'brilliant', 'enjoyable', 'fantastic', 'beautiful', 'moving', 'clever'];
const NEGA = ['bad', 'terrible', 'awful', 'boring', 'dull', 'weak', 'poor', 'disappointing', 'horrible', 'tedious'];
const NAMES = ['Mary', 'John', 'Aisha', 'Wei', 'Carlos', 'Priya', 'Olga', 'Kwame', 'Emma', 'Hiroshi'];
const CITIES = ['Chicago', 'Lagos', 'Mumbai', 'Paris', 'Toronto', 'Seoul', 'Cairo', 'Lima'];
const cross = (a, b) => a.flatMap((x) => b.map((y) => [x, y]));
function suite() {
  const r = rng(2020), pickN = (arr, n) => { const a = arr.slice(); for (let i = a.length - 1; i > 0; i--) { const j = Math.floor(r() * (i + 1)); [a[i], a[j]] = [a[j], a[i]]; } return a.slice(0, n); };
  const typo = (s) => { const ws = s.split(' '); const cand = ws.map((w, i) => [w, i]).filter(([w]) => /^[a-z]{4,}$/i.test(w)); const [w, i] = cand[Math.floor(r() * cand.length)]; const k = 1 + Math.floor(r() * (w.length - 2)); ws[i] = w.slice(0, k) + w[k + 1] + w[k] + w.slice(k + 2); return ws.join(' '); };
  const T = [];
  const mft = (name, cap, tpl, adj, expect) => T.push({ name, cap, type: 'MFT', cases: pickN(cross(NOUN, adj), 30).map(([n, a]) => ({ t: tpl(n, a), expect })) });
  mft('Simple positive', 'Vocabulary', (n, a) => `The ${n} was ${a}.`, POSA, 1);
  mft('Simple negative', 'Vocabulary', (n, a) => `The ${n} was ${a}.`, NEGA, 0);
  mft('Intensified negative', 'Vocabulary', (n, a) => `The ${n} was really ${a}.`, NEGA, 0);
  mft('Negated positive', 'Negation', (n, a) => `The ${n} was not ${a}.`, POSA, 0);
  mft('Negated negative → positive', 'Negation', (n, a) => `The ${n} was not ${a} at all.`, NEGA, 1);
  mft('Negation at a distance', 'Negation', (n, a) => `I don't think the ${n} was ${a}.`, POSA, 0);
  const inv = (name, cap, base, swap) => T.push({ name, cap, type: 'INV', pairs: base.map((b) => [b.t, swap(b)]) });
  const nameBase = pickN(cross(NAMES, cross(NOUN, [...POSA, ...NEGA])), 30).map(([nm, [n, a]]) => ({ nm, n, a, t: `${nm} thought the ${n} was ${a}.` }));
  inv('Change person names', 'Robustness: names', nameBase, (b) => `${NAMES[(NAMES.indexOf(b.nm) + 3) % NAMES.length]} thought the ${b.n} was ${b.a}.`);
  const cityBase = pickN(cross(CITIES, cross(NOUN, [...POSA, ...NEGA])), 30).map(([c, [n, a]]) => ({ c, n, a, t: `The ${n} I saw in ${c} was ${a}.` }));
  inv('Change locations', 'Robustness: locations', cityBase, (b) => `The ${b.n} I saw in ${CITIES[(CITIES.indexOf(b.c) + 3) % CITIES.length]} was ${b.a}.`);
  const sst = pickN(SST.sentence.filter((s) => s.split(' ').length >= 6 && s.split(' ').length <= 20), 30);
  inv('Add one typo', 'Robustness: typos', sst.map((t) => ({ t })), (b) => typo(b.t));
  const dir = (name, cap, add, sign) => T.push({ name, cap, type: 'DIR', sign, pairs: sst.map((t) => [t, `${t} ${add}`]) });
  dir('Add a positive phrase', 'Directional', 'I really enjoyed it .', 1);
  dir('Add a negative phrase', 'Directional', 'I hated it .', -1);
  return T;
}
const DIR_TOL = 0.1;
async function runSuite(T, kind, signal) {
  const score = await scorerFor(kind);
  const out = [];
  for (const t of T) {
    if (signal && signal.aborted) throw new DOMException('Cancelled', 'AbortError');
    let fails = [];
    if (t.type === 'MFT') {
      const z = await score(t.cases.map((c) => c.t), { signal });
      fails = t.cases.map((c, i) => ({ c, z: z[i] })).filter((x) => (x.z >= 0 ? 1 : 0) !== x.c.expect).map((x) => `${x.c.t} → P(pos) ${sigmoid(x.z).toFixed(2)}, expected ${x.c.expect ? 'positive' : 'negative'}`);
      out.push({ n: t.cases.length, fails });
    } else {
      const z = await score(t.pairs.flat(), { signal });
      const n = t.pairs.length;
      for (let i = 0; i < n; i++) {
        const a = z[2 * i], b = z[2 * i + 1], pa = sigmoid(a), pb = sigmoid(b);
        const bad = t.type === 'INV' ? (a >= 0) !== (b >= 0) : t.sign * (pb - pa) < -DIR_TOL;
        if (bad) fails.push(`${t.pairs[i][0]} (P(pos) ${pa.toFixed(2)}) → ${t.pairs[i][1]} (P(pos) ${pb.toFixed(2)})`);
      }
      out.push({ n, fails });
    }
  }
  return out;
}
let SUITE = null; const SR = {};
function renderSuite() {
  const kinds = Object.keys(SR);
  const tb = el('tbody');
  SUITE.forEach((t, i) => {
    const tr = el('tr', { class: 'clickable', tabindex: '0', 'data-test': t.name }, [el('td', { text: t.type }), el('td', { text: t.cap }), el('td', { text: t.name }),
      ...kinds.map((k) => { const r = SR[k][i], rate = r.fails.length / r.n; return el('td', { class: rate === 0 ? 'good' : rate < 0.2 ? 'mid' : 'bad', text: `${(rate * 100).toFixed(0)}% (${r.fails.length}/${r.n})` }); })]);
    const show = () => {
      [...tb.children].forEach((x) => x.classList.remove('sel')); tr.classList.add('sel');
      $('#clfail').replaceChildren(...kinds.map((k) => el('div', { class: 'stack' }, [el('h3', { text: `${NAME[k]} · ${t.name}: ${SR[k][i].fails.length} failures` }), ...SR[k][i].fails.slice(0, 6).map((f) => el('p', { class: 'note', text: f }))])));
    };
    tr.addEventListener('click', show); tr.addEventListener('keydown', (e) => { if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); show(); } });
    tb.append(tr);
  });
  $('#cltbl').replaceChildren(el('thead', {}, el('tr', {}, ['Type', 'Capability', 'Test', ...kinds.map((k) => NAME[k] + ' failure rate')].map((h) => el('th', { text: h })))), tb);
  state.suite = Object.fromEntries(kinds.map((k) => [k, SUITE.map((t, i) => ({ name: t.name, rate: SR[k][i].fails.length / SR[k][i].n }))]));
}
async function checklist(kinds) {
  const st = $('#clstatus');
  try {
    if (!SST) SST = await getJSON('/lab/calibration/data/sst2.json');
    SUITE ||= suite();
    for (const k of kinds) { st.textContent = `Running ${SUITE.reduce((a, t) => a + (t.cases ? t.cases.length : t.pairs.length * 2), 0)} cases on ${NAME[k]}…`; SR[k] = await runSuite(SUITE, k); }
    renderSuite();
    st.textContent = `${SUITE.length} tests. Failure = wrong label (MFT), label change (INV), or P(positive) moving the wrong way by more than ${DIR_TOL} (DIR).`;
    window.__lab.suite = (window.__lab.suite || 0) + 1;
  } catch (e) { st.textContent = 'Error: ' + (e.message || e); }
}

async function init() {
  try {
    [lex, W2V, LEX] = await Promise.all([lexModels(), loadW2V(), getJSON('./data/lexicon.json')]);
    $('#txt').value = SAMPLES[0][0]; expected = SAMPLES[0][1];
    $('#txt').addEventListener('input', () => { expected = null; });
    $('#samples').append(...SAMPLES.map(([s, l], i) => el('button', { class: 'chip', type: 'button', title: s, text: `Example ${i + 1} (${l === 'pos' ? 'positive' : 'negative'})`, onclick: () => { $('#txt').value = s; expected = l; single(); } })));
    $('#go').addEventListener('click', single);
    $('#batch').addEventListener('click', batch);
    $('#bcancel').addEventListener('click', () => bctrl && bctrl.abort());
    $('#cl').addEventListener('click', () => checklist(['lr', 'nb']));
    $('#clbert').addEventListener('click', async () => { $('#clbert').disabled = true; await checklist([...new Set([...Object.keys(SR), 'bert'])]); $('#clbert').disabled = false; });
    $('#model').addEventListener('change', () => { if ($('#model').value === 'bert' && !bert) $('#status').textContent = 'DistilBERT downloads once (≈67 MB) on the first attack.'; });
    state.attack = attack; state.neighbours = (w, k = 10, c = 0.5) => W2V.neighbours(w, k, c); state.relation = relation;
    ready({ adversarial: true });
    single();
  } catch (e) { fail('Couldn’t load the models: ' + (e.message || e)); }
}
init();
