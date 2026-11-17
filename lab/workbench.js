// Live mini-applications shown inside each tier on the NLP Lab hub (/lab/).
// Each entry renders a small working tool into a host element; heavy models load only when the user asks.
import { el, $ } from '/lab/shared/lab.js';

const out = (host) => host.querySelector('.wb-out');
const busy = (btn, on, label) => { btn.disabled = on; if (label) btn.textContent = label; };
const note = (text) => el('p', { class: 'note', text });

function shell({ title, blurb, href, controls, run, cta = 'Run', modelNote }) {
  const btn = el('button', { class: 'btn btn-primary', type: 'submit', text: cta });
  const form = el('form', { class: 'row wb-form', autocomplete: 'off' }, [...controls, btn]);
  const host = el('div', { class: 'panel pad stack wb' }, [
    el('div', { class: 'row', style: { justifyContent: 'space-between' } }, [
      el('h3', { class: 'wb-title', text: title }),
      el('a', { href, class: 'wb-link', text: 'Open the full module →' }),
    ]),
    el('p', { class: 'note', text: blurb }),
    modelNote ? el('p', { class: 'note wb-model', text: modelNote }) : null,
    form,
    el('div', { class: 'wb-out', 'aria-live': 'polite' }),
  ]);
  form.addEventListener('submit', async (e) => {
    e.preventDefault();
    const o = out(host); const label = btn.textContent;
    busy(btn, true, 'Working…');
    try { await run(o, btn); } catch (err) { o.replaceChildren(el('p', { class: 'msg', role: 'alert', text: String(err && err.message || err) })); }
    finally { busy(btn, false, label); }
  });
  return host;
}

const input = (attrs) => el('input', { type: 'text', class: 'wb-in', spellcheck: 'false', ...attrs });

// ---- Tier 1: tokenizer & counter (shares Text Studio's tokenizer) ---------------
function tokenizer() {
  const i = input({ value: 'Dr. Ada isn’t here—she’s at the U.S. Patent Office until 5:30 p.m. #history', 'aria-label': 'Text to tokenize' });
  return shell({
    title: 'Tokenizer & word counter', href: '/lab/tokenize/',
    blurb: 'Splits your text the way the rest of the lab does: contractions into two tokens, abbreviations kept whole, sentences found without being fooled by “Dr.”',
    controls: [i], cta: 'Tokenize',
    async run(o) {
      const { wordTokenize, sentences } = await import('/studio/lib/tokenize.js');
      const text = i.value, toks = wordTokenize(text), words = toks.filter((x) => x.type === 'word' || x.type === 'clitic');
      const types = new Set(words.map((x) => x.text.toLowerCase()));
      o.replaceChildren(
        el('div', {}, toks.map((x) => el('span', { class: 'tok', title: x.type, 'data-type': x.type, text: x.text }))),
        el('p', { class: 'note', text: `${toks.length} tokens · ${words.length} words · ${types.size} distinct · ${sentences(text, toks).length} sentence(s). Hover a token to see its type.` }),
      );
    },
  });
}

// ---- Tier 2: spell checker (noisy-channel model, shared with Text Studio) ---------
function speller() {
  const i = input({ value: 'Teh rivr flowd thru the vallley at nite.', 'aria-label': 'Sentence to spell-check' });
  return shell({
    title: 'Spell checker', href: '/lab/spell/',
    blurb: 'Each unknown word gets candidates within two edits, ranked by how common the word is times how likely that typo is.',
    controls: [i], cta: 'Correct',
    async run(o) {
      const { loadSpell } = await import('/studio/lib/spell.js');
      const sp = await loadSpell();
      const parts = sp.correctSentence(i.value);
      const fixes = parts.filter((p) => p.fixed);
      o.replaceChildren(
        el('p', { class: 'wb-result' }, parts.map((p) => p.fixed ? el('mark', { class: 'wb-fix', title: 'was: ' + p.orig, text: p.fixed }) : p.nos ? el('span', { class: 'wb-nos', title: 'no candidate within two edits', text: p.orig }) : p.orig)),
        el('p', { class: 'note', text: fixes.length ? fixes.map((p) => `${p.orig} → ${p.fixed}`).join(' · ') : 'No corrections needed.' }),
      );
    },
  });
}

// ---- Tier 3: part-of-speech tagger (HMM + Viterbi, shared with Text Studio) --------
function tagger() {
  const i = input({ value: 'Time flies like an arrow; fruit flies like a banana.', 'aria-label': 'Sentence to tag' });
  return shell({
    title: 'Part-of-speech tagger', href: '/lab/pos/',
    blurb: 'A hidden Markov model trained on the Penn Treebank picks the most likely tag sequence for the whole sentence with Viterbi decoding.',
    controls: [i], cta: 'Tag',
    async run(o) {
      const { loadPOS, PENN } = await import('/studio/lib/pos.js');
      const pos = await loadPOS();
      const { words, tags } = pos.tagSentence(i.value);
      o.replaceChildren(el('div', { class: 'wb-tags' }, words.map((w, k) => el('span', { class: 'wb-tag', title: PENN[tags[k]] || tags[k] }, [el('span', { text: w }), el('b', { text: tags[k] })]))),
        el('p', { class: 'note', text: 'Hover a tag for its meaning. The classic trap: a person would tag the halves flies/VBZ like/IN and then flies/NNS like/VBZ. A bigram HMM sees only one tag of context, so it often gives both halves the same reading; the full module shows its trellis.' }));
    },
  });
}

// ---- Tier 4: translation ----------------------------------------------------
function translator() {
  const i = input({ value: 'The river carries the town’s stories to the sea.', 'aria-label': 'English sentence to translate' });
  return shell({
    title: 'English → French translator', href: '/lab/translation/',
    blurb: 'A MarianMT encoder–decoder trained on OPUS translates your sentence.',
    modelNote: 'First use downloads the model (about 107 MB); later runs are local.',
    controls: [i], cta: 'Translate',
    async run(o) {
      const { pipe, progressUI } = await import('/lab/shared/hf.js');
      const load = el('div'); o.replaceChildren(load);
      const tr = await pipe('translation', 'Xenova/opus-mt-en-fr', { ui: progressUI(load, 'MarianMT en→fr') });
      const [r] = await tr(i.value.trim() || 'Hello.');
      o.replaceChildren(el('p', { class: 'wb-result', lang: 'fr', text: r.translation_text }));
    },
  });
}

// ---- Tier 5: masked language model -------------------------------------------
function fillMask() {
  const i = input({ value: 'The capital of France is [MASK].', 'aria-label': 'Sentence with one [MASK]' });
  return shell({
    title: 'Fill in the blank', href: '/lab/contextual/',
    blurb: 'DistilBERT reads the whole sentence in both directions and ranks words for the [MASK].',
    modelNote: 'First use downloads the model (about 68 MB).',
    controls: [i], cta: 'Predict',
    async run(o) {
      let text = i.value.trim();
      if (!text.includes('[MASK]')) throw new Error('Put [MASK] where the missing word goes.');
      const { pipe, progressUI } = await import('/lab/shared/hf.js');
      const load = el('div'); o.replaceChildren(load);
      const fm = await pipe('fill-mask', 'Xenova/distilbert-base-uncased', { ui: progressUI(load, 'DistilBERT') });
      const rs = await fm(text, { top_k: 5 });
      o.replaceChildren(el('ol', { class: 'wb-list' }, rs.map((r) => el('li', {}, [el('b', { text: r.token_str }), ' ', el('span', { class: 'note', text: (r.score * 100).toFixed(1) + '%' })]))));
    },
  });
}

// ---- Tier 6: word association meter on the shipped Word2Vec --------------------
const SETS = {
  pleasant: 'caress freedom health love peace cheer friend heaven loyal pleasure diamond gentle honest lucky rainbow diploma gift honor miracle sunrise family happy laughter paradise vacation',
  unpleasant: 'abuse crash filth murder sickness accident death grief poison stink assault disaster hatred pollute tragedy divorce jail poverty ugly cancer kill rotten vomit agony prison',
  male: 'he him his man men boy son brother father uncle',
  female: 'she her hers woman women girl daughter sister mother aunt',
};
let w2v;
async function loadW2V() {
  if (w2v) return w2v;
  const [vocab, vec, sc] = await Promise.all([
    fetch('/word2vec/model/vocab.json').then((r) => r.json()),
    fetch('/word2vec/model/vectors.bin').then((r) => r.arrayBuffer()),
    fetch('/word2vec/model/scales.bin').then((r) => r.arrayBuffer()),
  ]);
  const q = new Int8Array(vec), s = new Float32Array(sc), dim = vocab.dim, idx = new Map(vocab.words.map((w, i) => [w, i]));
  const get = (w) => { const i = idx.get(w); if (i == null) return null; const v = new Float32Array(dim); let n = 0; for (let d = 0; d < dim; d++) { v[d] = q[i * dim + d] * s[i]; n += v[d] * v[d]; } n = Math.sqrt(n); for (let d = 0; d < dim; d++) v[d] /= n; return v; };
  const cos = (a, b) => { let x = 0; for (let d = 0; d < a.length; d++) x += a[d] * b[d]; return x; };
  w2v = { get, cos };
  return w2v;
}
function meter(label, a, b, value) {
  const pct = Math.max(-1, Math.min(1, value / 0.15)) * 50 + 50; // ±0.15 cosine maps to the ends
  return el('div', { class: 'wb-meter' }, [
    el('div', { class: 'row', style: { justifyContent: 'space-between' } }, [el('span', { text: a }), el('span', { class: 'note', text: label + ' ' + (value >= 0 ? '+' : '') + value.toFixed(3) }), el('span', { text: b })]),
    el('div', { class: 'wb-track' }, el('i', { style: { left: pct + '%' } })),
  ]);
}
function association() {
  const i = input({ value: 'nurse, engineer, flower, weapon', 'aria-label': 'Words to measure, separated by commas' });
  return shell({
    title: 'Word association meter', href: '/lab/bias/',
    blurb: 'How close each word sits to two opposing word sets in our Word2Vec space: the measurement behind WEAT bias tests.',
    controls: [i], cta: 'Measure',
    async run(o) {
      const { get, cos } = await loadW2V();
      const mean = (v, set) => { let s = 0, n = 0; for (const w of SETS[set].split(' ')) { const u = get(w); if (u) { s += cos(v, u); n++; } } return s / n; };
      const words = i.value.toLowerCase().split(',').map((w) => w.trim()).filter(Boolean).slice(0, 6);
      const rows = words.map((w) => {
        const v = get(w);
        if (!v) return el('p', { class: 'note', text: `“${w}” isn’t in the 30,000-word vocabulary.` });
        return el('div', { class: 'stack', style: { gap: '6px' } }, [el('b', { text: w }),
          meter('valence', 'unpleasant', 'pleasant', mean(v, 'pleasant') - mean(v, 'unpleasant')),
          meter('gender', 'male terms', 'female terms', mean(v, 'female') - mean(v, 'male'))]);
      });
      o.replaceChildren(...rows, note('Trained on 17M words of Wikipedia, so scores are noisy; the full module adds effect sizes and permutation tests.'));
    },
  });
}

// ---- Tier 7: reward over-optimization (exact, from the preference module's core) ---
function overoptimization() {
  const i = el('input', { type: 'number', class: 'wb-in', value: '1.5', min: '0.05', max: '20', step: '0.05', 'aria-label': 'KL penalty β' });
  return shell({
    title: 'Reward over-optimization explorer', href: '/lab/preference/',
    blurb: 'Annotators who favour long, flattering answers produce a biased reward. Optimize a policy against it with KL penalty β (π* ∝ π_ref·exp(r/β)) and compare the reward it chases with the true quality it delivers. Try β = 20, 1.5 and 0.1.',
    controls: [el('label', { class: 'note', for: 'wb-beta', text: 'β' }), Object.assign(i, { id: 'wb-beta' })], cta: 'Optimize',
    async run(o) {
      const C = await import('/lab/preference/core.js');
      const beta = Math.max(0.05, +i.value || 1.5);
      const ref = C.sftRef(), proxy = (c) => C.annotScore(c);
      const pi = C.optimalPolicy(ref, proxy, beta);
      const row = (label, a, b) => el('tr', {}, [el('td', { text: label }), el('td', { text: a.toFixed(2) }), el('td', { text: b.toFixed(2) })]);
      const top = C.PROMPTS[0].c[pi[0].indexOf(Math.max(...pi[0]))];
      o.replaceChildren(
        el('table', { class: 'data' }, [el('thead', {}, el('tr', {}, [el('th', { text: '' }), el('th', { text: 'reference policy' }), el('th', { text: `optimized, β = ${beta}` })])),
          el('tbody', {}, [row('proxy reward it chases', C.expect(ref, proxy), C.expect(pi, proxy)), row('true quality (gold)', C.expect(ref, C.gold), C.expect(pi, C.gold)), row('KL from reference (nats)', 0, C.kl(pi, ref))])]),
        el('p', { class: 'note', text: `Most likely answer to “${C.PROMPTS[0].q}”: “${top.t.slice(0, 120)}${top.t.length > 120 ? '…' : ''}”` }),
        note('Exact at this toy scale: 4 prompts × 6 written answers, simulated annotators. The full module fits the reward model from preferences and runs RLHF and DPO.'),
      );
    },
  });
}

// ---- Tier 8: compute-optimal model size from published scaling-law fits ------------
function chinchilla() {
  const i = el('input', { type: 'text', class: 'wb-in', value: '5.76e23', 'aria-label': 'Training compute in FLOPs' });
  return shell({
    title: 'Compute-optimal model calculator', href: '/lab/scaling/',
    blurb: 'Enter a training budget in FLOPs. Using L(N, D) = E + A/N^α + B/D^β and C ≈ 6ND, this returns the parameter and token counts that minimize predicted loss. 5.76e23 is about what Chinchilla used.',
    controls: [i], cta: 'Allocate',
    async run(o) {
      const { FITS, optimal } = await import('/lab/scaling/laws.js');
      const C = Number(i.value.replace(/[, ]/g, ''));
      if (!(C > 1e15 && C < 1e32)) throw new Error('Enter a budget between 1e15 and 1e32 FLOPs, e.g. 5.76e23.');
      const fmt = (x) => x >= 1e12 ? (x / 1e12).toFixed(2) + ' T' : x >= 1e9 ? (x / 1e9).toFixed(1) + ' B' : (x / 1e6).toFixed(0) + ' M';
      const rows = Object.values(FITS).map((f) => { const r = optimal(f, C); return el('tr', {}, [el('td', { text: f.label }), el('td', { text: fmt(r.N) }), el('td', { text: fmt(r.D) }), el('td', { text: r.ratio.toFixed(1) }), el('td', { text: r.L.toFixed(3) })]); });
      o.replaceChildren(el('div', { class: 'scroll-x' }, el('table', { class: 'data' }, [el('thead', {}, el('tr', {}, ['fit', 'parameters N*', 'tokens D*', 'tokens / param', 'predicted loss'].map((h) => el('th', { text: h })))), el('tbody', {}, rows)])),
        note('The two fits disagree: the replication (Besiroglu et al. 2024) puts the optimum near 20 tokens per parameter, the original Approach 3 fit much higher. Loss is in nats per token on the paper’s data, so it isn’t comparable across datasets.'));
    },
  });
}

export const WORKBENCH = { 1: tokenizer, 2: speller, 3: tagger, 4: translator, 5: fillMask, 6: association, 7: overoptimization, 8: chinchilla };

export const WORKBENCH_CSS = `
.wb{margin-bottom:18px}
.wb-title{font-size:24px}
.wb-link{font-size:14px;text-decoration:none}
.wb-form{flex-wrap:nowrap}
.wb-in{flex:1 1 0;min-width:0;min-height:48px;background:var(--bg);color:var(--text);border:1px solid var(--divider);border-radius:0;padding:0 14px;font:500 16px var(--body)}
.wb-in:focus{border-color:var(--accent);outline:none;box-shadow:0 0 0 1px var(--accent)}
.wb-out:empty{display:none}
.wb-result{font-size:20px;margin:0;color:var(--accent-200)}
.wb-list{margin:0;padding-left:22px;display:flex;flex-direction:column;gap:4px}
.wb-meter{display:flex;flex-direction:column;gap:4px;font-size:13px;color:var(--muted)}
.wb-track{position:relative;height:6px;background:linear-gradient(90deg,#d98c7a55,var(--divider) 50%,#7ea3c855)}
.wb-track i{position:absolute;top:-4px;width:3px;height:14px;background:var(--accent-200);transform:translateX(-50%)}
.tok[data-type="clitic"]{border-color:var(--accent-500)}.tok[data-type="punct"]{color:var(--faint)}.tok[data-type="url"],.tok[data-type="email"],.tok[data-type="hashtag"],.tok[data-type="mention"]{border-color:#d8b66a;color:#d8b66a}.tok[data-type="number"],.tok[data-type="time"],.tok[data-type="date"]{border-color:#8fbf9a;color:#8fbf9a}
.wb-fix{background:color-mix(in srgb,#8fbf9a 22%,transparent);color:var(--text);padding:0 3px}.wb-nos{text-decoration:underline wavy #d98c7a}
.wb-tags{display:flex;flex-wrap:wrap;gap:6px}.wb-tag{display:inline-flex;flex-direction:column;align-items:center;border:1px solid var(--divider);padding:4px 8px;font-size:15px}.wb-tag b{font-size:11px;letter-spacing:.06em;color:var(--accent-300)}
@media (max-width:560px){.wb-form{flex-wrap:wrap}.wb-form .btn{width:100%}}
`;
