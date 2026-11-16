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

export const WORKBENCH = { 4: translator, 5: fillMask, 6: association };

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
@media (max-width:560px){.wb-form{flex-wrap:wrap}.wb-form .btn{width:100%}}
`;
