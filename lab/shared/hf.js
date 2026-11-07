// Shared loader for Hugging Face transformers.js (self-hosted v3.8.1).
// Models download from huggingface.co on first use and are cached by the browser (Cache API).
//
//   import { pipe, loadModel, progressUI } from '/lab/shared/hf.js';
//   const ui = progressUI(document.querySelector('#load'));        // optional status element
//   const clf = await pipe('sentiment-analysis', 'Xenova/distilbert-base-uncased-finetuned-sst-2-english', { ui });
//
// For lower-level access (logits, hidden states): const { tokenizer, model } = await loadModel('Xenova/gpt2', 'causal', { ui });
import * as T from '/assets/vendor/transformers-3.8.1/transformers.min.js';

export const transformers = T;
const { env, pipeline, AutoTokenizer, AutoModel, AutoModelForCausalLM, AutoModelForMaskedLM } = T;

env.allowLocalModels = false;
env.allowRemoteModels = true;
env.useBrowserCache = true;
env.backends.onnx.wasm.wasmPaths = '/assets/vendor/transformers-3.8.1/';
// Single-threaded WASM: no cross-origin-isolation headers needed on Netlify.
env.backends.onnx.wasm.numThreads = 1;

const cache = new Map();

/** Progress UI bound to an element containing (optionally) .bar-load i and a text span. */
export function progressUI(host, label = 'model') {
  if (!host) return null;
  host.classList.add('load');
  host.innerHTML = '<span class="bar-load"><i></i></span><span class="t"></span>';
  const bar = host.querySelector('i'), txt = host.querySelector('.t');
  const files = new Map();
  return {
    set(text, frac) { txt.textContent = text; if (frac != null) bar.style.width = Math.round(frac * 100) + '%'; },
    onProgress(p) {
      if (p.status === 'progress' && p.file) {
        files.set(p.file, { loaded: p.loaded || 0, total: p.total || 0 });
        let l = 0, t = 0; for (const f of files.values()) { l += f.loaded; t += f.total; }
        this.set(`Downloading ${label}… ${(l / 1e6).toFixed(1)} / ${(t / 1e6).toFixed(1)} MB`, t ? l / t : 0);
      } else if (p.status === 'ready') this.set(`Loading ${label} into memory…`, 1);
    },
    done(text = 'Ready') { this.set(text, 1); host.dataset.state = 'ready'; },
    error(e) { this.set('Couldn’t load the model: ' + (e && e.message || e)); host.dataset.state = 'error'; },
  };
}

/** Cached pipeline(task, model). opts: { ui, dtype='q8', label } */
export function pipe(task, model, opts = {}) {
  const key = `pipe|${task}|${model}|${opts.dtype || 'q8'}`;
  if (!cache.has(key)) {
    const ui = opts.ui;
    ui && ui.set(`Preparing ${opts.label || model}…`, 0);
    const p = pipeline(task, model, { dtype: opts.dtype || 'q8', device: 'wasm', progress_callback: ui ? (x) => ui.onProgress(x) : undefined })
      .then((r) => { ui && ui.done(opts.readyText || `Ready · ${model}`); return r; })
      .catch((e) => { cache.delete(key); ui && ui.error(e); throw e; });
    cache.set(key, p);
  }
  return cache.get(key);
}

/** Cached tokenizer + model. kind: 'causal' | 'masked' | 'base'. opts: { ui, dtype='q8', model_file_name, readyText } */
export function loadModel(model, kind = 'base', opts = {}) {
  const key = `model|${model}|${kind}|${opts.dtype || 'q8'}`;
  if (!cache.has(key)) {
    const ui = opts.ui;
    const cb = ui ? (x) => ui.onProgress(x) : undefined;
    const M = kind === 'causal' ? AutoModelForCausalLM : kind === 'masked' ? AutoModelForMaskedLM : AutoModel;
    const p = Promise.all([
      AutoTokenizer.from_pretrained(model, { progress_callback: cb }),
      // Decoder-only Xenova repos ship a merged decoder (with KV cache) that is much smaller than onnx/model_*.onnx
      // (distilgpt2: 85 MB vs 237 MB; gpt2 has no model_quantized.onnx at all), so default to it for causal LMs.
      M.from_pretrained(model, { dtype: opts.dtype || 'q8', device: 'wasm', progress_callback: cb,
        ...(opts.model_file_name || kind === 'causal' ? { model_file_name: opts.model_file_name || 'decoder_model_merged' } : {}) }),
    ]).then(([tokenizer, m]) => { ui && ui.done(opts.readyText || `Ready · ${model}`); return { tokenizer, model: m }; })
      .catch((e) => { cache.delete(key); ui && ui.error(e); throw e; });
    cache.set(key, p);
  }
  return cache.get(key);
}

/** Tokenizer only (small download). */
export function tokenizer(model) {
  const key = `tok|${model}`;
  if (!cache.has(key)) cache.set(key, AutoTokenizer.from_pretrained(model).catch((e) => { cache.delete(key); throw e; }));
  return cache.get(key);
}

/** Numerically stable softmax over a plain array / typed array. */
export function softmax(arr, temperature = 1) {
  let m = -Infinity; for (const v of arr) if (v > m) m = v;
  const out = new Float64Array(arr.length); let s = 0;
  for (let i = 0; i < arr.length; i++) { out[i] = Math.exp((arr[i] - m) / temperature); s += out[i]; }
  for (let i = 0; i < out.length; i++) out[i] /= s;
  return out;
}
