// Precompute passage embeddings for /lab/semantic-search/ with the exact model and pooling the page uses at runtime
// (transformers.js, Xenova/all-MiniLM-L6-v2 q8, mean pooling, L2-normalized), by running it in headless Chromium.
//
//   python3 -m http.server 8765            (from the repo root)
//   node training/semantic-search_embed.mjs [http://localhost:8765]
//
// Writes lab/semantic-search/data/emb.bin (int8, N×384, row-major) and emb-scales.bin (float32 per-row scale).
import { createRequire } from 'node:module';
import { writeFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import path from 'node:path';

const { chromium } = createRequire(import.meta.url)('playwright');
const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const base = (process.argv[2] || 'http://localhost:8765').replace(/\/$/, '');
const browser = await chromium.launch();
const page = await browser.newPage();
await page.goto(base + '/lab/semantic-search/data/corpus.json');
const vecs = await page.evaluate(async () => {
  const corpus = await (await fetch('/lab/semantic-search/data/corpus.json')).json();
  const { pipe } = await import('/lab/shared/hf.js');
  const fx = await pipe('feature-extraction', corpus.model);
  const out = [];
  for (let i = 0; i < corpus.docs.length; i += 16) {
    const t = await fx(corpus.docs.slice(i, i + 16).map((d) => d.text), { pooling: 'mean', normalize: true });
    const [n, d] = t.dims;
    for (let r = 0; r < n; r++) out.push(Array.from(t.data.slice(r * d, (r + 1) * d)));
  }
  return out;
}, null);
await browser.close();
const N = vecs.length, D = vecs[0].length, q = new Int8Array(N * D), sc = new Float32Array(N);
vecs.forEach((v, i) => {
  const m = Math.max(...v.map(Math.abs)) / 127; sc[i] = m;
  v.forEach((x, d) => { q[i * D + d] = Math.round(x / m); });
});
const dir = path.join(root, 'lab/semantic-search/data');
writeFileSync(path.join(dir, 'emb.bin'), Buffer.from(q.buffer));
writeFileSync(path.join(dir, 'emb-scales.bin'), Buffer.from(sc.buffer));
console.log(`wrote ${N}×${D} embeddings`);
