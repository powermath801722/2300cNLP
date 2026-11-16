// Adds DistilBERT SST-2 log-odds to lab/calibration/data/sst2.json, computed in headless Chromium with the
// exact browser code path the page uses (lab/attribution/models.js → transformers.js 3.8.1, q8 weights).
//   python3 -m http.server 8765        (repo root)
//   node training/calibration_logits.mjs http://localhost:8765 lab/calibration/data/sst2.json
import { createRequire } from 'node:module';
import { readFileSync, writeFileSync } from 'node:fs';
const { chromium } = createRequire(import.meta.url)('playwright');
const [base = 'http://localhost:8765', file = 'lab/calibration/data/sst2.json'] = process.argv.slice(2);
const data = JSON.parse(readFileSync(file, 'utf8'));
const b = await chromium.launch();
const p = await b.newPage();
await p.goto(base + '/lab/');
const t0 = Date.now();
const z = await p.evaluate(async (s) => {
  const { distilbert } = await import('/lab/attribution/models.js');
  const m = await distilbert();
  return m.score(s, { batch: 32 });
}, data.sentence);
await b.close();
data.distilbert_logodds = z.map((v) => Math.round(v * 1e4) / 1e4);
data.distilbert_note = 'Xenova/distilbert-base-uncased-finetuned-sst-2-english, q8 ONNX, transformers.js 3.8.1 (wasm), log P(pos) − log P(neg)';
writeFileSync(file, JSON.stringify(data));
const acc = z.filter((v, i) => (v > 0 ? 1 : 0) === data.label[i]).length / z.length;
console.log(`${z.length} sentences in ${((Date.now() - t0) / 1000).toFixed(1)} s, accuracy ${(acc * 100).toFixed(2)}%`);
