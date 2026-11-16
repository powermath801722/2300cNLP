// Braidwater Text Studio analysis worker: runs the classic pipeline off the main thread.
// Messages in:  { id, type: 'analyze', text, summaryLength } | { id, type: 'summary', text, k } | { id, type: 'warm' }
// Messages out: { id, type: 'report', report } (spelling pending) → { id, type: 'spelling', spelling, ms }
//               { id, type: 'summary', picked, lead } · { id, type: 'error', message }
import { analyzeText, summaryFor, loadClassic } from './lib/analyze.js';

self.onmessage = async (e) => {
  const { id, type } = e.data;
  try {
    if (type === 'warm') { await loadClassic(); self.postMessage({ id, type: 'warm' }); return; }
    if (type === 'summary') { const r = summaryFor(e.data.text, e.data.k); self.postMessage({ id, type: 'summary', ...r }); return; }
    if (type === 'analyze') {
      const report = await analyzeText(e.data.text, { summaryLength: e.data.summaryLength, spelling: false });
      self.postMessage({ id, type: 'report', report });
      const t0 = performance.now();
      const M = await loadClassic();
      const spelling = M.spell.checkText(e.data.text);
      self.postMessage({ id, type: 'spelling', spelling, ms: +(performance.now() - t0).toFixed(1) });
    }
  } catch (err) {
    self.postMessage({ id, type: 'error', message: String(err && err.message || err) });
  }
};
