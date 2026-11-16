// Braidwater Text Studio service worker: makes the classic (no-download) analysis work offline after one visit.
// Strategy: stale-while-revalidate for same-origin GETs made by Studio pages; precache the app shell and data.
// Hugging Face model files are cached separately by transformers.js (Cache API), so they are not handled here.
const CACHE = 'text-studio-v1';
const PRECACHE = [
  '/studio/', '/studio/app.js', '/studio/worker.js', '/studio/samples.js', '/studio/sw.js',
  '/studio/lib/analyze.js', '/studio/lib/tokenize.js', '/studio/lib/textrank.js', '/studio/lib/readability.js', '/studio/lib/style.js',
  '/studio/lib/sentiment.js', '/studio/lib/pos.js', '/studio/lib/topics.js', '/studio/lib/spell.js', '/studio/lib/ner.js', '/studio/lib/signals.js',
  '/lab/shared/lab.css', '/lab/shared/lab.js', '/lab/shared/modules.js',
  '/lab/sentiment/data/vader.json', '/lab/sentiment/data/models.json', '/lab/pos/data/hmm.json', '/lab/topics/data/lda.json',
  '/lab/spell/data/freq.txt', '/lab/textrank/data/samples.json', '/assets/vendor/plotly-2.35.2.min.js', '/favicon.svg',
];

self.addEventListener('install', (e) => {
  e.waitUntil(caches.open(CACHE).then((c) => Promise.all(PRECACHE.map((u) => c.add(u).catch(() => {})))).then(() => self.skipWaiting()));
});
self.addEventListener('activate', (e) => {
  e.waitUntil(caches.keys().then((ks) => Promise.all(ks.filter((k) => k.startsWith('text-studio-') && k !== CACHE).map((k) => caches.delete(k)))).then(() => self.clients.claim()));
});
self.addEventListener('fetch', (e) => {
  const req = e.request, url = new URL(req.url);
  if (req.method !== 'GET' || url.origin !== location.origin) return;
  e.respondWith(caches.open(CACHE).then(async (c) => {
    const hit = await c.match(req, { ignoreSearch: true });
    const net = fetch(req).then((res) => { if (res.ok && res.type === 'basic') c.put(req, res.clone()); return res; }).catch(() => hit);
    return hit ? (e.waitUntil(net), hit) : net;
  }));
});
