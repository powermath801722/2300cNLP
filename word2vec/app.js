/* Word2Vec Lab — client-side nearest neighbors, analogies and PCA plots. */
(function () {
  'use strict';

  const BASE = '/word2vec/model/';
  const K = 10, MAX_Q = 6;
  const COLORS = ['#7ea3c8', '#d8b66a', '#8fbf9a', '#d98c7a', '#b49bd6', '#6cc3c3'];
  const REDUCED = window.matchMedia && window.matchMedia('(prefers-reduced-motion: reduce)').matches;
  const $ = (id) => document.getElementById(id);
  const el = (tag, props, kids) => {
    const n = document.createElement(tag);
    if (props) for (const k in props) {
      if (k === 'style') n.style.cssText = props[k];
      else if (k === 'text') n.textContent = props[k];
      else if (k.startsWith('on')) n.addEventListener(k.slice(2), props[k]);
      else n.setAttribute(k, props[k]);
    }
    (kids || []).forEach((c) => c != null && n.append(c));
    return n;
  };
  const fmtInt = (v) => Math.round(v).toLocaleString('en-US');

  const M = { ready: false, words: [], counts: [], index: new Map(), dim: 0, n: 0, q: null, s: null, meta: null };
  const state = { mode: '2d', queries: [], results: [], spokes: true, labels: true };

  // ---------- loading ----------
  const progress = { vec: 0, other: 0 };
  function setProgress() {
    const p = Math.min(1, progress.vec * 0.9 + progress.other * 0.1);
    $('loadbar').style.width = (p * 100).toFixed(1) + '%';
  }
  async function fetchBuf(url, onProg) {
    const r = await fetch(url);
    if (!r.ok) throw new Error(url + ' → HTTP ' + r.status);
    const total = Number(r.headers.get('content-length')) || 0;
    if (!r.body || !total || !onProg) { const b = await r.arrayBuffer(); onProg && onProg(1); return b; }
    const reader = r.body.getReader(); const chunks = []; let got = 0;
    for (;;) {
      const { done, value } = await reader.read();
      if (done) break;
      chunks.push(value); got += value.length; onProg(Math.min(1, got / total));
    }
    const out = new Uint8Array(got); let o = 0;
    for (const c of chunks) { out.set(c, o); o += c.length; }
    onProg(1);
    return out.buffer;
  }
  async function load() {
    $('statustext').textContent = 'Loading model…';
    const [vocab, vbuf, sbuf] = await Promise.all([
      fetch(BASE + 'vocab.json').then((r) => { if (!r.ok) throw new Error('vocab.json → HTTP ' + r.status); return r.json(); })
        .then((j) => { progress.other += 0.5; setProgress(); return j; }),
      fetchBuf(BASE + 'vectors.bin', (p) => { progress.vec = p; setProgress(); $('statustext').textContent = 'Loading vectors… ' + Math.round(p * 100) + '%'; }),
      fetchBuf(BASE + 'scales.bin').then((b) => { progress.other += 0.5; setProgress(); return b; }),
    ]);
    M.meta = vocab; M.dim = vocab.dim; M.words = vocab.words; M.counts = vocab.counts || [];
    M.n = M.words.length; M.q = new Int8Array(vbuf); M.s = new Float32Array(sbuf);
    if (M.q.length !== M.n * M.dim || M.s.length !== M.n) throw new Error('Model files have inconsistent sizes.');
    M.words.forEach((w, i) => M.index.set(w, i));
    M.ready = true; api.ready = true; api.words = M.words;
  }

  // ---------- math ----------
  function vec(i) {
    const out = new Float32Array(M.dim), sc = M.s[i], o = i * M.dim;
    for (let d = 0; d < M.dim; d++) out[d] = M.q[o + d] * sc;
    return out;
  }
  function normalize(v) {
    let s = 0; for (let d = 0; d < v.length; d++) s += v[d] * v[d];
    s = Math.sqrt(s) || 1; for (let d = 0; d < v.length; d++) v[d] /= s; return v;
  }
  function cos(a, b) { let s = 0; for (let d = 0; d < a.length; d++) s += a[d] * b[d]; return s; }
  // top-k by cosine against the whole vocab (rows are ~unit length), excluding a set of indices
  function topK(qv, k, exclude) {
    const q = M.q, s = M.s, D = M.dim, N = M.n;
    const bestI = new Int32Array(k).fill(-1), bestS = new Float32Array(k).fill(-Infinity);
    let minPos = 0;
    for (let i = 0; i < N; i++) {
      const o = i * D; let dot = 0;
      for (let d = 0; d < D; d++) dot += q[o + d] * qv[d];
      dot *= s[i];
      if (dot <= bestS[minPos] || (exclude && exclude.has(i))) continue;
      bestS[minPos] = dot; bestI[minPos] = i;
      minPos = 0; for (let j = 1; j < k; j++) if (bestS[j] < bestS[minPos]) minPos = j;
    }
    const out = [];
    for (let j = 0; j < k; j++) if (bestI[j] >= 0) out.push({ i: bestI[j], word: M.words[bestI[j]], sim: bestS[j] });
    return out.sort((a, b) => b.sim - a.sim);
  }
  function nearest(word, k) {
    const i = M.index.get(word); if (i == null) return null;
    return topK(normalize(vec(i)), k || K, new Set([i]));
  }
  function analogy(a, b, c, k) {
    const ia = M.index.get(a), ib = M.index.get(b), ic = M.index.get(c);
    const missing = [[a, ia], [b, ib], [c, ic]].filter(([, i]) => i == null).map(([w]) => w);
    if (missing.length) return { missing };
    const va = normalize(vec(ia)), vb = normalize(vec(ib)), vc = normalize(vec(ic));
    const t = new Float32Array(M.dim);
    for (let d = 0; d < M.dim; d++) t[d] = vb[d] - va[d] + vc[d];
    return { results: topK(normalize(t), k || 5, new Set([ia, ib, ic])) };
  }
  // PCA: top `nc` components of rows X (array of Float32Array) via power iteration with deflation
  function pca(X, nc) {
    const n = X.length, D = X[0].length;
    const mean = new Float64Array(D);
    X.forEach((x) => { for (let d = 0; d < D; d++) mean[d] += x[d] / n; });
    const Xc = X.map((x) => { const y = new Float64Array(D); for (let d = 0; d < D; d++) y[d] = x[d] - mean[d]; return y; });
    const C = Array.from({ length: D }, () => new Float64Array(D));
    for (const x of Xc) for (let a = 0; a < D; a++) { const xa = x[a]; if (!xa) continue; const row = C[a]; for (let b = 0; b < D; b++) row[b] += xa * x[b]; }
    let total = 0; for (let a = 0; a < D; a++) total += C[a][a];
    const comps = [], vars = [];
    let seed = 7; const rnd = () => ((seed = (seed * 16807) % 2147483647) / 2147483647) - 0.5;
    for (let c = 0; c < nc; c++) {
      let v = new Float64Array(D).map(rnd), lambda = 0;
      for (let it = 0; it < 300; it++) {
        const w = new Float64Array(D);
        for (let a = 0; a < D; a++) { const row = C[a]; let s = 0; for (let b = 0; b < D; b++) s += row[b] * v[b]; w[a] = s; }
        let nrm = 0; for (let a = 0; a < D; a++) nrm += w[a] * w[a]; nrm = Math.sqrt(nrm);
        if (nrm < 1e-12) break;
        let diff = 0; for (let a = 0; a < D; a++) { const nv = w[a] / nrm; diff += Math.abs(nv - v[a]); v[a] = nv; }
        lambda = nrm; if (diff < 1e-9) break;
      }
      comps.push(v); vars.push(lambda);
      for (let a = 0; a < D; a++) for (let b = 0; b < D; b++) C[a][b] -= lambda * v[a] * v[b];
    }
    const proj = Xc.map((x) => comps.map((v) => { let s = 0; for (let d = 0; d < D; d++) s += x[d] * v[d]; return s; }));
    return { proj, ratio: vars.map((l) => (total > 0 ? l / total : 0)) };
  }

  // ---------- suggestions ----------
  function lev(a, b) {
    if (Math.abs(a.length - b.length) > 2) return 9;
    const m = a.length, n = b.length; let prev = Array.from({ length: n + 1 }, (_, j) => j);
    for (let i = 1; i <= m; i++) {
      const cur = [i];
      for (let j = 1; j <= n; j++) cur[j] = Math.min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (a[i - 1] === b[j - 1] ? 0 : 1));
      prev = cur;
    }
    return prev[n];
  }
  function didYouMean(w) {
    const out = [];
    for (let i = 0; i < M.n && out.length < 3; i++) { const v = M.words[i]; if (v.length > w.length && v.startsWith(w)) out.push(v); }
    if (out.length < 3) {
      const lim = Math.min(M.n, 20000);
      const cands = [];
      for (let i = 0; i < lim; i++) { const d = lev(w, M.words[i]); if (d <= 2) cands.push([d, i]); }
      cands.sort((x, y) => x[0] - y[0] || x[1] - y[1]);
      for (const [, i] of cands) { if (out.length >= 3) break; if (!out.includes(M.words[i])) out.push(M.words[i]); }
    }
    return out;
  }
  function prefixMatches(p, k) {
    const out = [];
    if (!p) return out;
    for (let i = 0; i < M.n && out.length < k; i++) if (M.words[i].startsWith(p)) out.push(i);
    return out;
  }

  // ---------- query ----------
  function parse(s) {
    const seen = new Set(), out = [];
    for (const raw of String(s).toLowerCase().split(',')) {
      const w = raw.trim().replace(/\s+/g, '_');
      if (w && !seen.has(w)) { seen.add(w); out.push(w); }
    }
    return out;
  }
  function showMsg(html) { const m = $('msg'); if (!html) { m.hidden = true; m.textContent = ''; return; } m.hidden = false; m.innerHTML = ''; m.append(...html); }
  function run(input, opts) {
    if (!M.ready) return;
    let words = parse(input);
    const extra = words.length > MAX_Q;
    words = words.slice(0, MAX_Q);
    const known = words.filter((w) => M.index.has(w)), unknown = words.filter((w) => !M.index.has(w));
    const parts = [];
    if (unknown.length) {
      parts.push(document.createTextNode((unknown.length === 1 ? '“' + unknown[0] + '” isn’t' : unknown.map((w) => '“' + w + '”').join(', ') + ' aren’t') + ' in the 30,000-word vocabulary.'));
      unknown.forEach((w) => {
        const sug = didYouMean(w);
        if (sug.length) {
          parts.push(document.createTextNode(' Did you mean '));
          sug.forEach((s, j) => {
            if (j) parts.push(document.createTextNode(j === sug.length - 1 ? ' or ' : ', '));
            parts.push(el('a', { href: '#q=' + encodeURIComponent(s), onclick: (e) => { e.preventDefault(); const nw = words.map((x) => (x === w ? s : x)); setInput(nw.join(', ')); run(nw.join(',')); } , text: s }));
          });
          parts.push(document.createTextNode('?'));
        }
      });
    }
    if (extra) parts.push(document.createTextNode(' Showing the first ' + MAX_Q + ' words.'));
    showMsg(parts.length ? parts : null);
    if (!known.length) return;
    state.queries = known;
    state.results = known.map((w) => ({ word: w, i: M.index.get(w), nn: nearest(w, K) }));
    if (!opts || !opts.keepHash) history.replaceState(null, '', '#q=' + known.map(encodeURIComponent).join(','));
    renderLists(); renderPlot();
  }
  function setInput(v) { $('q').value = v; }

  function renderLists() {
    const box = $('lists'); box.innerHTML = '';
    state.results.forEach((r, qi) => {
      const color = COLORS[qi % COLORS.length];
      const ol = el('ol', { class: 'nn' });
      r.nn.forEach((n) => {
        const pct = Math.max(0, Math.min(1, n.sim)) * 100;
        ol.append(el('li', null, [
          el('button', { type: 'button', title: 'Explore “' + n.word + '”', onclick: () => { setInput(n.word); run(n.word); $('q').scrollIntoView({ behavior: REDUCED ? 'auto' : 'smooth', block: 'center' }); } }, [
            el('span', { class: 'w', text: n.word }),
            el('span', { class: 'track', 'aria-hidden': 'true' }, [el('i', { style: 'width:' + pct.toFixed(1) + '%;background:' + color })]),
          ]),
          el('span', { class: 'sim', text: n.sim.toFixed(3) }),
        ]));
      });
      const cnt = M.counts[r.i];
      box.append(el('div', { class: 'panel card' }, [
        el('h3', null, [el('span', { class: 'dot', style: 'background:' + color, 'aria-hidden': 'true' }), document.createTextNode(r.word)]),
        el('div', { class: 'meta', text: 'Frequency rank #' + fmtInt(r.i + 1) + (cnt ? ' · ' + fmtInt(cnt) + ' occurrences' : '') }),
        ol,
      ]));
    });
  }

  // ---------- plot ----------
  function renderPlot() {
    const host = $('plot');
    if (!state.results.length) return;
    if (!window.Plotly) { host.innerHTML = '<div class="empty">Plotting library failed to load.</div>'; return; }
    // collect points
    const pts = [], seen = new Map();
    state.results.forEach((r, qi) => {
      const add = (i, word, isQ, sim) => {
        if (seen.has(i)) { const p = pts[seen.get(i)]; if (isQ) { p.isQ = true; p.q = qi; } return; }
        seen.set(i, pts.length); pts.push({ i, word, isQ, q: qi, sim });
      };
      add(r.i, r.word, true, 1);
      r.nn.forEach((n) => add(n.i, n.word, false, n.sim));
    });
    const nc = state.mode === '3d' ? 3 : 2;
    const { proj, ratio } = pca(pts.map((p) => normalize(vec(p.i))), 3);
    const is3 = nc === 3;
    const pctTxt = (ratio.slice(0, nc).reduce((a, b) => a + b, 0) * 100).toFixed(1);
    $('varinfo').textContent = 'PCA over ' + pts.length + ' plotted words · PC1–PC' + nc + ' explain ' + pctTxt + '% of their variance (' + ratio.slice(0, nc).map((r, j) => 'PC' + (j + 1) + ' ' + (r * 100).toFixed(1) + '%').join(', ') + ')';
    $('vizhint').textContent = is3 ? 'Drag to rotate · scroll to zoom · double-click to reset' : 'Drag to zoom · double-click to reset · hover for scores';

    const traces = [];
    const coord = (k) => pts.map((p, j) => proj[j][k]);
    const P = pts.map((p, j) => ({ p, x: proj[j][0], y: proj[j][1], z: proj[j][2] }));
    if (state.spokes) {
      state.results.forEach((r, qi) => {
        const qp = P[seen.get(r.i)];
        const xs = [], ys = [], zs = [];
        r.nn.forEach((n) => { const np = P[seen.get(n.i)]; xs.push(qp.x, np.x, null); ys.push(qp.y, np.y, null); zs.push(qp.z, np.z, null); });
        const line = { color: COLORS[qi % COLORS.length], width: 1 };
        traces.push(is3
          ? { type: 'scatter3d', mode: 'lines', x: xs, y: ys, z: zs, line: { ...line, width: 2 }, opacity: 0.35, hoverinfo: 'skip', showlegend: false }
          : { type: 'scatter', mode: 'lines', x: xs, y: ys, line, opacity: 0.35, hoverinfo: 'skip', showlegend: false });
      });
    }
    state.results.forEach((r, qi) => {
      const color = COLORS[qi % COLORS.length];
      [false, true].forEach((wantQ) => {
        const sel = P.filter((o) => o.p.q === qi && o.p.isQ === wantQ);
        if (!sel.length) return;
        const t = {
          type: is3 ? 'scatter3d' : 'scatter',
          mode: state.labels ? 'markers+text' : 'markers',
          name: r.word + (wantQ ? '' : ' · neighbors'),
          legendgroup: r.word, showlegend: wantQ,
          x: sel.map((o) => o.x), y: sel.map((o) => o.y),
          text: sel.map((o) => o.p.word),
          customdata: sel.map((o) => [r.word, o.p.isQ ? 'query word' : 'cos ' + o.p.sim.toFixed(3)]),
          hovertemplate: '<b>%{text}</b><br>cluster: %{customdata[0]}<br>%{customdata[1]}<extra></extra>',
          textposition: 'top center',
          textfont: { family: 'Barlow, sans-serif', size: wantQ ? 15 : 12, color: wantQ ? '#ffffff' : '#c9ced6' },
          marker: wantQ
            ? { size: is3 ? 9 : 16, color, symbol: 'diamond', line: { color: '#ffffff', width: 2 } }
            : { size: is3 ? 5 : 9, color, opacity: 0.9, line: { color: '#121820', width: is3 ? 0 : 1 } },
        };
        if (is3) t.z = sel.map((o) => o.z);
        traces.push(t);
      });
    });

    const grid = 'rgba(228,231,235,.08)', zero = 'rgba(228,231,235,.22)';
    const axis = (title) => ({ title: { text: title, font: { size: 13, color: '#9aa3ad' } }, gridcolor: grid, zerolinecolor: zero, color: '#9aa3ad', tickfont: { size: 11 } });
    const axis3 = (title) => ({ ...axis(title), backgroundcolor: 'rgba(0,0,0,0)', showbackground: false, showspikes: false });
    const narrow = host.clientWidth < 560;
    const layout = {
      paper_bgcolor: 'rgba(0,0,0,0)', plot_bgcolor: 'rgba(0,0,0,0)',
      font: { family: 'Barlow, system-ui, sans-serif', color: '#e4e7eb' },
      margin: is3 ? { l: 0, r: 0, t: 10, b: 0 } : { l: narrow ? 40 : 56, r: 16, t: 16, b: 48 },
      showlegend: state.results.length > 1,
      legend: { orientation: 'h', x: 0, y: 1.02, yanchor: 'bottom', font: { size: 13 }, bgcolor: 'rgba(0,0,0,0)' },
      hoverlabel: { bgcolor: '#1d2732', bordercolor: '#749dc4', font: { family: 'Barlow, sans-serif', color: '#e4e7eb' } },
      uirevision: state.mode + '|' + state.queries.join(','),
      dragmode: is3 ? 'orbit' : 'zoom',
    };
    if (is3) layout.scene = { xaxis: axis3('PC1'), yaxis: axis3('PC2'), zaxis: axis3('PC3'), aspectmode: 'cube', camera: { eye: { x: 1.5, y: 1.5, z: 1.1 } } };
    else { layout.xaxis = axis('PC1'); layout.yaxis = { ...axis('PC2'), scaleanchor: 'x', scaleratio: 1 }; }
    if (host.querySelector('.empty')) host.innerHTML = '';
    // Plotly can't morph a cartesian plot into a gl3d scene in place; start fresh when the mode changes.
    if (host._w2vMode && host._w2vMode !== state.mode) window.Plotly.purge(host);
    host._w2vMode = state.mode;
    window.Plotly.react(host, traces, layout, {
      displaylogo: false, responsive: true,
      modeBarButtonsToRemove: ['lasso2d', 'select2d', 'autoScale2d', 'toImage'],
    });
  }

  // ---------- analogy ----------
  function solve() {
    if (!M.ready) return;
    const [a, b, c] = ['a1', 'a2', 'a3'].map((id) => parse($(id).value)[0] || '');
    const out = $('aout'); out.innerHTML = '';
    if (!a || !b || !c) { out.append(el('p', { class: 'msg', text: 'Fill in all three words.' })); return; }
    const r = analogy(a, b, c, 5);
    if (r.missing) { out.append(el('p', { class: 'msg', text: r.missing.map((w) => '“' + w + '”').join(', ') + (r.missing.length > 1 ? ' aren’t' : ' isn’t') + ' in the vocabulary.' })); return; }
    r.results.forEach((x) => out.append(el('span', { class: 'pill' }, [document.createTextNode(x.word), el('small', { text: x.sim.toFixed(3) })])));
  }

  // ---------- autocomplete ----------
  let sIdx = -1, sItems = [];
  function segment() {
    const inp = $('q'), v = inp.value, pos = inp.selectionStart == null ? v.length : inp.selectionStart;
    const start = v.lastIndexOf(',', pos - 1) + 1, endC = v.indexOf(',', pos), end = endC < 0 ? v.length : endC;
    return { start, end, text: v.slice(start, end).trim().toLowerCase() };
  }
  function closeSugg() { $('sugg').hidden = true; $('q').setAttribute('aria-expanded', 'false'); $('q').removeAttribute('aria-activedescendant'); sIdx = -1; sItems = []; }
  function openSugg() {
    if (!M.ready) return;
    const seg = segment();
    sItems = seg.text.length ? prefixMatches(seg.text, 8) : [];
    if (!sItems.length || (sItems.length === 1 && M.words[sItems[0]] === seg.text)) return closeSugg();
    const ul = $('sugg'); ul.innerHTML = ''; sIdx = -1;
    sItems.forEach((i, j) => ul.append(el('li', { id: 'sg' + j, role: 'option', 'aria-selected': 'false', onmousedown: (e) => { e.preventDefault(); pick(j); } }, [
      el('span', { text: M.words[i] }), el('small', { text: '#' + fmtInt(i + 1) }),
    ])));
    ul.hidden = false; $('q').setAttribute('aria-expanded', 'true');
  }
  function highlight(j) {
    sIdx = j;
    [...$('sugg').children].forEach((li, k) => li.setAttribute('aria-selected', k === j ? 'true' : 'false'));
    if (j >= 0) { $('q').setAttribute('aria-activedescendant', 'sg' + j); $('sugg').children[j].scrollIntoView({ block: 'nearest' }); }
  }
  function pick(j) {
    const inp = $('q'), seg = segment(), w = M.words[sItems[j]];
    const before = inp.value.slice(0, seg.start), after = inp.value.slice(seg.end);
    const lead = before && !before.endsWith(' ') ? ' ' : '';
    inp.value = before + lead + w + after;
    const p = (before + lead + w).length; inp.setSelectionRange(p, p);
    closeSugg();
  }

  // ---------- wiring ----------
  const api = { ready: false, words: [], nearest: (w, k) => nearest(w, k), analogy: (a, b, c, k) => analogy(a, b, c, k), state, run };
  window.__w2v = api;

  function wire() {
    $('qform').addEventListener('submit', (e) => { e.preventDefault(); closeSugg(); run($('q').value); });
    $('q').addEventListener('input', openSugg);
    $('q').addEventListener('blur', () => setTimeout(closeSugg, 100));
    $('q').addEventListener('keydown', (e) => {
      if ($('sugg').hidden) return;
      if (e.key === 'ArrowDown') { e.preventDefault(); highlight((sIdx + 1) % sItems.length); }
      else if (e.key === 'ArrowUp') { e.preventDefault(); highlight(sIdx <= 0 ? sItems.length - 1 : sIdx - 1); }
      else if (e.key === 'Enter' && sIdx >= 0) { e.preventDefault(); pick(sIdx); }
      else if (e.key === 'Escape') { e.preventDefault(); closeSugg(); }
    });
    const tab = (mode) => {
      state.mode = mode;
      $('t2').setAttribute('aria-selected', String(mode === '2d'));
      $('t3').setAttribute('aria-selected', String(mode === '3d'));
      renderPlot();
    };
    $('t2').addEventListener('click', () => tab('2d'));
    $('t3').addEventListener('click', () => tab('3d'));
    [$('t2'), $('t3')].forEach((b) => b.addEventListener('keydown', (e) => {
      if (e.key === 'ArrowRight' || e.key === 'ArrowLeft') { e.preventDefault(); const nxt = state.mode === '2d' ? 't3' : 't2'; $(nxt).focus(); $(nxt).click(); }
    }));
    $('spokes').addEventListener('change', (e) => { state.spokes = e.target.checked; renderPlot(); });
    $('labels').addEventListener('change', (e) => { state.labels = e.target.checked; renderPlot(); });
    $('aform').addEventListener('submit', (e) => { e.preventDefault(); solve(); });
    window.addEventListener('hashchange', () => { const h = hashQuery(); if (h) { setInput(h.replace(/,/g, ', ')); run(h, { keepHash: true }); } });
  }
  function hashQuery() {
    const m = /(?:^#|&)q=([^&]*)/.exec(location.hash);
    if (!m) return '';
    try { return decodeURIComponent(m[1]); } catch (_) { return m[1]; }
  }

  function chips() {
    const has = (ws) => ws.every((w) => M.index.has(w));
    const qs = ['river', 'king', 'france', 'computer', 'einstein', 'guitar', 'king, queen, prince', 'paris, guitar, physics', 'red, monday, seven'];
    const box = $('chips'); box.innerHTML = '';
    qs.filter((q) => has(parse(q))).forEach((q) => box.append(el('button', { type: 'button', class: 'chip', text: q, onclick: () => { setInput(q); run(q); } })));
    const an = [['man', 'king', 'woman'], ['france', 'paris', 'italy'], ['big', 'bigger', 'small'], ['walk', 'walked', 'swim'], ['germany', 'berlin', 'japan'], ['brother', 'sister', 'father']];
    const ab = $('achips'); ab.innerHTML = '';
    an.filter(has).forEach(([a, b, c]) => ab.append(el('button', { type: 'button', class: 'chip', text: a + ' : ' + b + ' :: ' + c + ' : ?', onclick: () => { $('a1').value = a; $('a2').value = b; $('a3').value = c; solve(); } })));
  }
  function params() {
    const p = M.meta.params || {}, dl = $('params'); dl.innerHTML = '';
    const rows = [
      ['Corpus', M.meta.corpus || 'text8'],
      ['Algorithm', p.sg === 0 ? 'CBOW' : 'Skip-gram'],
      ['Dimensions', String(M.dim)],
      ['Window', p.window != null ? '±' + p.window : '—'],
      ['Negative samples', p.negative != null ? String(p.negative) : '—'],
      ['Min count', p.min_count != null ? String(p.min_count) : '—'],
      ['Subsampling', p.sample != null ? String(p.sample) : '—'],
      ['Epochs', p.epochs != null ? String(p.epochs) : '—'],
      ['Shipped vocab', fmtInt(M.n) + ' words'],
    ];
    rows.forEach(([k, v]) => dl.append(el('dt', { text: k }), el('dd', { text: v })));
  }

  async function init() {
    wire();
    try {
      await load();
    } catch (err) {
      console.error(err);
      $('statustext').textContent = 'Couldn’t load the model (' + err.message + '). Try reloading.';
      return;
    }
    $('statustext').textContent = 'Ready · ' + fmtInt(M.n) + ' words · ' + M.dim + ' dimensions';
    $('loadbar').style.width = '100%';
    $('go').disabled = false; $('ago').disabled = false;
    params(); chips();
    const h = hashQuery(), start = h || 'river';
    setInput(start.replace(/,\s*/g, ', '));
    run(start, { keepHash: !h });
    solve();
  }
  if (document.readyState === 'loading') document.addEventListener('DOMContentLoaded', init); else init();
})();
