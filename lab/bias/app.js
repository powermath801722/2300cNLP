// WEAT, gender direction, hard debiasing and the Gonen & Goldberg clustering test on the lab's Word2Vec vectors.
import { el, $, ready, fail, plotly, plotLayout, PLOT_CONFIG, COLORS, getJSON } from '/lab/shared/lab.js';

// Word lists from Caliskan, Bryson & Narayanan (2017), lowercased for text8.
const PRESETS = {
  flowers: {
    name: 'WEAT 1 · Flowers vs insects × pleasant vs unpleasant',
    X: 'aster clover hyacinth marigold poppy azalea crocus iris orchid rose bluebell daffodil lilac pansy tulip buttercup daisy lily peony violet carnation gladiola magnolia petunia zinnia',
    Y: 'ant caterpillar flea locust spider bedbug centipede fly maggot tarantula bee cockroach gnat mosquito termite beetle cricket hornet moth wasp blackfly dragonfly horsefly roach weevil',
    A: 'caress freedom health love peace cheer friend heaven loyal pleasure diamond gentle honest lucky rainbow diploma gift honor miracle sunrise family happy laughter paradise vacation',
    B: 'abuse crash filth murder sickness accident death grief poison stink assault disaster hatred pollute tragedy divorce jail poverty ugly cancer kill rotten vomit agony prison',
  },
  instruments: {
    name: 'WEAT 2 · Instruments vs weapons × pleasant vs unpleasant',
    X: 'bagpipe cello guitar lute trombone banjo clarinet harmonica mandolin trumpet bassoon drum harp oboe tuba bell fiddle harpsichord piano viola bongo flute horn saxophone violin',
    Y: 'arrow club gun missile spear axe dagger harpoon pistol sword blade dynamite hatchet rifle tank bomb firearm knife shotgun teargas cannon grenade mace slingshot whip',
    A: 'caress freedom health love peace cheer friend heaven loyal pleasure diamond gentle honest lucky rainbow diploma gift honor miracle sunrise family happy laughter paradise vacation',
    B: 'abuse crash filth murder sickness accident death grief poison stink assault disaster hatred pollute tragedy divorce jail poverty ugly cancer kill rotten vomit agony prison',
  },
  career: {
    name: 'WEAT 6 · Male vs female names × career vs family',
    X: 'john paul mike kevin steve greg jeff bill',
    Y: 'amy joan lisa sarah diana kate ann donna',
    A: 'executive management professional corporation salary office business career',
    B: 'home parents children family cousins marriage wedding relatives',
  },
  math: {
    name: 'WEAT 7 · Math vs arts × male vs female terms',
    X: 'math algebra geometry calculus equations computation numbers addition',
    Y: 'poetry art dance literature novel symphony drama sculpture',
    A: 'male man boy brother he him his son',
    B: 'female woman girl sister she her hers daughter',
  },
  science: {
    name: 'WEAT 8 · Science vs arts × male vs female terms',
    X: 'science technology physics chemistry einstein nasa experiment astronomy',
    Y: 'poetry art shakespeare dance literature novel symphony drama',
    A: 'brother father uncle grandfather son he his him',
    B: 'sister mother aunt grandmother daughter she hers her',
  },
};
const PAIRS = 'she:he her:his woman:man mary:john herself:himself daughter:son mother:father gal:guy girl:boy female:male';
const OCC = 'homemaker nurse receptionist librarian socialite hairdresser nanny bookkeeper stylist housekeeper secretary dancer maid teacher singer designer model actress waitress doctor engineer programmer scientist architect carpenter plumber captain philosopher surgeon mechanic pilot soldier lawyer professor manager banker physicist mathematician president judge chef farmer writer artist journalist photographer warrior magician financier broadcaster';

// ---------- Word2Vec ----------
const W = { ready: false };
async function loadVectors() {
  const bar = $('#loadbar'), txt = $('#statustext');
  const vocabP = getJSON('/word2vec/model/vocab.json');
  const scalesP = fetch('/word2vec/model/scales.bin').then((r) => { if (!r.ok) throw new Error('scales.bin ' + r.status); return r.arrayBuffer(); });
  const r = await fetch('/word2vec/model/vectors.bin');
  if (!r.ok || !r.body) throw new Error('vectors.bin ' + r.status);
  const total = +r.headers.get('content-length') || 3e6, reader = r.body.getReader(), chunks = [];
  let got = 0;
  for (;;) { const { done, value } = await reader.read(); if (done) break; chunks.push(value); got += value.length; bar.style.width = Math.min(100, got / total * 100) + '%'; }
  const buf = new Uint8Array(got); let o = 0; for (const c of chunks) { buf.set(c, o); o += c.length; }
  const [vocab, sb] = await Promise.all([vocabP, scalesP]);
  const q = new Int8Array(buf.buffer), sc = new Float32Array(sb), N = vocab.words.length, D = vocab.dim;
  const E = new Float32Array(N * D);
  for (let i = 0; i < N; i++) {
    let n = 0; for (let d = 0; d < D; d++) { const v = q[i * D + d] * sc[i]; E[i * D + d] = v; n += v * v; }
    n = Math.sqrt(n) || 1; for (let d = 0; d < D; d++) E[i * D + d] /= n;
  }
  Object.assign(W, { words: vocab.words, index: new Map(vocab.words.map((w, i) => [w, i])), E, N, D, ready: true });
  txt.textContent = `Ready · ${N.toLocaleString('en-US')} words · ${D} dimensions`;
  bar.style.width = '100%';
}
const vecOf = (i) => W.E.subarray(i * W.D, (i + 1) * W.D);
const dot = (a, b) => { let s = 0; for (let i = 0; i < a.length; i++) s += a[i] * b[i]; return s; };
const unit = (v) => { const n = Math.sqrt(dot(v, v)) || 1; return v.map((x) => x / n); };
const parseList = (s) => [...new Set(String(s).toLowerCase().split(/[\s,;]+/).map((w) => w.trim()).filter(Boolean))];

// ---------- gender direction ----------
function parsePairs(s) {
  return String(s).toLowerCase().split(/[\s,;]+/).filter(Boolean).map((p) => p.split(':')).filter((p) => p.length === 2 && p[0] && p[1]);
}
/** Returns { g, ratio[], used[], missing[] } */
function genderDirection(method, pairsText) {
  const pairs = parsePairs(pairsText), used = [], missing = [];
  for (const [f, m] of pairs) {
    const a = W.index.get(f), b = W.index.get(m);
    if (a == null || b == null) { missing.push(a == null ? f : m); continue; }
    used.push([a, b, f, m]);
  }
  if (!used.length) throw new Error('None of the definitional pairs are in the vocabulary.');
  if (method === 'diff') {
    const [a, b] = used[0];
    return { g: unit(Float32Array.from(vecOf(a), (x, d) => x - vecOf(b)[d])), ratio: null, used, missing, label: `${used[0][2]} − ${used[0][3]}` };
  }
  // PCA of pair-centered vectors (Bolukbasi et al. 2016)
  const rows = [];
  for (const [a, b] of used) {
    const va = vecOf(a), vb = vecOf(b), mu = va.map((x, d) => (x + vb[d]) / 2);
    rows.push(va.map((x, d) => x - mu[d]), vb.map((x, d) => x - mu[d]));
  }
  const { comps, ratio } = pca(rows, 5);
  let g = comps[0];
  // orient so the female words of the pairs project positively
  let s = 0; for (const [a] of used) s += dot(vecOf(a), g);
  if (s < 0) g = g.map((x) => -x);
  return { g: unit(g), ratio, used, missing, label: 'PC1 of definitional pairs' };
}

/** Uncentered-by-design PCA on given rows (rows are already centered per pair). Power iteration + deflation. */
function pca(rows, k) {
  const D = rows[0].length, C = new Float64Array(D * D);
  for (const r of rows) for (let i = 0; i < D; i++) { const ri = r[i]; if (!ri) continue; for (let j = 0; j < D; j++) C[i * D + j] += ri * r[j]; }
  let trace = 0; for (let i = 0; i < D; i++) trace += C[i * D + i];
  const comps = [], ratio = [];
  for (let c = 0; c < k; c++) {
    let v = new Float64Array(D).map((_, i) => Math.sin(i * 7.3 + c * 1.7) + 0.01);
    let lam = 0;
    for (let it = 0; it < 300; it++) {
      const w = new Float64Array(D);
      for (let i = 0; i < D; i++) { let s = 0; for (let j = 0; j < D; j++) s += C[i * D + j] * v[j]; w[i] = s; }
      const n = Math.hypot(...w) || 1; lam = n; v = w.map((x) => x / n);
    }
    comps.push(Float32Array.from(v)); ratio.push(trace ? lam / trace : 0);
    for (let i = 0; i < D; i++) for (let j = 0; j < D; j++) C[i * D + j] -= lam * v[i] * v[j];
  }
  return { comps, ratio };
}
const neutralize = (v, g) => { const p = dot(v, g); return unit(Float32Array.from(v, (x, d) => x - p * g[d])); };

// ---------- WEAT ----------
let running = false, pending = false;
async function runWEAT() {
  if (!W.ready) return;
  if (running) { pending = true; return; }
  running = true; $('#run').disabled = true;
  window.__lab.lastWEAT = null;
  try {
    const lists = Object.fromEntries(['X', 'Y', 'A', 'B'].map((k) => [k, parseList($('#set' + k).value)]));
    const missing = [];
    const idx = {};
    for (const k of ['X', 'Y', 'A', 'B']) {
      idx[k] = [];
      for (const w of lists[k]) { const i = W.index.get(w); if (i == null) missing.push(w); else idx[k].push([w, i]); }
    }
    $('#missing').hidden = !missing.length;
    $('#missing').textContent = missing.length ? `Not in the 30,000-word vocabulary (skipped): ${missing.join(', ')}` : '';
    for (const k of ['X', 'Y', 'A', 'B']) if (idx[k].length < 2) throw new Error(`Set ${k} needs at least 2 in-vocabulary words.`);

    const debias = $('#wdebias').checked;
    let getV = (i) => vecOf(i), gInfo = null;
    if (debias) {
      gInfo = genderDirection($('#method').value, $('#pairs').value);
      const definitional = new Set(gInfo.used.flatMap(([a, b]) => [a, b]));
      // Attributes that are themselves definitional gender words keep their gender; everything else is neutralized.
      getV = (i) => definitional.has(i) ? vecOf(i) : neutralize(vecOf(i), gInfo.g);
    }
    const Av = idx.A.map(([, i]) => getV(i)), Bv = idx.B.map(([, i]) => getV(i));
    const s = (v) => Av.reduce((t, a) => t + dot(v, a), 0) / Av.length - Bv.reduce((t, b) => t + dot(v, b), 0) / Bv.length;
    const sX = idx.X.map(([w, i]) => ({ w, s: s(getV(i)) })), sY = idx.Y.map(([w, i]) => ({ w, s: s(getV(i)) }));
    const mean = (a) => a.reduce((t, x) => t + x, 0) / a.length;
    const all = [...sX, ...sY].map((o) => o.s);
    const S = mean(sX.map((o) => o.s)) - mean(sY.map((o) => o.s));
    const mu = mean(all), sd = Math.sqrt(all.reduce((t, x) => t + (x - mu) ** 2, 0) / (all.length - 1));
    const d = S / sd;

    // permutation test in chunks
    const N = +$('#perms').value, nx = sX.length, arr = all.slice(), tot = arr.reduce((t, x) => t + x, 0);
    let ge = 0, done = 0;
    let seed = 12345; const rnd = () => ((seed = (seed * 1664525 + 1013904223) >>> 0) / 4294967296);
    renderTiles({ d, S, p: null, nx, ny: sY.length, debias });
    while (done < N) {
      const chunk = Math.min(5000, N - done);
      for (let k = 0; k < chunk; k++) {
        let sumX = 0;
        for (let i = 0; i < nx; i++) { const j = i + Math.floor(rnd() * (arr.length - i)); const t = arr[i]; arr[i] = arr[j]; arr[j] = t; sumX += arr[i]; }
        const Sp = sumX / nx - (tot - sumX) / (arr.length - nx);
        if (Sp >= S - 1e-12) ge++;
      }
      done += chunk;
      $('#statustext').textContent = `Permutation test… ${done.toLocaleString('en-US')} / ${N.toLocaleString('en-US')}`;
      await new Promise((r) => setTimeout(r));
    }
    const p = (1 + ge) / (1 + N);
    $('#statustext').textContent = `Ready · ${W.N.toLocaleString('en-US')} words · ${W.D} dimensions`;
    renderTiles({ d, S, p, nx, ny: sY.length, debias, N });
    window.__lab.lastWEAT = { d, S, p, nx, ny: sY.length, missing, debias };
    await plotWEAT(sX, sY, lists);
    $('#weatnote').textContent = debias
      ? `Vectors neutralized along the gender direction (${gInfo.label}); definitional words in the attribute sets keep their gender. Compare d with the box unticked.`
      : 'Bars show s(w, A, B) for each target word: right of zero means closer to A, left means closer to B.';
  } catch (e) {
    $('#missing').hidden = false; $('#missing').textContent = e.message;
  } finally {
    running = false; $('#run').disabled = false;
    if (pending) { pending = false; runWEAT(); }
  }
}
function renderTiles({ d, S, p, nx, ny, debias, N }) {
  const mag = Math.abs(d) >= 0.8 ? 'large' : Math.abs(d) >= 0.5 ? 'medium' : Math.abs(d) >= 0.2 ? 'small' : 'negligible';
  const tiles = [
    [d.toFixed(2), `effect size d (${mag})`],
    [p == null ? '…' : p < 0.001 ? p.toExponential(1) : p.toFixed(3), p == null ? 'permutation p-value (running)' : `one-sided p (${N.toLocaleString('en-US')} permutations)`],
    [S.toFixed(4), 'S = mean s(X) − mean s(Y)'],
    [`${nx} / ${ny}`, 'target words used (X / Y)'],
    [debias ? 'on' : 'off', 'hard debiasing'],
  ];
  $('#tiles').replaceChildren(...tiles.map(([b, s]) => el('div', {}, [el('b', { text: b }), el('span', { text: s })])));
}
async function plotWEAT(sX, sY, lists) {
  const P = await plotly();
  const rows = [...sX.map((o) => ({ ...o, g: 'X' })), ...sY.map((o) => ({ ...o, g: 'Y' }))].sort((a, b) => a.s - b.s);
  const host = $('#weatplot');
  host.style.height = Math.max(360, rows.length * 16 + 90) + 'px';
  const tr = (g, c) => ({
    type: 'bar', orientation: 'h', name: g === 'X' ? 'X targets' : 'Y targets',
    y: rows.filter((r) => r.g === g).map((r) => r.w), x: rows.filter((r) => r.g === g).map((r) => r.s),
    marker: { color: c }, hovertemplate: '<b>%{y}</b><br>s = %{x:.4f}<extra></extra>',
  });
  P.react(host, [tr('X', COLORS[0]), tr('Y', COLORS[1])], plotLayout({
    margin: { l: 110, r: 16, t: 30, b: 48 }, barmode: 'overlay', showlegend: true,
    legend: { orientation: 'h', x: 0, y: 1.02, yanchor: 'bottom', bgcolor: 'rgba(0,0,0,0)' },
    xaxis: { title: { text: 's(w, A, B)  →  closer to A', font: { size: 12, color: '#9aa3ad' } }, gridcolor: 'rgba(228,231,235,.08)', zerolinecolor: 'rgba(228,231,235,.35)', color: '#9aa3ad' },
    yaxis: { categoryorder: 'array', categoryarray: rows.map((r) => r.w), automargin: true, color: '#c9ced6', tickfont: { size: 11 } },
  }), PLOT_CONFIG);
}

// ---------- projection ----------
async function runProjection() {
  if (!W.ready) return;
  try {
    const G = genderDirection($('#method').value, $('#pairs').value);
    const words = parseList($('#occ').value), rows = [], missing = [];
    for (const w of words) { const i = W.index.get(w); if (i == null) missing.push(w); else rows.push({ w, p: dot(vecOf(i), G.g) }); }
    rows.sort((a, b) => a.p - b.p);
    const info = [`Direction: ${G.label}${G.ratio ? ` · PC1 explains ${(G.ratio[0] * 100).toFixed(1)}% of pair variance (PC2 ${(G.ratio[1] * 100).toFixed(1)}%)` : ''}.`,
      `${G.used.length} pairs used.`];
    if (G.missing.length) info.push(`Missing pair words: ${G.missing.join(', ')}.`);
    if (missing.length) info.push(`Not in vocabulary: ${missing.join(', ')}.`);
    info.push('After neutralizing, every bar would be exactly 0, which is why the clustering test below matters.');
    $('#dirinfo').textContent = info.join(' ');
    const P = await plotly(), host = $('#projplot');
    host.style.height = Math.max(520, rows.length * 15 + 80) + 'px';
    P.react(host, [{
      type: 'bar', orientation: 'h', y: rows.map((r) => r.w), x: rows.map((r) => r.p),
      marker: { color: rows.map((r) => (r.p >= 0 ? COLORS[3] : COLORS[0])) },
      hovertemplate: '<b>%{y}</b><br>w · g = %{x:.3f}<extra></extra>',
    }], plotLayout({
      margin: { l: 110, r: 16, t: 20, b: 48 },
      xaxis: { title: { text: '← male-leaning   projection on g   female-leaning →', font: { size: 12, color: '#9aa3ad' } }, gridcolor: 'rgba(228,231,235,.08)', zerolinecolor: 'rgba(228,231,235,.35)', color: '#9aa3ad' },
      yaxis: { categoryorder: 'array', categoryarray: rows.map((r) => r.w), automargin: true, color: '#c9ced6', tickfont: { size: 11 } },
    }), PLOT_CONFIG);
    window.__lab.projection = rows;
  } catch (e) { $('#dirinfo').textContent = e.message; }
}

// ---------- Gonen & Goldberg clustering test ----------
function kmeans2(vs, restarts = 10) {
  const n = vs.length, D = vs[0].length;
  let best = null;
  let seed = 7; const rnd = () => ((seed = (seed * 1664525 + 1013904223) >>> 0) / 4294967296);
  for (let r = 0; r < restarts; r++) {
    // k-means++ init
    const c0 = vs[Math.floor(rnd() * n)];
    const d2 = vs.map((v) => { let s = 0; for (let d = 0; d < D; d++) s += (v[d] - c0[d]) ** 2; return s; });
    let tot = d2.reduce((a, b) => a + b, 0), pick = rnd() * tot, j = 0;
    while (pick > d2[j] && j < n - 1) { pick -= d2[j]; j++; }
    let C = [Float64Array.from(c0), Float64Array.from(vs[j])], lab = new Int8Array(n), inertia = 0;
    for (let it = 0; it < 60; it++) {
      let changed = false; inertia = 0;
      for (let i = 0; i < n; i++) {
        let a = 0, b = 0; for (let d = 0; d < D; d++) { a += (vs[i][d] - C[0][d]) ** 2; b += (vs[i][d] - C[1][d]) ** 2; }
        const l = a <= b ? 0 : 1; if (l !== lab[i]) changed = true; lab[i] = l; inertia += Math.min(a, b);
      }
      const S = [new Float64Array(D), new Float64Array(D)], cnt = [0, 0];
      for (let i = 0; i < n; i++) { cnt[lab[i]]++; for (let d = 0; d < D; d++) S[lab[i]][d] += vs[i][d]; }
      C = S.map((s, k) => (cnt[k] ? s.map((x) => x / cnt[k]) : C[k]));
      if (!changed && it > 0) break;
    }
    if (!best || inertia < best.inertia) best = { lab: Array.from(lab), inertia };
  }
  return best.lab;
}
const alignAcc = (lab, truth) => { let a = 0; for (let i = 0; i < lab.length; i++) a += lab[i] === truth[i] ? 1 : 0; return Math.max(a, lab.length - a) / lab.length; };

async function runGG() {
  if (!W.ready) return;
  $('#gg').disabled = true;
  try {
    $('#ggstatus').textContent = 'Projecting 20,000 words…';
    await new Promise((r) => setTimeout(r));
    const G = genderDirection($('#method').value, $('#pairs').value);
    const defin = new Set(G.used.flatMap(([a, b]) => [a, b]));
    const M = Math.min(20000, W.N), proj = [];
    for (let i = 0; i < M; i++) if (!defin.has(i) && /^[a-z]+$/.test(W.words[i])) proj.push([i, dot(vecOf(i), G.g)]);
    proj.sort((a, b) => a[1] - b[1]);
    const male = proj.slice(0, 250), female = proj.slice(-250);
    const items = [...female.map(([i]) => ({ i, t: 1 })), ...male.map(([i]) => ({ i, t: 0 }))];
    const truth = items.map((o) => o.t);
    $('#ggstatus').textContent = 'Clustering original vectors…';
    await new Promise((r) => setTimeout(r));
    const before = alignAcc(kmeans2(items.map((o) => vecOf(o.i))), truth);
    $('#ggstatus').textContent = 'Clustering neutralized vectors…';
    await new Promise((r) => setTimeout(r));
    const nv = items.map((o) => neutralize(vecOf(o.i), G.g));
    const after = alignAcc(kmeans2(nv), truth);
    $('#ggtiles').replaceChildren(...[
      [(before * 100).toFixed(1) + '%', 'cluster ↔ lean agreement, original'],
      [(after * 100).toFixed(1) + '%', 'agreement after neutralizing'],
      ['50%', 'chance level'],
    ].map(([b, s]) => el('div', {}, [el('b', { text: b }), el('span', { text: s })])));
    window.__lab.gg = { before, after };
    // 2D PCA of neutralized vectors
    const mean = new Float64Array(W.D); for (const v of nv) for (let d = 0; d < W.D; d++) mean[d] += v[d] / nv.length;
    const centered = nv.map((v) => Float32Array.from(v, (x, d) => x - mean[d]));
    const { comps } = pca(centered, 2);
    const xy = centered.map((v) => [dot(v, comps[0]), dot(v, comps[1])]);
    const P = await plotly();
    const tr = (t, name, c) => ({
      type: 'scattergl', mode: 'markers', name,
      x: xy.filter((_, k) => items[k].t === t).map((p) => p[0]), y: xy.filter((_, k) => items[k].t === t).map((p) => p[1]),
      text: items.filter((o) => o.t === t).map((o) => W.words[o.i]),
      marker: { color: c, size: 6, opacity: 0.8 }, hovertemplate: '<b>%{text}</b><extra></extra>',
    });
    P.react($('#ggplot'), [tr(1, 'originally female-leaning', COLORS[3]), tr(0, 'originally male-leaning', COLORS[0])], plotLayout({
      showlegend: true, legend: { orientation: 'h', x: 0, y: 1.02, yanchor: 'bottom', bgcolor: 'rgba(0,0,0,0)' },
      margin: { l: 48, r: 16, t: 30, b: 44 },
      xaxis: { title: { text: 'PC1 (neutralized)', font: { size: 12, color: '#9aa3ad' } }, gridcolor: 'rgba(228,231,235,.08)', zerolinecolor: 'rgba(228,231,235,.22)', color: '#9aa3ad' },
      yaxis: { title: { text: 'PC2', font: { size: 12, color: '#9aa3ad' } }, gridcolor: 'rgba(228,231,235,.08)', zerolinecolor: 'rgba(228,231,235,.22)', color: '#9aa3ad' },
    }), PLOT_CONFIG);
    $('#ggstatus').textContent = after > 0.75
      ? 'Clusters still recover most of the original lean: the bias is hidden, not removed.'
      : 'Agreement dropped a lot on this small model; on large embeddings Gonen & Goldberg report much higher numbers.';
  } catch (e) { $('#ggstatus').textContent = e.message; }
  finally { $('#gg').disabled = false; }
}

// ---------- UI ----------
function fillPreset(key) {
  const p = PRESETS[key]; if (!p) return;
  for (const k of ['X', 'Y', 'A', 'B']) $('#set' + k).value = p[k].split(' ').join(', ');
}
function init() {
  const sel = $('#preset');
  for (const [k, p] of Object.entries(PRESETS)) sel.append(el('option', { value: k, text: p.name }));
  sel.append(el('option', { value: 'custom', text: 'Custom' }));
  sel.addEventListener('change', () => { if (sel.value !== 'custom') { fillPreset(sel.value); runWEAT(); } });
  for (const k of ['X', 'Y', 'A', 'B']) $('#set' + k).addEventListener('input', () => { sel.value = 'custom'; });
  fillPreset('flowers');
  $('#pairs').value = PAIRS;
  $('#occ').value = OCC.split(' ').join(', ');
  $('#run').addEventListener('click', runWEAT);
  $('#wdebias').addEventListener('change', runWEAT);
  $('#project').addEventListener('click', runProjection);
  $('#method').addEventListener('change', runProjection);
  $('#gg').addEventListener('click', runGG);
}

init();
ready({ module: 'bias' });
try {
  await loadVectors();
  for (const id of ['#run', '#project', '#gg']) $(id).disabled = false;
  window.__lab.vectorsReady = true;
  await runProjection();
  await runWEAT();
} catch (e) { fail('Couldn’t load the Word2Vec vectors: ' + e.message); }
