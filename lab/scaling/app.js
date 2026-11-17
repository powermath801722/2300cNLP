import { el, $, ready, fail, plotly, plotLayout, PLOT_CONFIG, COLORS } from '/lab/shared/lab.js';
import { FITS, KAPLAN, PF_DAY, loss, flops, optimal, computeFor, kaplanN, kaplanD, kaplanC, kaplanND, kaplanAllocation, MODELS } from './laws.js';

const lab = (window.__lab = window.__lab || {});
const GPT3 = MODELS.find((m) => m.name === 'GPT-3');
const C_GOPHER = flops(280e9, 300e9);

const big = (x) => {
  if (!isFinite(x)) return '∞';
  const a = Math.abs(x);
  if (a >= 1e12) return (x / 1e12).toPrecision(3) + 'T';
  if (a >= 1e9) return (x / 1e9).toPrecision(3) + 'B';
  if (a >= 1e6) return (x / 1e6).toPrecision(3) + 'M';
  return x.toPrecision(3);
};
const sci = (x) => { if (!isFinite(x)) return '∞'; const e = Math.floor(Math.log10(x)); return `${(x / 10 ** e).toFixed(2)}×10${String(e).replace(/-/g, '⁻').replace(/\d/g, (d) => '⁰¹²³⁴⁵⁶⁷⁸⁹'[d])}`; };
const f3 = (x) => (isFinite(x) ? x.toFixed(3) : '—');
const C = () => Math.pow(10, +$('#logc').value);

function stats(host, items) { const h = $(host); h.innerHTML = ''; for (const [v, l] of items) h.append(el('div', {}, [el('b', { text: v }), el('span', { text: l })])); }

// ---------- 01 budget table ----------
function renderBudget() {
  const c = C(), H = FITS.hoffmann, Bz = FITS.besiroglu;
  $('#cout').textContent = sci(c);
  $('#cnote').textContent = `${sci(c)} FLOPs ≈ ${big(c / PF_DAY)} petaflop/s-days. For scale: Gopher and Chinchilla each used about ${sci(C_GOPHER)} FLOPs (6 × N × D with their reported sizes).`;
  const rows = [];
  const oh = optimal(H, c), ob = optimal(Bz, c), k = kaplanAllocation(c, GPT3);
  const row = (name, N, D, note) => rows.push({ name, N, D, note, lh: loss(H, N, D), lb: loss(Bz, N, D) });
  row('Compute-optimal · Hoffmann fit', oh.N, oh.D);
  row('Compute-optimal · Besiroglu fit', ob.N, ob.D);
  row('20 tokens per parameter (rule of thumb)', Math.sqrt(c / 120), 20 * Math.sqrt(c / 120));
  row('N ∝ C^0.73 through GPT-3 (Kaplan exponent)', k.N, k.D);
  const tb = $('#opttable tbody'); tb.innerHTML = '';
  const bestH = Math.min(...rows.map((r) => r.lh)), bestB = Math.min(...rows.map((r) => r.lb));
  for (const r of rows) {
    tb.append(el('tr', {}, [el('td', { text: r.name }), el('td', { text: big(r.N) }), el('td', { text: big(r.D) }), el('td', { text: (r.D / r.N).toFixed(1) }),
      el('td', { text: f3(r.lh), style: { color: r.lh === bestH ? '#8fbf9a' : 'inherit' } }), el('td', { text: f3(r.lb), style: { color: r.lb === bestB ? '#8fbf9a' : 'inherit' } })]));
  }
  lab.budget = { C: c, hoffmann: oh, besiroglu: ob, kaplan: k, rows };
  renderKaplan(c);
}

// ---------- 02 map ----------
async function renderMap() {
  const P = await plotly(), f = FITS[$('#fit').value], c = C();
  const xs = [], ys = [];
  for (let i = 0; i <= 90; i++) xs.push(7 + i * (13 - 7) / 90);      // log10 N
  for (let j = 0; j <= 90; j++) ys.push(9 + j * (15 - 9) / 90);      // log10 D
  const z = ys.map((ld) => xs.map((ln) => loss(f, 10 ** ln, 10 ** ld)));
  const traces = [{
    type: 'contour', x: xs, y: ys, z, colorscale: [[0, '#d6ebff'], [0.35, '#7ea3c8'], [1, '#18202a']], reversescale: false,
    contours: { start: Math.ceil((f.E + 0.05) * 20) / 20, end: 3.2, size: 0.1, coloring: 'heatmap', showlabels: true, labelfont: { size: 10, color: '#121820' } },
    line: { color: 'rgba(18,24,32,.6)', width: 0.6 }, colorbar: { title: { text: 'loss', side: 'right' }, thickness: 12, tickfont: { color: '#9aa3ad' } },
    hovertemplate: 'N = 10^%{x:.2f}<br>D = 10^%{y:.2f}<br>loss %{z:.3f}<extra></extra>', name: 'loss',
  }];
  // iso-FLOP line for the chosen budget: log D = log(C/6) − log N
  const lnLine = xs.filter((ln) => { const ld = Math.log10(c / 6) - ln; return ld >= 9 && ld <= 15; });
  traces.push({ x: lnLine, y: lnLine.map((ln) => Math.log10(c / 6) - ln), mode: 'lines', name: `iso-FLOP ${sci(c)}`, line: { color: '#d8b66a', dash: 'dash', width: 2 }, hoverinfo: 'skip' });
  // compute-optimal frontiers
  for (const [key, col] of [['hoffmann', COLORS[3]], ['besiroglu', '#e4e7eb']]) {
    const pts = []; for (let lc = 18; lc <= 28.01; lc += 0.25) { const o = optimal(FITS[key], 10 ** lc); pts.push([Math.log10(o.N), Math.log10(o.D)]); }
    traces.push({ x: pts.map((p) => p[0]), y: pts.map((p) => p[1]), mode: 'lines', name: `optimal · ${key === 'hoffmann' ? 'Hoffmann' : 'Besiroglu'}`, line: { color: col, width: 2.5 }, hoverinfo: 'skip' });
  }
  const o = optimal(f, c);
  traces.push({ x: [Math.log10(o.N)], y: [Math.log10(o.D)], mode: 'markers', name: 'optimum at this budget', marker: { size: 13, symbol: 'star', color: '#d8b66a', line: { color: '#121820', width: 1 } }, hovertemplate: `N* ${big(o.N)}<br>D* ${big(o.D)}<br>loss ${f3(o.L)}<extra></extra>` });
  traces.push({
    x: MODELS.map((m) => Math.log10(m.N)), y: MODELS.map((m) => Math.log10(m.D)), mode: 'markers+text', type: 'scatter', name: 'reported models',
    text: MODELS.map((m) => m.name), textposition: 'top center', textfont: { size: 11, color: '#e4e7eb' },
    marker: { size: 9, color: COLORS[2], line: { color: '#121820', width: 1 } },
    customdata: MODELS.map((m) => [big(m.N), big(m.D), sci(flops(m.N, m.D)), (m.D / m.N).toFixed(1), f3(loss(f, m.N, m.D)), m.src]),
    hovertemplate: '<b>%{text}</b><br>N %{customdata[0]} · D %{customdata[1]}<br>C ≈ %{customdata[2]} FLOPs · %{customdata[3]} tokens/param<br>fit predicts %{customdata[4]} (illustrative)<br>%{customdata[5]}<extra></extra>',
  });
  P.react('map', traces, plotLayout({
    margin: { l: 60, r: 10, t: 10, b: 90 },
    xaxis: { ...plotLayout().xaxis, title: { text: 'parameters N (log₁₀)' }, range: [7, 13] },
    yaxis: { ...plotLayout().yaxis, title: { text: 'training tokens D (log₁₀)' }, range: [9, 15] },
    legend: { orientation: 'h', y: -0.2, font: { size: 11 } },
  }), PLOT_CONFIG);
}

// ---------- 03 loss vs compute ----------
async function renderCurves() {
  const P = await plotly(), cs = [];
  for (let lc = 18; lc <= 28.01; lc += 0.1) cs.push(10 ** lc);
  const H = FITS.hoffmann, Bz = FITS.besiroglu;
  P.react('lossc', [
    { x: cs, y: cs.map((c) => optimal(H, c).L), name: 'optimal · Hoffmann', mode: 'lines', line: { color: COLORS[3], width: 2.5 } },
    { x: cs, y: cs.map((c) => optimal(Bz, c).L), name: 'optimal · Besiroglu', mode: 'lines', line: { color: '#e4e7eb', width: 2.5 } },
    { x: cs, y: cs.map((c) => { const k = kaplanAllocation(c, GPT3); return loss(Bz, k.N, k.D); }), name: 'N ∝ C^0.73 via GPT-3 · Besiroglu', mode: 'lines', line: { color: COLORS[1], width: 2, dash: 'dash' } },
    { x: [cs[0], cs[cs.length - 1]], y: [Bz.E, Bz.E], name: `irreducible E = ${Bz.E}`, mode: 'lines', line: { color: 'rgba(228,231,235,.35)', dash: 'dot', width: 1 } },
    { x: [C(), C()], y: [1.6, 4], name: 'chosen budget', mode: 'lines', line: { color: '#d8b66a', width: 1 } },
  ], plotLayout({ margin: { l: 52, r: 10, t: 10, b: 84 }, xaxis: { ...plotLayout().xaxis, type: 'log', title: { text: 'training compute (FLOPs)' }, exponentformat: 'power' }, yaxis: { ...plotLayout().yaxis, title: { text: 'predicted loss (nats/token)' }, range: [1.6, 4] }, legend: { orientation: 'h', y: -0.32, font: { size: 11 } } }), PLOT_CONFIG);
  P.react('ratio', [
    { x: cs, y: cs.map((c) => optimal(H, c).ratio), name: 'Hoffmann fit', mode: 'lines', line: { color: COLORS[3], width: 2.5 } },
    { x: cs, y: cs.map((c) => optimal(Bz, c).ratio), name: 'Besiroglu fit', mode: 'lines', line: { color: '#e4e7eb', width: 2.5 } },
    { x: cs, y: cs.map((c) => { const k = kaplanAllocation(c, GPT3); return k.D / k.N; }), name: 'N ∝ C^0.73 via GPT-3', mode: 'lines', line: { color: COLORS[1], width: 2, dash: 'dash' } },
    { x: [cs[0], cs[cs.length - 1]], y: [20, 20], name: '20 tokens/param', mode: 'lines', line: { color: 'rgba(228,231,235,.35)', dash: 'dot', width: 1 } },
  ], plotLayout({ margin: { l: 52, r: 10, t: 10, b: 84 }, xaxis: { ...plotLayout().xaxis, type: 'log', title: { text: 'training compute (FLOPs)' }, exponentformat: 'power' }, yaxis: { ...plotLayout().yaxis, type: 'log', title: { text: 'optimal tokens per parameter' } }, legend: { orientation: 'h', y: -0.32, font: { size: 11 } } }), PLOT_CONFIG);
}

// ---------- 04 calculator ----------
function renderCalc() {
  const f = FITS[$('#calcfit').value];
  const N = Math.max(1e6, (+$('#nB').value || 7) * 1e9), D = Math.max(1e8, (+$('#dB').value || 2000) * 1e9), target = +$('#target').value || 2;
  const c = flops(N, D), L = loss(f, N, D), o = optimal(f, c), need = computeFor(f, L), needT = computeFor(f, target);
  stats('#calcstats', [
    [sci(c), 'training FLOPs (6ND)'], [(D / N).toFixed(1), 'tokens per parameter'], [f3(L), 'predicted loss'],
    [f3(o.L), `best loss at same compute (N* ${big(o.N)})`], [(100 * need / c).toFixed(0) + '%', 'compute an optimal model would need for this loss'],
    [isFinite(needT) ? sci(needT) : 'unreachable', `FLOPs to reach loss ${target.toFixed(2)} optimally`],
  ]);
  $('#calcnote').textContent = isFinite(needT)
    ? `Under the ${f.label} fit, loss ${target.toFixed(2)} needs about ${sci(needT)} FLOPs with N* ≈ ${big(optimal(f, needT).N)} and D* ≈ ${big(optimal(f, needT).D)}. No budget gets below E = ${f.E}.`
    : `Loss ${target.toFixed(2)} is below this fit’s irreducible loss E = ${f.E}, so no amount of compute reaches it.`;
  lab.calc = { N, D, C: c, L, optimalL: o.L, efficiency: need / c, needTarget: needT };
}

// ---------- 05 Kaplan ----------
function renderKaplan(c) {
  const k = kaplanAllocation(c, GPT3);
  const rows = [
    ['C in petaflop/s-days', big(c / PF_DAY)],
    ['L(C) — compute-efficient frontier', f3(kaplanC(c))],
    [`L(N) at N = ${big(k.N)} (N ∝ C^0.73 via GPT-3)`, f3(kaplanN(k.N))],
    [`L(D) at D = ${big(k.D)}`, f3(kaplanD(k.D))],
    ['L(N, D) for that allocation', f3(kaplanND(k.N, k.D))],
    [`L(N, D) for GPT-3 itself (175B, 300B tokens)`, f3(kaplanND(GPT3.N, GPT3.D))],
  ];
  const tb = $('#kaptable tbody'); tb.innerHTML = '';
  for (const [a, b] of rows) tb.append(el('tr', {}, [el('td', { text: a }), el('td', { text: b })]));
}

// ---------- wiring ----------
function all() { renderBudget(); renderMap(); renderCurves(); }
$('#logc').addEventListener('input', () => { renderBudget(); renderMap(); renderCurves(); });
$('#fit').addEventListener('change', renderMap);
for (const id of ['nB', 'dB', 'calcfit', 'target']) $('#' + id).addEventListener('input', renderCalc);
lab.setLogC = (v) => { $('#logc').value = v; all(); return lab.budget; };
lab.setCalc = (nB, dB, target) => { $('#nB').value = nB; $('#dB').value = dB; if (target != null) $('#target').value = target; renderCalc(); return lab.calc; };
lab.laws = { FITS, KAPLAN, optimal, loss, flops };

try {
  $('#logc').value = Math.log10(C_GOPHER).toFixed(2);
  all(); renderCalc();
  ready();
} catch (e) { fail(e.message); }
