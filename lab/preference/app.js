import { el, $, ready, fail, plotly, plotLayout, PLOT_CONFIG, COLORS } from '/lab/shared/lab.js';
import { PROMPTS, gold, annotScore, sftRef, ANNOT, GOLD } from './core.js';

const worker = new Worker(new URL('./worker.js', import.meta.url), { type: 'module' });
const lab = (window.__lab = window.__lab || {});
const S = { data: null, rm: null, sweep: null, policy: null, align: null, rmCurve: [], alCurve: [] };
const ref = sftRef();
const fmt = (x, d = 2) => (x == null || isNaN(x) ? '—' : Number(x).toFixed(d));
const pct = (x) => (x == null ? '—' : (x * 100).toFixed(1) + '%');
const waiters = {};
const once = (type) => new Promise((res) => (waiters[type] ||= []).push(res));
const betaVal = () => Math.pow(10, +$('#beta').value);

function stats(host, items) { const h = $(host); h.innerHTML = ''; for (const [v, l] of items) h.append(el('div', {}, [el('b', { text: v }), el('span', { text: l })])); }
function status(id, text, frac) { const s = $(id); s.querySelector('.t').textContent = text; if (frac != null) s.querySelector('i').style.width = Math.round(frac * 100) + '%'; }

// ---------- candidates table ----------
PROMPTS.forEach((P, i) => $('#prompt').append(el('option', { value: i, text: P.q })));
function renderTable() {
  const p = +$('#prompt').value, tb = $('#cands tbody'); tb.innerHTML = '';
  const best = Math.max(...PROMPTS[p].c.map(gold));
  PROMPTS[p].c.forEach((c, i) => {
    const r = S.rm ? S.rm.rewards[p][i] : null;
    tb.append(el('tr', {}, [
      el('td', {}, [c.t, c.flat ? el('span', { class: 'flag', text: 'flattery' }) : null, gold(c) === best ? el('span', { class: 'flag', style: { color: '#8fbf9a' }, text: 'best' }) : null]),
      el('td', { text: c.h.toFixed(2) }), el('td', { text: String(c.words) }), el('td', { text: fmt(gold(c)) }),
      el('td', { text: fmt(annotScore(c)) }), el('td', { text: fmt(r) }), el('td', { text: pct(ref[p][i]) }),
    ]));
  });
}

// ---------- plots ----------
async function plotRM() {
  const P = await plotly();
  P.react('rmplot', [
    { x: S.rmCurve.map((c) => c.i), y: S.rmCurve.map((c) => c.nll), name: 'training NLL', mode: 'lines', line: { color: COLORS[0], width: 2.5 } },
    { x: S.rmCurve.map((c) => c.i), y: S.rmCurve.map((c) => c.test), name: 'held-out NLL', mode: 'lines', line: { color: COLORS[1], width: 2 } },
  ], plotLayout({ margin: { l: 52, r: 12, t: 12, b: 64 }, xaxis: { ...plotLayout().xaxis, title: { text: 'gradient step' } }, yaxis: { ...plotLayout().yaxis, title: { text: 'mean −log likelihood' } }, legend: { orientation: 'h', y: -0.25 } }), PLOT_CONFIG);
}

async function plotSweep() {
  if (!S.sweep) return;
  const P = await plotly(), pts = S.sweep.points, cur = S.policy;
  const x = pts.map((q) => q.sqrtKL);
  const traces = [
    { x, y: pts.map((q) => q.proxy), name: 'reward model (proxy)', mode: 'lines', line: { color: COLORS[1], width: 2.5 } },
    { x, y: pts.map((q) => q.gold), name: 'gold utility', mode: 'lines', line: { color: COLORS[2], width: 2.5 } },
    { x: [0, Math.max(...x)], y: [S.sweep.bestPossible, S.sweep.bestPossible], name: 'best possible gold', mode: 'lines', line: { color: 'rgba(143,191,154,.5)', dash: 'dot', width: 1 } },
  ];
  if (cur) traces.push({ x: [Math.sqrt(cur.kl), Math.sqrt(cur.kl)], y: [cur.gold, cur.proxy], name: `current β = ${fmt(cur.beta)}`, mode: 'markers', marker: { size: 11, color: ['#8fbf9a', '#d8b66a'], line: { color: '#fff', width: 1.5 } } });
  P.react('ooplot', traces, plotLayout({ margin: { l: 52, r: 12, t: 12, b: 64 }, xaxis: { ...plotLayout().xaxis, title: { text: '√KL(π ‖ π_ref)  (nats^½, mean over prompts)' } }, yaxis: { ...plotLayout().yaxis, title: { text: 'expected score per prompt' } }, legend: { orientation: 'h', y: -0.25 } }), PLOT_CONFIG);
}

const shortLabel = (c, i) => `${i + 1}. ${c.h >= 0.9 ? 'best' : c.words > 150 ? (c.flat ? 'long+flattery' : 'long') : c.flat ? 'flattery' : c.h < 0.1 ? 'wrong' : 'weak'}`;

async function plotPolicy() {
  if (!S.policy) return;
  const P = await plotly(), p = +$('#prompt').value, cs = PROMPTS[p].c, labels = cs.map(shortLabel);
  P.react('polplot', [
    { x: labels, y: S.policy.ref[p], name: 'π_ref (SFT)', type: 'bar', marker: { color: 'rgba(228,231,235,.35)' } },
    { x: labels, y: S.policy.pi[p], name: `π* at β = ${fmt(S.policy.beta)}`, type: 'bar', marker: { color: COLORS[0] } },
  ], plotLayout({ barmode: 'group', margin: { l: 52, r: 12, t: 12, b: 90 }, yaxis: { ...plotLayout().yaxis, title: { text: 'probability' }, range: [0, 1] }, xaxis: { ...plotLayout().xaxis, tickangle: -25 }, legend: { orientation: 'h', y: -0.42 } }), PLOT_CONFIG);
}

async function plotAlign() {
  const P = await plotly(), c = S.alCurve;
  P.react('alplot', [
    { x: c.map((q) => q.i), y: c.map((q) => Math.max(q.klRL, 1e-16)), name: 'RLHF → π*(reward model)', mode: 'lines', line: { color: COLORS[0], width: 2.5 } },
    { x: c.map((q) => q.i), y: c.map((q) => Math.max(q.klDPOtab, 1e-16)), name: 'DPO → π*(tabular BT, same pairs)', mode: 'lines', line: { color: COLORS[1], width: 2.5 } },
    { x: c.map((q) => q.i), y: c.map((q) => Math.max(q.klDPOrm, 1e-16)), name: 'DPO → π*(reward model)', mode: 'lines', line: { color: COLORS[3], width: 2, dash: 'dot' } },
  ], plotLayout({ margin: { l: 60, r: 12, t: 12, b: 64 }, xaxis: { ...plotLayout().xaxis, title: { text: 'iteration' } }, yaxis: { ...plotLayout().yaxis, title: { text: 'KL to target (log scale)' }, type: 'log', exponentformat: 'power' }, legend: { orientation: 'h', y: -0.25 } }), PLOT_CONFIG);
  if (!S.align) return;
  const p = +$('#prompt').value, labels = PROMPTS[p].c.map(shortLabel), A = S.align;
  P.react('albars', [
    { x: labels, y: A.ref[p], name: 'π_ref', type: 'bar', marker: { color: 'rgba(228,231,235,.35)' } },
    { x: labels, y: A.piRL[p], name: 'RLHF', type: 'bar', marker: { color: COLORS[0] } },
    { x: labels, y: A.piDPO[p], name: 'DPO', type: 'bar', marker: { color: COLORS[1] } },
  ], plotLayout({ barmode: 'group', margin: { l: 52, r: 12, t: 12, b: 90 }, yaxis: { ...plotLayout().yaxis, title: { text: 'probability' }, range: [0, 1] }, xaxis: { ...plotLayout().xaxis, tickangle: -25 }, legend: { orientation: 'h', y: -0.42 } }), PLOT_CONFIG);
}

// ---------- worker messages ----------
worker.onmessage = (e) => {
  const m = e.data;
  if (m.type === 'data') {
    S.data = m; S.rm = null; S.rmCurve = [];
    stats('#datastats', [[m.nPairs.toLocaleString('en-US'), 'comparisons collected'], [pct(m.longerWon), 'won by the longer answer'], [pct(m.flatWon), 'won by the flattering answer'], [pct(m.goldAgree), 'agree with the gold ranking'], [pct(m.bayes), 'best possible held-out accuracy']]);
    $('#fitrm').disabled = false; status('#rmstatus', 'Ready to train.', 0);
    renderTable();
  } else if (m.type === 'rmProgress') {
    S.rmCurve.push(m); status('#rmstatus', `Training… step ${m.i} of ${m.iters} · held-out NLL ${m.test.toFixed(3)}`, m.i / m.iters);
    if (m.i % 200 === 0) plotRM();
  } else if (m.type === 'rm') {
    S.rm = m; lab.rm = m;
    $('#fitrm').disabled = false; $('#pausebtn').disabled = true;
    status('#rmstatus', 'Trained.', 1);
    plotRM(); renderTable();
    stats('#rmstats', [[pct(m.accTest), 'held-out pairs ranked like the annotators'], [pct(S.data.bayes), 'ceiling (true annotator scores)'], [pct(m.goldAgree), 'held-out pairs ranked like gold']]);
    const tb = $('#wtable tbody'); tb.innerHTML = '';
    [['helpfulness', m.w[0], ANNOT.h, GOLD.h], ['length (per 100 words)', m.w[1], ANNOT.len * 100, 'penalty past 80'], ['flattery', m.w[2], ANNOT.flat, GOLD.flat]]
      .forEach(([n, l, a, g]) => tb.append(el('tr', {}, [el('td', { text: n }), el('td', { text: fmt(l) }), el('td', { text: fmt(a) }), el('td', { text: typeof g === 'number' ? fmt(g) : g })])));
    $('#beta').disabled = false; $('#align').disabled = false; status('#alstatus', 'Ready.', 0);
    worker.postMessage({ cmd: 'sweep' }); worker.postMessage({ cmd: 'policy', beta: betaVal() });
  } else if (m.type === 'sweep') {
    S.sweep = m; lab.sweep = m; plotSweep();
  } else if (m.type === 'policy') {
    S.policy = m; lab.policy = m;
    stats('#oostats', [[fmt(m.kl, 3), 'KL(π* ‖ π_ref), nats'], [fmt(m.proxy), 'reward-model score'], [fmt(m.gold), 'gold utility'], [S.sweep ? fmt(S.sweep.refGold) : '—', 'gold utility of π_ref'], [S.sweep ? `${fmt(S.sweep.best.gold)} at β ${fmt(S.sweep.best.beta)}` : '—', 'peak gold utility']]);
    plotSweep(); plotPolicy();
  } else if (m.type === 'alignProgress') {
    S.alCurve.push(m); status('#alstatus', `Running… iteration ${m.i} of ${m.iters}`, m.i / m.iters);
    if (m.i % 200 === 0) plotAlign();
  } else if (m.type === 'align') {
    S.align = m; lab.align = m; $('#align').disabled = false;
    const last = S.alCurve[S.alCurve.length - 1];
    status('#alstatus', 'Done.', 1);
    stats('#alstats', [[fmt(m.gold.ref), 'gold utility · π_ref'], [fmt(m.gold.rl), 'gold utility · RLHF'], [fmt(m.gold.dpo), 'gold utility · DPO'], [Math.max(0, last.klRL).toExponential(1), 'KL(RLHF ‖ closed form)'], [Math.max(0, last.klDPOtab).toExponential(1), 'KL(DPO ‖ tabular-BT closed form)']]);
    $('#alnote').textContent = `At β = ${fmt(m.beta)}: RLHF reaches its closed-form target to numerical precision; DPO matches the optimum for a per-answer Bradley–Terry reward fitted on the same ${S.data.nTrain.toLocaleString('en-US')} training pairs, and sits ${fmt(Math.max(0, last.klDPOrm), 3)} nats (KL) from the reward-model policy because it can’t share what it learns across answers. Both started from π_ref with gold utility ${fmt(m.gold.ref)}.`;
    plotAlign();
  } else if (m.type === 'paused') {
    $('#fitrm').disabled = false; $('#align').disabled = false; $('#pausebtn').disabled = true;
  } else if (m.type === 'error') { fail(m.message); }
  (waiters[m.type] || []).splice(0).forEach((f) => f(m));
};
worker.onerror = (e) => fail(e.message || 'Worker failed');

// ---------- controls ----------
function collect() {
  worker.postMessage({ cmd: 'pause' });
  S.sweep = null; S.policy = null; S.align = null; S.alCurve = [];
  $('#beta').disabled = true; $('#align').disabled = true;
  worker.postMessage({ cmd: 'collect', n: +$('#npairs').value, tau: +$('#tau').value, seed: Math.max(1, Math.round(+$('#seed').value || 1)) });
}
function fitRM() { S.rmCurve = []; $('#fitrm').disabled = true; $('#pausebtn').disabled = false; worker.postMessage({ cmd: 'fitRM', iters: 2000, l2: 1e-4 }); }
function align() { S.alCurve = []; S.align = null; $('#align').disabled = true; worker.postMessage({ cmd: 'align', beta: betaVal(), iters: +$('#iters').value }); }

$('#collect').addEventListener('click', async () => { collect(); await once('data'); fitRM(); });
$('#fitrm').addEventListener('click', fitRM);
$('#pausebtn').addEventListener('click', () => worker.postMessage({ cmd: 'pause' }));
$('#align').addEventListener('click', align);
$('#prompt').addEventListener('change', () => { renderTable(); plotPolicy(); plotAlign(); });
const showBeta = () => { $('#betaout').textContent = `β = ${betaVal() >= 10 ? betaVal().toFixed(0) : betaVal() >= 1 ? betaVal().toFixed(2) : betaVal().toFixed(3)}`; };
$('#beta').addEventListener('input', () => { showBeta(); worker.postMessage({ cmd: 'policy', beta: betaVal() }); });

lab.S = S;
lab.setBeta = (b) => { $('#beta').value = Math.log10(b); showBeta(); worker.postMessage({ cmd: 'policy', beta: b }); return once('policy'); };
lab.runAlign = () => { const p = once('align'); align(); return p; };
lab.retrain = async (opts = {}) => { if (opts.n) $('#npairs').value = opts.n; if (opts.tau) $('#tau').value = opts.tau; collect(); await once('data'); const p = once('rm'); fitRM(); return p; };

try {
  renderTable(); showBeta();
  collect();
  once('data').then(() => { fitRM(); return once('rm'); }).then(() => ready());
} catch (e) { fail(e.message); }
