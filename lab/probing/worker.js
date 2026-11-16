// Probe trainer: softmax regression (linear probe) or a one-hidden-layer MLP, trained with Adam
// on standardized features. Runs off the main thread.
//   in:  { id, X: Float32Array (n*d), y: Int32Array, Xt, yt, d, k, kind: 'linear'|'mlp', hidden, epochs, lr, l2, seed }
//   out: { id, acc, trainAcc, pred: Int32Array, curve: [testAcc per epoch] }
function rng(seed) { let s = seed >>> 0 || 1; return () => { s ^= s << 13; s >>>= 0; s ^= s >>> 17; s ^= s << 5; s >>>= 0; return s / 4294967296; }; }
function gauss(r) { let u = 0, v = 0; while (!u) u = r(); v = r(); return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v); }

function standardize(X, Xt, n, nt, d) {
  const mu = new Float64Array(d), sd = new Float64Array(d);
  for (let i = 0; i < n; i++) for (let j = 0; j < d; j++) mu[j] += X[i * d + j] / n;
  for (let i = 0; i < n; i++) for (let j = 0; j < d; j++) { const z = X[i * d + j] - mu[j]; sd[j] += z * z / n; }
  for (let j = 0; j < d; j++) sd[j] = Math.sqrt(sd[j]) || 1;
  const A = new Float32Array(X.length), B = new Float32Array(Xt.length);
  for (let i = 0; i < n; i++) for (let j = 0; j < d; j++) A[i * d + j] = (X[i * d + j] - mu[j]) / sd[j];
  for (let i = 0; i < nt; i++) for (let j = 0; j < d; j++) B[i * d + j] = (Xt[i * d + j] - mu[j]) / sd[j];
  return [A, B];
}

class Adam {
  constructor(size, lr) { this.m = new Float32Array(size); this.v = new Float32Array(size); this.t = 0; this.lr = lr; }
  step(w, g) {
    this.t++; const b1 = 0.9, b2 = 0.999, c1 = 1 - b1 ** this.t, c2 = 1 - b2 ** this.t;
    for (let i = 0; i < w.length; i++) { this.m[i] = b1 * this.m[i] + (1 - b1) * g[i]; this.v[i] = b2 * this.v[i] + (1 - b2) * g[i] * g[i]; w[i] -= this.lr * (this.m[i] / c1) / (Math.sqrt(this.v[i] / c2) + 1e-8); }
  }
}

function train({ X, y, Xt, yt, d, k, kind, hidden = 64, epochs = 30, lr = 0.01, l2 = 1e-4, seed = 1, batch = 64 }) {
  const n = y.length, nt = yt.length, r = rng(seed);
  [X, Xt] = standardize(X, Xt, n, nt, d);
  const mlp = kind === 'mlp', h = mlp ? hidden : 0, din = mlp ? h : d;
  const W1 = mlp ? new Float32Array(d * h).map(() => gauss(r) * Math.sqrt(2 / d)) : null, b1 = mlp ? new Float32Array(h) : null;
  const W = new Float32Array(din * k).map(() => gauss(r) * 0.01), b = new Float32Array(k);
  const gW = new Float32Array(W.length), gb = new Float32Array(k), gW1 = mlp ? new Float32Array(W1.length) : null, gb1 = mlp ? new Float32Array(h) : null;
  const opt = [new Adam(W.length, lr), new Adam(k, lr)], opt1 = mlp ? [new Adam(W1.length, lr), new Adam(h, lr)] : null;
  const H = new Float32Array(Math.max(batch, 1) * Math.max(h, 1)), P = new Float32Array(k), dH = new Float32Array(Math.max(h, 1));

  function forward(Xa, i, hid, probs) {
    let inp = Xa, off = i * d;
    if (mlp) {
      for (let a = 0; a < h; a++) { let s = b1[a]; for (let j = 0; j < d; j++) s += Xa[off + j] * W1[j * h + a]; hid[a] = s > 0 ? s : 0; }
      inp = hid; off = 0;
    }
    let mx = -Infinity;
    for (let c = 0; c < k; c++) { let s = b[c]; for (let j = 0; j < din; j++) s += inp[off + j] * W[j * k + c]; probs[c] = s; if (s > mx) mx = s; }
    let z = 0; for (let c = 0; c < k; c++) { probs[c] = Math.exp(probs[c] - mx); z += probs[c]; }
    for (let c = 0; c < k; c++) probs[c] /= z;
    return inp === hid ? hid : null;
  }
  function accuracy(Xa, ya, m, keep) {
    let ok = 0; const hid = new Float32Array(Math.max(h, 1)), probs = new Float32Array(k), pred = keep ? new Int32Array(m) : null;
    for (let i = 0; i < m; i++) { forward(Xa, i, hid, probs); let best = 0; for (let c = 1; c < k; c++) if (probs[c] > probs[best]) best = c; if (best === ya[i]) ok++; if (pred) pred[i] = best; }
    return { acc: ok / m, pred };
  }

  const order = Array.from({ length: n }, (_, i) => i), curve = [];
  const hid = new Float32Array(Math.max(h, 1));
  for (let ep = 0; ep < epochs; ep++) {
    for (let i = n - 1; i > 0; i--) { const j = Math.floor(r() * (i + 1)); [order[i], order[j]] = [order[j], order[i]]; }
    for (let s = 0; s < n; s += batch) {
      const idx = order.slice(s, s + batch), m = idx.length;
      gW.fill(0); gb.fill(0); if (mlp) { gW1.fill(0); gb1.fill(0); }
      for (const i of idx) {
        const hv = forward(X, i, hid, P);
        P[y[i]] -= 1; // dL/dlogits for cross-entropy
        const inp = mlp ? hv : X, off = mlp ? 0 : i * d;
        for (let j = 0; j < din; j++) { const xj = inp[off + j]; if (xj === 0) continue; for (let c = 0; c < k; c++) gW[j * k + c] += xj * P[c]; }
        for (let c = 0; c < k; c++) gb[c] += P[c];
        if (mlp) {
          for (let a = 0; a < h; a++) { if (hv[a] <= 0) { dH[a] = 0; continue; } let s2 = 0; for (let c = 0; c < k; c++) s2 += W[a * k + c] * P[c]; dH[a] = s2; }
          for (let j = 0; j < d; j++) { const xj = X[i * d + j]; for (let a = 0; a < h; a++) gW1[j * h + a] += xj * dH[a]; }
          for (let a = 0; a < h; a++) gb1[a] += dH[a];
        }
      }
      for (let q = 0; q < gW.length; q++) gW[q] = gW[q] / m + l2 * W[q];
      for (let c = 0; c < k; c++) gb[c] /= m;
      opt[0].step(W, gW); opt[1].step(b, gb);
      if (mlp) { for (let q = 0; q < gW1.length; q++) gW1[q] = gW1[q] / m + l2 * W1[q]; for (let a = 0; a < h; a++) gb1[a] /= m; opt1[0].step(W1, gW1); opt1[1].step(b1, gb1); }
    }
    curve.push(accuracy(Xt, yt, nt, false).acc);
  }
  const test = accuracy(Xt, yt, nt, true);
  return { acc: test.acc, pred: test.pred, trainAcc: accuracy(X, y, n, false).acc, curve };
}

self.onmessage = (e) => {
  const { id } = e.data;
  try { const r = train(e.data); self.postMessage({ id, ...r }, [r.pred.buffer]); }
  catch (err) { self.postMessage({ id, error: String(err && err.message || err) }); }
};
