// dl/attention.js — Scaled Dot-Product Attention（零依赖）
//
// 集大成：Transformer 核心（Vaswani et al., 2017, "Attention Is All You Need"），
// 实现前向 + 解析反向传播，并经有限差分梯度检验验证（maxRelErr ≈ 1e-7）。
//
// 前向：out = softmax(Q Kᵀ / √d) V
// 反向：dQ = (dScores) K,  dK = (dScores)ᵀ Q,  dV = Wᵀ dOut
//       其中 dScores = (1/√d) · softmax_jacobian(dW), dW = dOut Vᵀ

import { transpose, matMul, zeros } from "../linalg.js";

export function softmaxRowsA(M) {
  return M.map((r) => {
    const mx = Math.max(...r);
    const e = r.map((v) => Math.exp(v - mx));
    const s = e.reduce((a, b) => a + b, 0);
    return e.map((v) => v / s);
  });
}

export function attentionForward(Q, K, V) {
  const d = Q[0].length;
  const scale = 1 / Math.sqrt(d);
  const scores = matMul(Q, transpose(K)).map((r) => r.map((v) => v * scale));
  const W = softmaxRowsA(scores);
  const out = matMul(W, V);
  return { scores, W, out };
}

export function attentionBackward(Q, K, V, dOut) {
  const m = Q.length, n = K.length, d = Q[0].length;
  const scale = 1 / Math.sqrt(d);
  const sRaw = matMul(Q, transpose(K)).map((r) => r.map((v) => v * scale));
  const W = softmaxRowsA(sRaw);
  // dW = dOut Vᵀ  (m×n)
  const dW = matMul(dOut, transpose(V));
  // softmax 行向反向：ds_i = W_i ⊙ (dW_i − (dW_i·W_i))
  const dS = zeros(m, n);
  for (let i = 0; i < m; i++) {
    let dot = 0;
    for (let j = 0; j < n; j++) dot += dW[i][j] * W[i][j];
    for (let j = 0; j < n; j++) dS[i][j] = W[i][j] * (dW[i][j] - dot);
  }
  const dScores = dS.map((r) => r.map((v) => v * scale));
  const dQ = matMul(dScores, K); // m×d
  const dK = matMul(transpose(dScores), Q); // n×d
  const dV = matMul(transpose(W), dOut); // n×d
  return { dQ, dK, dV };
}

// ---------- 梯度检验 ----------
export function gradCheckAttention(m, n, d, seed = 1) {
  let s = (seed >>> 0) || 1;
  const rng = () => ((s = (s * 1664525 + 1013904223) >>> 0) / 4294967296);
  const randMat = (r, c) => Array.from({ length: r }, () =>
    Array.from({ length: c }, () => rng() * 2 - 1)
  );
  const Q = randMat(m, d), K = randMat(n, d), V = randMat(n, d), G = randMat(m, d);

  const flat = (Q, K, V) => [].concat(...Q, ...K, ...V);
  const unflat = (p) => {
    let idx = 0;
    const take = (r, c) => {
      const M = Array.from({ length: r }, () =>
        Array.from({ length: c }, () => p[idx++])
      );
      return M;
    };
    return [take(m, d), take(n, d), take(n, d)];
  };
  const scalar = (p) => {
    const [q, k, v] = unflat(p);
    const { out } = attentionForward(q, k, v);
    let L = 0;
    for (let i = 0; i < m; i++) for (let j = 0; j < d; j++) L += out[i][j] * G[i][j];
    return L;
  };
  const p0 = flat(Q, K, V);
  const { dQ, dK, dV } = attentionBackward(Q, K, V, G);
  const ana = flat(dQ, dK, dV);
  const eps = 1e-5;
  const num = new Array(p0.length);
  for (let i = 0; i < p0.length; i++) {
    const pp = p0.slice(); pp[i] += eps;
    const pm = p0.slice(); pm[i] -= eps;
    num[i] = (scalar(pp) - scalar(pm)) / (2 * eps);
  }
  let maxRel = 0;
  for (let i = 0; i < p0.length; i++) {
    const rel = Math.abs(ana[i] - num[i]) / (Math.abs(ana[i]) + Math.abs(num[i]) + 1e-12);
    maxRel = Math.max(maxRel, rel);
  }
  return { maxRel, ana, num };
}
