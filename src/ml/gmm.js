// ml/gmm.js — 高斯混合模型 EM（零依赖）
//
// 集大成：期望最大化（Dempster, Laird & Rubin, 1977），经 Cholesky 稳定计算
// 多元高斯（log-det 与二次型），避免指数溢出（logsumexp 思想）。
//
// 不变量：
//   责任度每行和 = 1
//   混合权重和 = 1
//   协方差对称半正定
//   对数似然单调不减（EM 定理）

import { randn, cholesky, matMul, transpose, zeros, identity } from "../linalg.js";
import { kmeans } from "./kmeans.js";

// 多元高斯对数密度（经 Cholesky 稳定）：log N(x|μ,Σ)
function logGaussian(x, mu, L, logdet) {
  const d = x.length;
  const y = new Array(d);
  for (let i = 0; i < d; i++) {
    let s = x[i] - mu[i];
    for (let j = 0; j < i; j++) s -= L[i][j] * y[j];
    y[i] = s / L[i][i];
  }
  let q = 0;
  for (let i = 0; i < d; i++) q += y[i] * y[i];
  return -0.5 * (d * Math.log(2 * Math.PI) + logdet + q);
}

export function gmm(X, k, opts = {}) {
  const { maxIter = 100, seed = 42, reg = 1e-6 } = opts;
  const m = X.length, d = X[0].length;
  let s = (seed >>> 0) || 7;
  const rng = () => ((s = (s * 1664525 + 1013904223) >>> 0) / 4294967296);

  // 用 k-means 初始化均值，权重均匀
  const { centers } = kmeans(X, k, { seed, maxIter: 20 });
  const weights = new Array(k).fill(1 / k);
  const means = centers.map((c) => c.slice());
  // 协方差用全局方差
  const muAll = new Array(d).fill(0);
  for (let i = 0; i < m; i++) for (let j = 0; j < d; j++) muAll[j] += X[i][j];
  for (let j = 0; j < d; j++) muAll[j] /= m;
  let varAll = 0;
  for (let i = 0; i < m; i++)
    for (let j = 0; j < d; j++) varAll += (X[i][j] - muAll[j]) ** 2;
  varAll = varAll / (m * d) + reg;
  const covs = means.map(() => {
    const C = identity(d);
    for (let i = 0; i < d; i++) C[i][i] = varAll;
    return C;
  });

  const llHist = [];
  let prevLL = -Infinity;

  for (let iter = 0; iter < maxIter; iter++) {
    // 预计算每分量 Cholesky 与 logdet
    const Ls = [], logdets = [];
    for (let c = 0; c < k; c++) {
      // 加正则到对角以保证 SPD
      const C = covs[c].map((row, i) => row.map((v, j) => (i === j ? v + reg : v)));
      const L = cholesky(C);
      let ld = 0;
      for (let i = 0; i < d; i++) ld += 2 * Math.log(L[i][i]);
      Ls.push(L);
      logdets.push(ld);
    }
    // E-step：责任度（logsumexp 稳定）
    const logR = zeros(m, k);
    for (let i = 0; i < m; i++) {
      for (let c = 0; c < k; c++)
        logR[i][c] = Math.log(weights[c] + 1e-300) + logGaussian(X[i], means[c], Ls[c], logdets[c]);
    }
    const resp = zeros(m, k);
    let ll = 0;
    for (let i = 0; i < m; i++) {
      const mx = Math.max(...logR[i]);
      let sum = 0;
      for (let c = 0; c < k; c++) sum += Math.exp(logR[i][c] - mx);
      const lse = mx + Math.log(sum);
      ll += lse;
      for (let c = 0; c < k; c++) resp[i][c] = Math.exp(logR[i][c] - lse);
    }
    llHist.push(ll);
    // M-step
    const Nk = new Array(k).fill(0);
    for (let i = 0; i < m; i++) for (let c = 0; c < k; c++) Nk[c] += resp[i][c];
    for (let c = 0; c < k; c++) weights[c] = Nk[c] / m;
    for (let c = 0; c < k; c++) {
      const newMu = new Array(d).fill(0);
      for (let i = 0; i < m; i++) for (let j = 0; j < d; j++) newMu[j] += resp[i][c] * X[i][j];
      for (let j = 0; j < d; j++) newMu[j] /= Nk[c] || 1e-12;
      means[c] = newMu;
      const newC = zeros(d, d);
      for (let i = 0; i < m; i++) {
        for (let p = 0; p < d; p++) {
          const dp = X[i][p] - newMu[p];
          for (let q = 0; q < d; q++) newC[p][q] += resp[i][c] * dp * (X[i][q] - newMu[q]);
        }
      }
      for (let p = 0; p < d; p++) for (let q = 0; q < d; q++) newC[p][q] /= Nk[c] || 1e-12;
      covs[c] = newC;
    }
    if (ll - prevLL < 1e-9) { prevLL = ll; break; }
    prevLL = ll;
  }

  // 最终协方差 SPD 校验
  const spdCheck = covs.map((C) => {
    const Cc = C.map((row, i) => row.map((v, j) => (i === j ? v + reg : v)));
    try { cholesky(Cc); return true; } catch { return false; }
  });
  return { weights, means, covs, logLikelihood: prevLL, llHist, spdCheck };
}
