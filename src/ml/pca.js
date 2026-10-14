// ml/pca.js — 主成分分析（零依赖）
//
// 集大成：协方差矩阵经 Jacobi 特征分解（同 linalg，世界顶级数值法）。
//
// 不变量：
//   协方差矩阵对称且半正定
//   特征向量彼此正交归一（VᵀV = I）
//   重构误差在截断秩-k 下最优（Eckart–Young）
//   解释方差比单调递增

import { jacobiEigen, transpose, matMul, zeros, frobenius } from "../linalg.js";

export function colMean(X) {
  const m = X.length, d = X[0].length;
  const mu = new Array(d).fill(0);
  for (let i = 0; i < m; i++) for (let j = 0; j < d; j++) mu[j] += X[i][j];
  return mu.map((v) => v / m);
}

export function pca(X, nComponents) {
  const m = X.length, d = X[0].length;
  const mu = colMean(X);
  const Xc = X.map((r) => r.map((v, j) => v - mu[j]));
  // 协方差（无偏 1/(m-1)）
  const Xt = transpose(Xc);
  const C = matMul(Xt, Xc).map((row) => row.map((v) => v / (m - 1)));
  const { values, vectors: V } = jacobiEigen(C);
  const order = Array.from({ length: d }, (_, i) => i).sort(
    (a, b) => values[b] - values[a]
  );
  const k = nComponents || d;
  const comps = order.slice(0, k).map((idx) => V.map((row) => row[idx])); // k×d 行即主成分
  const explained = order.map((idx) => Math.max(values[idx], 0));
  const total = explained.reduce((a, b) => a + b, 0);
  const explainedRatio = explained.map((v) => v / total);

  function transform(Xin) {
    const centered = Xin.map((r) => r.map((v, j) => v - mu[j]));
    return centered.map((r) =>
      comps.map((c) => {
        let s = 0;
        for (let j = 0; j < d; j++) s += r[j] * c[j];
        return s;
      })
    );
  }
  function reconstruct(Z) {
    return Z.map((z) => {
      const r = mu.slice();
      for (let a = 0; a < k; a++)
        for (let j = 0; j < d; j++) r[j] += z[a] * comps[a][j];
      return r;
    });
  }
  return { mu, components: comps, explainedVariance: explained, explainedRatio, transform, reconstruct };
}
