// ml/kmeans.js — K-Means 聚类（零依赖）
//
// 集大成：k-means++ 初始化（Arthur & Vassilvitskii, 2007, SODA）——理论保证
// 期望代价 ≤ O(log k) 近似比；Lloyd 迭代单调下降 SSE。
//
// 不变量：每次迭代簇内平方和（inertia）单调不增。

import { randn } from "../linalg.js";

export function kmeans(X, k, opts = {}) {
  const { maxIter = 100, seed = 42, tol = 1e-9 } = opts;
  const m = X.length, d = X[0].length;
  let s = (seed >>> 0) || 1;
  const rng = () => ((s = (s * 1664525 + 1013904223) >>> 0) / 4294967296);

  // k-means++ 初始化
  const centers = [];
  centers.push(X[Math.floor(rng() * m)].slice());
  const dist2 = X.map((x) => Math.hypot(...x.map((v, j) => v - centers[0][j])) ** 2);
  for (let c = 1; c < k; c++) {
    let sum = dist2.reduce((a, b) => a + b, 0);
    let r = rng() * sum;
    let idx = 0;
    for (let i = 0; i < m; i++) {
      r -= dist2[i];
      if (r <= 0) { idx = i; break; }
    }
    centers.push(X[idx].slice());
    for (let i = 0; i < m; i++) {
      const dd = Math.hypot(...X[i].map((v, j) => v - centers[c][j])) ** 2;
      if (dd < dist2[i]) dist2[i] = dd;
    }
  }

  const assign = new Array(m).fill(0);
  const inertiaHist = [];
  let prevInertia = Infinity;

  for (let iter = 0; iter < maxIter; iter++) {
    // 分配
    let inertia = 0;
    for (let i = 0; i < m; i++) {
      let best = 0, bd = Infinity;
      for (let c = 0; c < k; c++) {
        const dd = Math.hypot(...X[i].map((v, j) => v - centers[c][j])) ** 2;
        if (dd < bd) { bd = dd; best = c; }
      }
      assign[i] = best;
      inertia += bd;
    }
    inertiaHist.push(inertia);
    // 更新中心
    const acc = Array.from({ length: k }, () => new Array(d).fill(0));
    const cnt = new Array(k).fill(0);
    for (let i = 0; i < m; i++) {
      const c = assign[i];
      cnt[c]++;
      for (let j = 0; j < d; j++) acc[c][j] += X[i][j];
    }
    for (let c = 0; c < k; c++)
      if (cnt[c] > 0) centers[c] = acc[c].map((v) => v / cnt[c]);
    if (Math.abs(prevInertia - inertia) < tol) { prevInertia = inertia; break; }
    prevInertia = inertia;
  }
  return { centers, assign, inertia: prevInertia, inertiaHist };
}
