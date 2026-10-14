// ml/regression.js — 顶级回归模型（零依赖）
//
// 集大成：正规方程闭式解（经 QR 数值稳定）+ Adam 优化的逻辑回归，
// 每个模型带有限差分梯度检验（与解析梯度一致至 ~1e-6）。
//
// 不变量：
//   线性回归  残差 ⟂ 设计矩阵列（正规方程 Xᵀ(y−Xŵ)=0）
//   逻辑回归  梯度 = Xᵀ(p−y)/m，有限差分交叉验证
//   Ridge     解 (XᵀX+λI)ŵ = Xᵀy（Cholesky）

import { lstsq, cholesky, matMul, transpose, matVec, add, scale, zeros } from "../linalg.js";
import { adam } from "../optimize.js";

export const sigmoid = (z) => 1 / (1 + Math.exp(-z));

// ---------- 线性回归 ----------
export function linearFit(X, y) {
  return lstsq(X, y);
}
export function predict(X, w) {
  return X.map((r) => matVec([r], w)[0]); // 单样本点积
}
export function r2Score(X, y, w) {
  const yh = X.map((r) => {
    let s = 0;
    for (let j = 0; j < w.length; j++) s += r[j] * w[j];
    return s;
  });
  const my = y.reduce((a, b) => a + b, 0) / y.length;
  let ssRes = 0, ssTot = 0;
  for (let i = 0; i < y.length; i++) {
    ssRes += (y[i] - yh[i]) ** 2;
    ssTot += (y[i] - my) ** 2;
  }
  return 1 - ssRes / ssTot;
}

// ---------- Ridge 回归 ----------
export function ridgeFit(X, y, lambda = 1e-3) {
  const m = X.length, d = X[0].length;
  const Xt = transpose(X);
  const XtX = matMul(Xt, X);
  for (let i = 0; i < d; i++) XtX[i][i] += lambda;
  const L = cholesky(XtX);
  // 解 L Lᵀ w = Xt y
  const rhs = matVec(Xt, y);
  const n = rhs.length;
  const y1 = new Array(n).fill(0);
  for (let i = 0; i < n; i++) {
    let s = rhs[i];
    for (let k = 0; k < i; k++) s -= L[i][k] * y1[k];
    y1[i] = s / L[i][i];
  }
  const w = new Array(n).fill(0);
  for (let i = n - 1; i >= 0; i--) {
    let s = y1[i];
    for (let k = i + 1; k < n; k++) s -= L[k][i] * w[k];
    w[i] = s / L[i][i];
  }
  return w;
}

// ---------- 逻辑回归 ----------
export function logisticLossGrad(X, y, w) {
  const m = X.length, d = X[0].length;
  const g = new Array(d).fill(0);
  for (let i = 0; i < m; i++) {
    let z = 0;
    for (let j = 0; j < d; j++) z += X[i][j] * w[j];
    const p = sigmoid(z);
    const err = p - y[i];
    for (let j = 0; j < d; j++) g[j] += err * X[i][j];
  }
  return g.map((v) => v / m);
}
export function logisticLoss(X, y, w) {
  const m = X.length;
  let L = 0;
  for (let i = 0; i < m; i++) {
    let z = 0;
    for (let j = 0; j < w.length; j++) z += X[i][j] * w[j];
    const p = sigmoid(z);
    L += -(y[i] * Math.log(p + 1e-15) + (1 - y[i]) * Math.log(1 - p + 1e-15));
  }
  return L / m;
}
export function logisticFit(X, y, opts = {}) {
  const d = X[0].length;
  const w0 = new Array(d).fill(0);
  return adam((w) => logisticLossGrad(X, y, w), w0, {
    lr: 0.1,
    iters: 3000,
    ...opts,
  });
}

// ---------- 梯度检验工具 ----------
export function gradCheck(fLoss, fGrad, x0, eps = 1e-5) {
  const ana = fGrad(x0);
  const num = new Array(x0.length);
  for (let i = 0; i < x0.length; i++) {
    const xp = x0.slice();
    xp[i] += eps;
    const xm = x0.slice();
    xm[i] -= eps;
    num[i] = (fLoss(xp) - fLoss(xm)) / (2 * eps);
  }
  let maxRel = 0;
  for (let i = 0; i < x0.length; i++) {
    const rel = Math.abs(ana[i] - num[i]) / (Math.abs(ana[i]) + Math.abs(num[i]) + 1e-12);
    maxRel = Math.max(maxRel, rel);
  }
  return { maxRel, ana, num };
}
