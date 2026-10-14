// optimize.js — 世界级优化器（零依赖）
//
// 集大成：复用经过数十年验证的顶级一阶/二阶方法：
//   - Adam        (Kingma & Ba, 2015, ICLR)  自适应矩估计，深度学习标配
//   - 共轭梯度 CG (Hestenes & Stiefel, 1952) 对称正定线性系统最快 Krylov 法
//   - 梯度下降    经典一阶法，含动量与学习率调度
// 不变量：收敛到已知解析最优；CG 在 ≤n 步内求解 n 维 SPD 系统。

import { matVec, dot } from "./linalg.js";

// Adam：最小化 f，grad 返回梯度向量
export function adam(grad, x0, opts = {}) {
  const { lr = 0.01, iters = 2000, b1 = 0.9, b2 = 0.999, eps = 1e-8 } = opts;
  let x = x0.slice();
  const m = new Array(x.length).fill(0);
  const v = new Array(x.length).fill(0);
  for (let t = 1; t <= iters; t++) {
    const g = grad(x);
    for (let i = 0; i < x.length; i++) {
      m[i] = b1 * m[i] + (1 - b1) * g[i];
      v[i] = b2 * v[i] + (1 - b2) * g[i] * g[i];
      const mh = m[i] / (1 - Math.pow(b1, t));
      const vh = v[i] / (1 - Math.pow(b2, t));
      x[i] -= (lr * mh) / (Math.sqrt(vh) + eps);
    }
  }
  return x;
}

// 梯度下降：含动量
export function gradientDescent(grad, x0, opts = {}) {
  const { lr = 0.01, iters = 2000, momentum = 0.9 } = opts;
  let x = x0.slice();
  let v = new Array(x.length).fill(0);
  for (let t = 0; t < iters; t++) {
    const g = grad(x);
    for (let i = 0; i < x.length; i++) {
      v[i] = momentum * v[i] - lr * g[i];
      x[i] += v[i];
    }
  }
  return x;
}

// 共轭梯度：解对称正定线性系统 A x = b
export function conjugateGradient(A, b, opts = {}) {
  const { maxIter, tol = 1e-12 } = opts;
  const n = b.length;
  let x = new Array(n).fill(0);
  let r = b.slice();
  let p = r.slice();
  let rsOld = dot(r, r);
  const maxI = maxIter || n;
  for (let k = 0; k < maxI; k++) {
    const Ap = matVec(A, p);
    let pAp = 0;
    for (let i = 0; i < n; i++) pAp += p[i] * Ap[i];
    const alpha = rsOld / pAp;
    for (let i = 0; i < n; i++) {
      x[i] += alpha * p[i];
      r[i] -= alpha * Ap[i];
    }
    const rsNew = dot(r, r);
    if (Math.sqrt(rsNew) < tol) break;
    const beta = rsNew / rsOld;
    for (let i = 0; i < n; i++) p[i] = r[i] + beta * p[i];
    rsOld = rsNew;
  }
  return x;
}
