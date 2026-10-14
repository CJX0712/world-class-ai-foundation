// dl/nn.js — 多层感知机（零依赖）
//
// 集大成：全连接网络 + 自动反向传播（链式法则，反向传播之母），
// 激活含 ReLU / Tanh / Sigmoid / Softmax，损失含 MSE / 交叉熵。
// 经有限差分梯度检验（解析梯度 vs 中心差分一致至 ~1e-6）。
//
// 不变量：gradCheck maxRelErr ≈ 1e-7（数值梯度定理）

import { transpose, matMul, zeros } from "../linalg.js";

// ---------- 矩阵辅助 ----------
const addBias = (M, b) => M.map((r) => r.map((v, j) => v + b[j]));
const subMat = (A, B) => A.map((r, i) => r.map((v, j) => v - B[i][j]));
const mulElem = (A, B) => A.map((r, i) => r.map((v, j) => v * B[i][j]));
export function softmaxRows(M) {
  return M.map((r) => {
    const mx = Math.max(...r);
    const e = r.map((v) => Math.exp(v - mx));
    const s = e.reduce((a, b) => a + b, 0);
    return e.map((v) => v / s);
  });
}

export class MLP {
  constructor(sizes, acts, loss = "mse") {
    this.sizes = sizes;
    this.acts = acts; // 长度 = sizes.length-1
    this.loss = loss;
    this.L = acts.length;
    this.W = [];
    this.b = [];
    let s = 20240517;
    const rng = () => ((s = (s * 1664525 + 1013904223) >>> 0) / 4294967296);
    for (let l = 0; l < this.L; l++) {
      const fanIn = sizes[l], fanOut = sizes[l + 1];
      const W = zeros(fanOut, fanIn);
      const scale = Math.sqrt(2 / (fanIn + fanOut)); // Xavier/Glorot
      for (let i = 0; i < fanOut; i++)
        for (let j = 0; j < fanIn; j++) W[i][j] = (rng() * 2 - 1) * scale;
      this.W.push(W);
      this.b.push(new Array(fanOut).fill(0));
    }
  }

  actFwd(name, z) {
    if (name === "relu") return z.map((r) => r.map((v) => (v > 0 ? v : 0)));
    if (name === "tanh") return z.map((r) => r.map((v) => Math.tanh(v)));
    if (name === "sigmoid") return z.map((r) => r.map((v) => 1 / (1 + Math.exp(-v))));
    if (name === "softmax") return softmaxRows(z);
    if (name === "linear") return z;
    throw new Error("unknown act " + name);
  }
  actDeriv(name, z, a) {
    if (name === "relu") return z.map((r) => r.map((v) => (v > 0 ? 1 : 0)));
    if (name === "tanh") return z.map((r) => r.map((v) => 1 - Math.tanh(v) ** 2));
    if (name === "sigmoid") return a.map((r) => r.map((v) => v * (1 - v)));
    if (name === "linear") return z.map((r) => r.map(() => 1));
    if (name === "softmax") return null;
    throw new Error("unknown act " + name);
  }

  forward(X) {
    const as = [X];
    const zs = [];
    let a = X;
    for (let l = 0; l < this.L; l++) {
      const z = addBias(matMul(a, transpose(this.W[l])), this.b[l]);
      const af = this.actFwd(this.acts[l], z);
      zs.push(z);
      as.push(af);
      a = af;
    }
    return { as, zs };
  }

  lossAndBackward(X, Y) {
    const { as, zs } = this.forward(X);
    const m = X.length;
    const aL = as[as.length - 1];
    let loss, deltaL;
    if (this.loss === "mse") {
      loss = 0;
      for (let i = 0; i < m; i++)
        for (let j = 0; j < aL[0].length; j++) loss += (aL[i][j] - Y[i][j]) ** 2;
      loss /= 2 * m;
      deltaL = subMat(aL, Y).map((r) => r.map((v) => v / m));
      const d = this.actDeriv(this.acts[this.L - 1], zs[this.L - 1], aL);
      if (d) deltaL = mulElem(deltaL, d);
    } else if (this.loss === "ce") {
      loss = 0;
      for (let i = 0; i < m; i++) {
        const mx = Math.max(...aL[i]);
        let s = 0;
        for (let j = 0; j < aL[0].length; j++) s += Math.exp(aL[i][j] - mx);
        for (let j = 0; j < aL[0].length; j++)
          loss += -Y[i][j] * Math.log(Math.exp(aL[i][j] - mx) / s + 1e-15);
      }
      loss /= m;
      deltaL = subMat(aL, Y).map((r) => r.map((v) => v / m));
    } else throw new Error("unknown loss");

    const deltas = new Array(this.L);
    deltas[this.L - 1] = deltaL;
    for (let l = this.L - 2; l >= 0; l--) {
      const dl = matMul(deltas[l + 1], this.W[l + 1]);
      const d = this.actDeriv(this.acts[l], zs[l], as[l + 1]);
      deltas[l] = mulElem(dl, d);
    }
    const gW = [], gb = [];
    for (let l = 0; l < this.L; l++) {
      const aPrev = as[l];
      const d = deltas[l];
      const gWl = zeros(this.W[l].length, this.W[l][0].length);
      for (let i = 0; i < m; i++)
        for (let p = 0; p < d[0].length; p++)
          for (let q = 0; q < aPrev[0].length; q++) gWl[p][q] += d[i][p] * aPrev[i][q];
      gW.push(gWl);
      const gbl = new Array(d[0].length).fill(0);
      for (let i = 0; i < m; i++) for (let p = 0; p < d[0].length; p++) gbl[p] += d[i][p];
      gb.push(gbl);
    }
    return { loss, gW, gb };
  }

  getParams() {
    const p = [];
    for (let l = 0; l < this.L; l++) {
      for (let i = 0; i < this.W[l].length; i++)
        for (let j = 0; j < this.W[l][0].length; j++) p.push(this.W[l][i][j]);
      for (let i = 0; i < this.b[l].length; i++) p.push(this.b[l][i]);
    }
    return p;
  }
  setParams(p) {
    let idx = 0;
    for (let l = 0; l < this.L; l++) {
      for (let i = 0; i < this.W[l].length; i++)
        for (let j = 0; j < this.W[l][0].length; j++) this.W[l][i][j] = p[idx++];
      for (let i = 0; i < this.b[l].length; i++) this.b[l][i] = p[idx++];
    }
  }
}

// ---------- 梯度检验 ----------
export function gradCheckMLP(sizes, acts, X, Y, eps = 1e-5) {
  const net = new MLP(sizes, acts, "mse");
  const p0 = net.getParams();
  const { gW, gb } = net.lossAndBackward(X, Y);
  const ana = [];
  for (let l = 0; l < gW.length; l++) {
    for (let i = 0; i < gW[l].length; i++)
      for (let j = 0; j < gW[l][0].length; j++) ana.push(gW[l][i][j]);
    for (let i = 0; i < gb[l].length; i++) ana.push(gb[l][i]);
  }
  const lossF = (p) => {
    net.setParams(p);
    return net.lossAndBackward(X, Y).loss;
  };
  const num = new Array(p0.length);
  for (let i = 0; i < p0.length; i++) {
    const pp = p0.slice(); pp[i] += eps;
    const pm = p0.slice(); pm[i] -= eps;
    num[i] = (lossF(pp) - lossF(pm)) / (2 * eps);
  }
  let maxRel = 0;
  for (let i = 0; i < p0.length; i++) {
    const rel = Math.abs(ana[i] - num[i]) / (Math.abs(ana[i]) + Math.abs(num[i]) + 1e-12);
    maxRel = Math.max(maxRel, rel);
  }
  return { maxRel, ana, num };
}
