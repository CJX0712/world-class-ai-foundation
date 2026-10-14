// linalg.js — 世界级数值线性代数内核（零依赖，浏览器/Node 通用）
//
// 复用并站在世界顶级数值方法之上（LAPACK 级算法）：
//   - Householder 反射 QR 分解        (Golub & Van Loan, Matrix Computations)
//   - 经典 Jacobi 循环旋转 SVD/特征值 (Jacobi 1846 / modern cyclic sweep)
//   - Cholesky 分解                   (positive-definite  symmetrix)
// 每个算法均带可被独立实现交叉验证的数学不变量（重构误差 / 正交性 / 对称性）。
//
// 矩阵采用行主序二维数组：A[i] 为第 i 行。

// ---------- 基础矩阵运算 ----------
export const zeros = (r, c) => Array.from({ length: r }, () => new Array(c).fill(0));
export const identity = (n) => Array.from({ length: n }, (_, i) =>
  Array.from({ length: n }, (_, j) => (i === j ? 1 : 0))
);

export function transpose(A) {
  const r = A.length, c = A[0].length;
  const T = zeros(c, r);
  for (let i = 0; i < r; i++) for (let j = 0; j < c; j++) T[j][i] = A[i][j];
  return T;
}

export function matMul(A, B) {
  const m = A.length, k = A[0].length, n = B[0].length;
  const C = zeros(m, n);
  for (let i = 0; i < m; i++) {
    const Ai = A[i];
    for (let p = 0; p < k; p++) {
      const a = Ai[p];
      if (a === 0) continue;
      const Bp = B[p];
      const Ci = C[i];
      for (let j = 0; j < n; j++) Ci[j] += a * Bp[j];
    }
  }
  return C;
}

export function matVec(A, x) {
  const m = A.length, n = A[0].length;
  const y = new Array(m).fill(0);
  for (let i = 0; i < m; i++) {
    let s = 0;
    for (let j = 0; j < n; j++) s += A[i][j] * x[j];
    y[i] = s;
  }
  return y;
}

export const dot = (a, b) => {
  let s = 0;
  for (let i = 0; i < a.length; i++) s += a[i] * b[i];
  return s;
};
export const add = (a, b) => a.map((v, i) => v + b[i]);
export const sub = (a, b) => a.map((v, i) => v - b[i]);
export const scale = (a, s) => a.map((v) => v * s);

export function frobenius(A) {
  let s = 0;
  for (let i = 0; i < A.length; i++)
    for (let j = 0; j < A[0].length; j++) s += A[i][j] * A[i][j];
  return Math.sqrt(s);
}

// ---------- Jacobi 对称特征值分解 ----------
// 输入对称矩阵 S (n×n)，返回特征值 values 与特征向量矩阵 V（列即特征向量）。
// 经典 cyclic-by-row Jacobi 旋转，数值稳定，特征向量天然正交归一。
export function jacobiEigen(S, maxSweeps = 100, tol = 1e-14) {
  const n = S.length;
  const A = S.map((r) => r.slice());
  const V = identity(n);
  for (let sweep = 0; sweep < maxSweeps; sweep++) {
    let off = 0;
    for (let p = 0; p < n; p++)
      for (let q = p + 1; q < n; q++) off += A[p][q] * A[p][q];
    if (off < tol) break;
    for (let p = 0; p < n; p++) {
      for (let q = p + 1; q < n; q++) {
        if (Math.abs(A[p][q]) < 1e-300) continue;
        const app = A[p][p], aqq = A[q][q], apq = A[p][q];
        // 旋转角：消去 (p,q)
        const phi = 0.5 * Math.atan2(2 * apq, aqq - app);
        const c = Math.cos(phi), s = Math.sin(phi);
        // 旋转 A（仅相关行列）
        for (let i = 0; i < n; i++) {
          if (i === p || i === q) continue;
          const aip = A[i][p], aiq = A[i][q];
          A[i][p] = c * aip - s * aiq;
          A[i][q] = s * aip + c * aiq;
          A[p][i] = A[i][p];
          A[q][i] = A[i][q];
        }
        A[p][p] = c * c * app - 2 * s * c * apq + s * s * aqq;
        A[q][q] = s * s * app + 2 * s * c * apq + c * c * aqq;
        A[p][q] = 0;
        A[q][p] = 0;
        // 累积特征向量 V = V * J
        for (let i = 0; i < n; i++) {
          const vip = V[i][p], viq = V[i][q];
          V[i][p] = c * vip - s * viq;
          V[i][q] = s * vip + c * viq;
        }
      }
    }
  }
  const values = A.map((r, i) => r[i]);
  return { values, vectors: V };
}

// ---------- 奇异值分解 (thin SVD) ----------
// 通过 AᵀA 的 Jacobi 特征分解获得：A = U Σ Vᵀ。
// 返回 { U: m×n, S: [σ₁..σₙ], V: n×n }，满足 A ≈ U diag(S) Vᵀ，U、V 列正交。
export function svd(A) {
  const m = A.length, n = A[0].length;
  const At = transpose(A);
  const C = matMul(At, A); // n×n 对称半正定
  const { values, vectors: V } = jacobiEigen(C);
  const order = Array.from({ length: n }, (_, i) => i).sort(
    (a, b) => values[b] - values[a]
  );
  const S = order.map((i) => Math.sqrt(Math.max(values[i], 0)));
  // 重排 V 列
  const Vs = V.map((row) => order.map((j) => row[j]));
  // U 列 j = A v_j / σ_j
  const U = A.map((row) => {
    const u = new Array(n).fill(0);
    for (let j = 0; j < n; j++) {
      if (S[j] > 1e-12) {
        let s = 0;
        for (let i = 0; i < n; i++) s += row[i] * Vs[i][j];
        u[j] = s / S[j];
      }
    }
    return u;
  });
  return { U, S, V: Vs };
}

// ---------- Householder QR 分解 ----------
// A (m×n, m≥n) = Q (m×m 正交) R (m×n 上三角)。世界顶级 QR 算法。
export function qrHouseholder(A) {
  const m = A.length, n = A[0].length;
  const R = A.map((r) => r.slice());
  const Q = identity(m);
  for (let k = 0; k < n; k++) {
    const x = new Array(m - k);
    for (let i = k; i < m; i++) x[i - k] = R[i][k];
    const normx = Math.hypot(...x);
    if (normx < 1e-14) continue;
    const sign = x[0] < 0 ? -1 : 1;
    const v = x.slice();
    v[0] = x[0] + sign * normx;
    const vnorm = Math.hypot(...v);
    if (vnorm < 1e-14) continue;
    const vn = v.map((c) => c / vnorm); // 单位反射向量（子空间 k..m）
    // R = H R
    for (let j = k; j < n; j++) {
      let d = 0;
      for (let i = k; i < m; i++) d += vn[i - k] * R[i][j];
      for (let i = k; i < m; i++) R[i][j] -= 2 * vn[i - k] * d;
    }
    // Q = Q H （累积到列）
    for (let j = 0; j < m; j++) {
      let d = 0;
      for (let i = k; i < m; i++) d += vn[i - k] * Q[j][i];
      for (let i = k; i < m; i++) Q[j][i] -= 2 * vn[i - k] * d;
    }
  }
  for (let i = 0; i < m; i++)
    for (let j = 0; j < n; j++) if (i > j) R[i][j] = 0;
  return { Q, R };
}

// ---------- 最小二乘 ----------
// 解 min_x ‖A x − b‖₂，经 QR 得到数值稳定的正规方程解。
export function lstsq(A, b) {
  const { Q, R } = qrHouseholder(A);
  const m = A.length, n = A[0].length;
  const Qtb = new Array(n).fill(0);
  for (let i = 0; i < n; i++) {
    let s = 0;
    for (let j = 0; j < m; j++) s += Q[j][i] * b[j];
    Qtb[i] = s;
  }
  const x = new Array(n).fill(0);
  for (let i = n - 1; i >= 0; i--) {
    let s = Qtb[i];
    for (let j = i + 1; j < n; j++) s -= R[i][j] * x[j];
    x[i] = s / R[i][i];
  }
  return x;
}

// ---------- Cholesky 分解 ----------
// 对称正定 A = L Lᵀ，L 下三角。Golub & Van Loan Algorithm 4.2.1。
export function cholesky(A) {
  const n = A.length;
  const L = zeros(n, n);
  for (let i = 0; i < n; i++) {
    for (let j = 0; j <= i; j++) {
      let s = 0;
      for (let k = 0; k < j; k++) s += L[i][k] * L[j][k];
      if (i === j) L[i][j] = Math.sqrt(Math.max(A[i][i] - s, 1e-300));
      else L[i][j] = (A[i][j] - s) / L[j][j];
    }
  }
  return L;
}

// ---------- 数值工具 ----------
export function randn(seed) {
  // 确定性高斯（Box–Muller），便于可复现验证
  let s = seed >>> 0;
  const rng = () => {
    s = (s * 1664525 + 1013904223) >>> 0;
    return s / 4294967296;
  };
  return () => {
    let u = 0, v = 0;
    while (u === 0) u = rng();
    while (v === 0) v = rng();
    return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
  };
}
