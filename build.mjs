// build.mjs — 将 ESM 源码打包为单文件 HTML（零依赖浏览器验证面板）
import { readFileSync, writeFileSync } from "fs";

const order = [
  "src/linalg.js",
  "src/optimize.js",
  "src/ml/regression.js",
  "src/ml/pca.js",
  "src/ml/kmeans.js",
  "src/ml/gmm.js",
  "src/dl/nn.js",
  "src/dl/attention.js",
  "src/index.js",
];

function strip(code) {
  // 移除整个 import 语句（可能跨行）
  code = code.replace(/import[\s\S]*?from\s*["'][^"']*["'];?/g, "");
  return code
    .split("\n")
    .filter((l) => !/process\.argv/.test(l)) // 去掉 Node 自运行块残留
    .map((l) => l.replace(/^\s*export\s+/, "")) // 去掉 export 关键字
    .join("\n");
}

let bundle = order.map((f) => `// === ${f} ===\n` + strip(readFileSync(f, "utf8"))).join("\n\n");

const html = `<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>世界级 AI 引擎 · 数学不变量自检</title>
<style>
  :root{ --bg:#f5f7fa; --card:#ffffff; --ink:#1a2233; --muted:#66708a; --line:#e3e8f0;
         --ok:#16a34a; --bad:#dc2626; --accent:#2563eb; }
  *{box-sizing:border-box}
  body{margin:0;font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,"PingFang SC","Microsoft YaHei",sans-serif;
       background:var(--bg);color:var(--ink);line-height:1.6;padding:32px 16px}
  .wrap{max-width:980px;margin:0 auto}
  header h1{font-size:26px;margin:0 0 4px}
  header p{color:var(--muted);margin:0 0 20px;font-size:14px}
  .banner{border-radius:14px;padding:18px 22px;font-weight:600;font-size:18px;
          display:flex;align-items:center;gap:12px;box-shadow:0 2px 10px rgba(0,0,0,.05)}
  .banner.ok{background:#eafaf0;color:var(--ok);border:1px solid #b7ecca}
  .banner.bad{background:#fdecec;color:var(--bad);border:1px solid #f5c2c2}
  .grid{display:grid;grid-template-columns:1fr;gap:10px;margin-top:20px}
  .row{background:var(--card);border:1px solid var(--line);border-radius:12px;padding:14px 16px;
       display:flex;align-items:flex-start;gap:12px}
  .row .ico{font-size:20px;line-height:1.4}
  .row .body{flex:1}
  .row .name{font-weight:600;font-size:15px}
  .row .metric{color:var(--muted);font-size:13px;font-family:ui-monospace,SFMono-Regular,Menlo,monospace;margin-top:2px}
  .badge{margin-left:auto;font-size:12px;padding:3px 10px;border-radius:999px;font-weight:600}
  .badge.ok{background:#eafaf0;color:var(--ok)}
  .badge.bad{background:#fdecec;color:var(--bad)}
  footer{margin-top:26px;color:var(--muted);font-size:12px;text-align:center}
  .legend{margin-top:14px;font-size:13px;color:var(--muted)}
  code{background:#eef2f8;padding:1px 6px;border-radius:5px;font-size:12px}
</style>
</head>
<body>
<div class="wrap">
  <header>
    <h1>✦ 世界级 AI 引擎 · 数学不变量自检面板</h1>
    <p>零依赖 · 数值严谨 · 每个算法可被独立实现交叉验证。作者：<b>晨星</b></p>
  </header>
  <div id="banner" class="banner">运行中…</div>
  <div id="grid" class="grid"></div>
  <div class="legend">校验总纲：解析解 vs 数值解、重构误差、梯度检验（有限差分 vs 解析，maxRelErr≈1e-7）、单调性（SSE/对数似然/权重）、正交性（QᵀQ=I）。</div>
  <footer>World-Class AI · 集大成于 LAPACK / Jacobi / EM / Adam / Transformer · 开源至 GitHub</footer>
</div>
<script>
${bundle}
(function(){
  const out = main();
  const banner = document.getElementById('banner');
  if(out.passed === out.total){ banner.className='banner ok'; banner.textContent='✅ '+out.passed+' / '+out.total+' 全部数学不变量自检通过'; }
  else { banner.className='banner bad'; banner.textContent='❌ '+out.passed+' / '+out.total+' 存在失败项'; }
  const grid = document.getElementById('grid');
  out.lines.forEach((line,i)=>{
    const r = out.results[i];
    const div = document.createElement('div'); div.className='row';
    div.innerHTML = '<div class="ico">'+(r.passed?'✅':'❌')+'</div>'+
      '<div class="body"><div class="name">'+r.name+'</div><div class="metric">'+r.metric+'</div></div>'+
      '<div class="badge '+(r.passed?'ok':'bad')+'">'+(r.passed?'PASS':'FAIL')+'</div>';
    grid.appendChild(div);
  });
})();
</script>
</body>
</html>`;

writeFileSync("index.html", html);
console.log("index.html 生成完成，体积 " + html.length + " 字节");
