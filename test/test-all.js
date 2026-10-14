// test/test-all.js — 运行全部数学不变量自检
import { runAllTests } from "../src/index.js";

const { total, passed, results } = runAllTests();
for (const r of results) {
  console.log(`${r.passed ? "✅ PASS" : "❌ FAIL"}  ${r.name}\n        └─ ${r.metric}`);
}
console.log(`\n=== 世界级 AI 引擎自检：${passed}/${total} 全绿 ===`);
process.exit(passed === total ? 0 : 1);
