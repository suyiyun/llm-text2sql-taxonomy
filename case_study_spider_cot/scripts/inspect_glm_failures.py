"""
检查 ID>=9859 的样本，GLM 返回了什么。
"""
import json
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent.parent
filepath = PROJECT_DIR / "outcome" / "eval_results_glm_3shot.json"

with open(filepath, "r", encoding="utf-8") as f:
    d = json.load(f)

results = d.get("results", d)
failures = [r for r in results if r["id"] >= 9859][:5]

print("ID>=9859 的前5条样本的原始响应：")
print("=" * 80)
for r in failures:
    print(f"\nid={r['id']} db={r['db_id']}")
    print(f"Q: {r['question'][:60]}...")
    print(f"Gold: {r['gold_sql'][:80]}")
    print(f"Pred: {r['pred_sql'][:120]}")
    print(f"Resp (first 300 chars): {r['full_response'][:300]}")
    print("-" * 40)
