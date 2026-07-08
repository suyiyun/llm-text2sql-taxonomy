"""
检查 GLM 结果：按样本 ID 排序后，准确率是否还单调递减。
如果是并发完成顺序导致的假象，按 ID 排序后曲线会不同。
"""
import json
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent.parent
filepath = PROJECT_DIR / "outcome" / "eval_results_glm_3shot.json"

with open(filepath, "r", encoding="utf-8") as f:
    d = json.load(f)

results = d.get("results", d)

# 按 id 排序
sorted_results = sorted(results, key=lambda r: r["id"])

# 每 200 条计算一次 EX，看是否单调
print("按样本 ID 排序后的分段 EX 准确率：")
print("-" * 50)

for start in range(0, len(sorted_results), 200):
    end = min(start + 200, len(sorted_results))
    batch = sorted_results[start:end]
    ex = sum(1 for r in batch if r["execution_accuracy"])
    acc = ex / len(batch) * 100
    ids = f"{batch[0]['id']}-{batch[-1]['id']}"
    print(f"  ID {ids:>15}  EX={ex:>3}/{len(batch)} = {acc:.1f}%")

# 总体
total_ex = sum(1 for r in results if r["execution_accuracy"])
total_em = sum(1 for r in results if r["exact_match"])
print("-" * 50)
print(f"总体 EX={total_ex}/{len(results)} = {total_ex/len(results)*100:.2f}%")
print(f"总体 EM={total_em}/{len(results)} = {total_em/len(results)*100:.2f}%")

# 检查结果顺序：文件中前10条的id是否连续
print("\n文件中前10条结果的 id 顺序（完成顺序）：")
for r in results[:10]:
    print(f"  id={r['id']}, db={r['db_id']}, EX={r['execution_accuracy']}")

print("\n按 id 排序后前10条：")
for r in sorted_results[:10]:
    print(f"  id={r['id']}, db={r['db_id']}, EX={r['execution_accuracy']}")
