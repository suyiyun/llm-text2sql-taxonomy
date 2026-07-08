import json
import sys

filepath = sys.argv[1] if len(sys.argv) > 1 else "outcome/eval_results_openai_3shot.json"

with open(filepath, "r", encoding="utf-8") as f:
    d = json.load(f)

results = d.get("results", d)
for r in results[:5]:
    print(f"Q: {r['question'][:60]}")
    print(f"Gold: {r['gold_sql'][:80]}")
    print(f"Pred: {r['pred_sql'][:80]}")
    print(f"Resp: {r['full_response'][:200]}")
    print(f"EX={r['execution_accuracy']} EM={r['exact_match']}")
    print("---")
