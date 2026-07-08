"""统计有效样本（非 ERROR）的 EX/EM"""
import json
import sys
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent.parent
filepath = sys.argv[1] if len(sys.argv) > 1 else str(PROJECT_DIR / "outcome" / "eval_results_glm_3shot.json")

with open(filepath, "r", encoding="utf-8") as f:
    d = json.load(f)

results = d.get("results", d)
valid = [r for r in results if not r["full_response"].startswith("ERROR")]
errors = len(results) - len(valid)

ex = sum(1 for r in valid if r["execution_accuracy"])
em = sum(1 for r in valid if r["exact_match"])

print(f"Total: {len(results)}, Valid: {len(valid)}, Errors: {errors}")
if valid:
    print(f"Valid EX: {ex}/{len(valid)} = {ex/len(valid)*100:.2f}%")
    print(f"Valid EM: {em}/{len(valid)} = {em/len(valid)*100:.2f}%")
