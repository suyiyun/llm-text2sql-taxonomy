import json
import sys

filepath = sys.argv[1] if len(sys.argv) > 1 else "outcome/eval_results_openai_3shot.json"

with open(filepath, "r", encoding="utf-8") as f:
    d = json.load(f)

results = d.get("results", d)
total = len(results)
ex = sum(1 for r in results if r["execution_accuracy"])
em = sum(1 for r in results if r["exact_match"])
errors = sum(1 for r in results if r["full_response"].startswith("ERROR"))

print(f"Total: {total}, Errors: {errors}, Valid: {total - errors}")
print(f"EX: {ex}/{total} = {ex/total*100:.2f}%")
print(f"EM: {em}/{total} = {em/total*100:.2f}%")
